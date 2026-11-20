# %% [markdown]
# ## kan.py — KANLinear: B-spline Kolmogorov-Arnold Network layer
#
# A KAN layer replaces the linear dot-product of a standard MLP layer with a
# sum of learnable univariate spline functions:
#
#   y_j = sum_i  phi_{ij}(x_i)
#
# where phi_{ij} is a learnable function parametrised as a B-spline plus a
# SiLU-activated linear "base" term.  Unlike ReLU-networks (which learn
# linear piece-wise functions through depth), a single KAN layer with order-3
# B-splines can represent smooth non-linear functions in one forward pass.
#
# B-spline formulation ("efficient-kan", Liu et al. 2024):
#   phi_{ij}(x) = w_b * SiLU(x)                      (base branch)
#              +  sum_k c_{ijk} * B_{k,p}(x)           (spline branch)
#
# where B_{k,p} are order-p B-spline basis functions defined over a uniform
# grid of grid_size intervals on [grid_range[0], grid_range[1]].
#
# Why B-splines? They form a partition of unity (sum to 1 over the grid),
# are compact-support (each affects only p+1 intervals), and their
# coefficients can be learned by gradient descent via the spline branch.
# Gradient flow is smooth everywhere, unlike piecewise-linear activations.
#
# Grid range [-1, 1]: inputs to the KAN head are squeezed into (-1, 1) by
# LayerNorm followed by tanh before the first KANLinear.  This keeps inputs
# inside the spline grid so basis functions are always active (no extrapolation).
#
# Reference:
#   Z. Liu et al., "KAN: Kolmogorov–Arnold Networks," arXiv:2404.19756, 2024.
#   efficient-kan formulation: https://github.com/Blealtan/efficient-kan
#   KAN-Former (Wang et al. 2026): uses KAN in the classification head.

# %%
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# %%
def b_splines(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Evaluate B-spline basis of order k via Cox-de Boor recursion.

    B_{i,0}(x) = 1  if t_i <= x < t_{i+1}, else 0
    B_{i,d}(x) = (x - t_i)/(t_{i+d} - t_i) * B_{i,d-1}(x)
               + (t_{i+d+1} - x)/(t_{i+d+1} - t_{i+1}) * B_{i+1,d-1}(x)

    Parameters
    ----------
    x    : (batch, in_features)       — clamped to grid range before call
    grid : (in_features, n_knots)     — n_knots = grid_size + 2*k + 1
    k    : spline order (3 for cubic)

    Returns
    -------
    bases : (batch, in_features, n_basis)  where n_basis = grid_size + k
    """
    x = x.unsqueeze(-1)  # (B, in, 1) for broadcasting against (in, n_knots)

    # Order-0: indicator for each interval [t_i, t_{i+1}).
    # For the last interval we include the right endpoint.
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    # Include right endpoint of the last interval
    at_end = (x[..., 0] == grid[:, -1])
    # (bases is a float tensor, so use max instead of bitwise-or)
    bases[..., -1] = torch.maximum(bases[..., -1], at_end.to(x.dtype))

    for d in range(1, k + 1):
        # grid slices for the de Boor recursion step d
        # Denominators are differences of knot values; zero when repeated knots.
        denom_l = grid[:, d:-1]     - grid[:, :-(d+1)]   # (in, N-d-1)
        denom_r = grid[:, (d+1):]   - grid[:, 1:-d]      # (in, N-d-1)

        # Safe division: 0/0 → 0 (convention for clamped B-splines)
        safe_l = torch.where(denom_l == 0,
                             torch.zeros_like(denom_l), denom_l)
        safe_r = torch.where(denom_r == 0,
                             torch.zeros_like(denom_r), denom_r)

        coeff_l = torch.where(denom_l == 0,
                              torch.zeros_like(x),
                              (x - grid[:, :-(d+1)]) / safe_l)
        coeff_r = torch.where(denom_r == 0,
                              torch.zeros_like(x),
                              (grid[:, (d+1):] - x)  / safe_r)

        bases = coeff_l * bases[..., :-1] + coeff_r * bases[..., 1:]

    return bases   # (B, in, grid_size + k)


# %%
class KANLinear(nn.Module):
    """
    One KAN layer: in_features -> out_features using learnable B-splines.

    Parameters
    ----------
    in_features   : number of input dimensions
    out_features  : number of output dimensions
    grid_size     : number of uniform intervals in [grid_range[0], grid_range[1]]
    spline_order  : B-spline order (3 = cubic)
    grid_range    : spline domain; inputs should be in this range (apply tanh before)
    scale_base    : initial scale of the SiLU base weight (like a residual connection)
    scale_spline  : initial scale of spline coefficients
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3,
                 grid_range: tuple = (-1.0, 1.0),
                 scale_base: float = 1.0, scale_spline: float = 1.0):
        super().__init__()
        self.in_features   = in_features
        self.out_features  = out_features
        self.grid_size     = grid_size
        self.spline_order  = spline_order
        self.grid_range    = grid_range

        n_basis = grid_size + spline_order   # number of B-spline basis functions

        # Base (SiLU shortcut): a standard linear layer applied to SiLU(x).
        # This ensures KANLinear is at least as expressive as a Linear layer,
        # and helps gradient flow at initialisation.
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features) * scale_base
        )

        # Spline coefficients: (in, out, n_basis) — one coefficient per
        # (input dim, output dim, basis function) triple.
        self.spline_weight = nn.Parameter(
            torch.empty(in_features, out_features, n_basis) * scale_spline
        )

        # Uniform grid on [grid_range[0], grid_range[1]] extended by spline_order
        # on each side (clamped B-splines).
        h = (grid_range[1] - grid_range[0]) / grid_size
        g = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        # Register as buffer: moves with .to(device) but not a parameter.
        # Expand to (in_features, n_knots) for vectorised basis evaluation.
        self.register_buffer("grid", g.unsqueeze(0).expand(in_features, -1).contiguous())

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming uniform for base (matches nn.Linear default)
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        # Small uniform for spline coefficients so the spline branch starts near 0
        nn.init.uniform_(self.spline_weight, -0.1 / self.grid_size,
                          0.1 / self.grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (*, in_features) — any batch shape
        Returns (*, out_features)
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)   # (N, in)

        # Clamp to grid range so basis functions are always active
        x_clamped = x_flat.clamp(self.grid_range[0], self.grid_range[1])

        # Base path: SiLU(x) @ base_weight^T
        base_out = F.linear(F.silu(x_flat), self.base_weight)   # (N, out)

        # Spline path: evaluate B-spline bases, then contract with coefficients
        splines = b_splines(x_clamped, self.grid, self.spline_order)
        # splines: (N, in, n_basis)
        # spline_weight: (in, out, n_basis)
        # output: sum over in and n_basis -> (N, out)
        spline_out = torch.einsum("bid,iod->bo", splines, self.spline_weight)

        out = base_out + spline_out
        return out.reshape(orig_shape[:-1] + (self.out_features,))

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"grid_size={self.grid_size}, order={self.spline_order}, "
                f"range={self.grid_range}")
