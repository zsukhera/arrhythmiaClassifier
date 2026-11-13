# %% [markdown]
# ## model.py — KAN-Former-1D architecture
#
# Architecture (DEFAULT, ~1M parameters):
#
#   beat_in (B,1,256)          context_in (B,1,1080)        rr_in (B,6)
#        |                           |
#   [Stem: Conv1d(1,32,k=7)]   [Stem: Conv1d(1,32,k=7)]
#   BN → GELU                  BN → GELU
#        |                           |
#   [MS Block×3: kernels 3,7,15 depthwise-sep, SE, residual, MaxPool2]
#   32→64→128 channels          32→64→128 channels
#        |                           |
#   (B,128,32)                  (B,128,135)
#   project+permute             project+permute
#   (B,32,128) tokens           (B,135,128) tokens
#        \                          /
#   [CLS] + beat_tokens + context_tokens + [RR_token]  → (B,169,128)
#   + learnable positional embeddings + segment type embeddings
#        |
#   [Transformer Encoder × 3 layers]
#   pre-norm, 4 heads, GLU-FFN(256), dropout 0.1, stochastic depth 0.1
#        |
#   CLS out (B,128) + mean pool (B,128) + rr_in (B,6) → (B,262)
#   LayerNorm → tanh (squash to (-1,1) for spline grid)
#   KANLinear(262→64) → dropout 0.2 → KANLinear(64→n_classes)
#
# References:
#   Wang et al. (2026), KAN-Former: MS-CA blocks, TF-FFN, KAN head.
#   Farag (2023): RR features fed directly to head (shortcut for S beats).
#   Xia et al. (2023): multi-scale 1D-CNN + Transformer for inter-patient AAMI.

# %%
import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .kan import KANLinear
except ImportError:
    from kan import KANLinear

log = logging.getLogger(__name__)


# %%
class StochasticDepth(nn.Module):
    """
    Randomly drops an entire layer's output during training (Huang et al. 2016).
    At inference the output is the layer's normal output scaled by survival_prob.
    With pre-norm residual connections this is equivalent to randomly skipping layers.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Sample per-batch mask (shape: B×1×...×1) for consistent scaling
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.empty(shape, device=x.device).bernoulli_(keep_prob)
        return x * (mask / keep_prob)


class SE1d(nn.Module):
    """
    Squeeze-and-Excitation for 1D feature maps (channel attention).
    Globally pools (B,C,L) → (B,C) → 2-layer MLP → sigmoid gate → scale (B,C,L).
    Reduction factor 8 follows the 2D SE-Net convention adapted for 1D.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1  = nn.Linear(channels, mid)
        self.fc2  = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x).squeeze(-1)        # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)          # (B, C, L)


class DSConv1d(nn.Module):
    """Depthwise-separable Conv1d: depthwise(k) + pointwise(1)."""
    def __init__(self, in_ch: int, out_ch: int, k: int):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch,  k, padding=k // 2, groups=in_ch)
        self.pw = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class MultiScaleBlock(nn.Module):
    """
    Multi-Scale Channel-Attention block (1D adaptation of MS-CA, Wang et al. 2026).

    Three parallel depthwise-separable branches with kernels {3, 7, 15}:
      k=3  → fine local features (individual peaks)
      k=7  → medium-scale (QRS complex, ~19 ms at 360 Hz)
      k=15 → coarse scale (full QRS, ~42 ms at 360 Hz — the main QRS scale)
    Outputs are concatenated → pointwise projection → BN → GELU → SE → residual.

    If in_ch != out_ch a 1×1 projection is used for the skip connection.
    MaxPool1d(2) halves the temporal dimension.
    """
    def __init__(self, in_ch: int, out_ch: int, kernels: List[int] = (3, 7, 15),
                 se_reduction: int = 8):
        super().__init__()
        self.branches = nn.ModuleList([DSConv1d(in_ch, in_ch, k) for k in kernels])
        self.proj   = nn.Conv1d(in_ch * len(kernels), out_ch, 1)
        self.bn     = nn.BatchNorm1d(out_ch)
        self.se     = SE1d(out_ch, se_reduction)
        self.skip   = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
        self.pool   = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale branches
        feats = torch.cat([b(x) for b in self.branches], dim=1)  # (B, 3*in, L)
        feats = F.gelu(self.bn(self.proj(feats)))                 # (B, out, L)
        feats = self.se(feats)                                    # channel attention
        feats = feats + self.skip(x)                              # residual
        return self.pool(feats)                                   # (B, out, L//2)


class CNNStem(nn.Module):
    """Stem + 3 MS blocks for one input branch (beat or context)."""
    def __init__(self, in_ch: int, channels: List[int], kernels: List[int],
                 se_reduction: int, d_model: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )
        blocks = []
        ch_in = channels[0]
        for ch_out in channels:
            blocks.append(MultiScaleBlock(ch_in, ch_out, kernels, se_reduction))
            ch_in = ch_out
        self.blocks = nn.Sequential(*blocks)
        # Project final CNN channels to d_model (may be identity if equal)
        self.proj = nn.Conv1d(channels[-1], d_model, 1) if channels[-1] != d_model else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) → (B, T, d_model) token sequence."""
        h = self.stem(x)
        h = self.blocks(h)   # (B, d_model_cnn, T)
        h = self.proj(h)     # (B, d_model, T)
        return h.permute(0, 2, 1)   # (B, T, d_model) — tokens


# %%
class GLUFFN(nn.Module):
    """
    Gated Linear Unit FFN (TF-FFN in Wang et al. 2026).
    h → W1(h) split into value and gate halves → gate * GELU(value) → W2.
    Uses less memory than standard FFN for the same hidden dimension because
    the effective width is d_ff (not 2*d_ff), but it has gating non-linearity.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff * 2)   # outputs value||gate
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        v, g = self.w1(x).chunk(2, dim=-1)
        return self.w2(self.dropout(v * F.gelu(g)))


class TransformerLayer(nn.Module):
    """
    One pre-norm Transformer encoder layer with GLU FFN and stochastic depth.
    Pre-norm (LN before attention/FFN) is more stable than post-norm at smaller
    model sizes and is used by most modern Transformers.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, drop_path: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = GLUFFN(d_model, d_ff, dropout)
        self.drop  = StochasticDepth(drop_path)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with pre-norm
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(self.attn_drop(h))
        # FFN with pre-norm
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# %%
class KANFormer(nn.Module):
    """
    KAN-Former-1D: two-branch CNN encoder + Transformer + KAN classification head.

    Inputs:
      beat     : (B, C, 256)   — morphology window
      context  : (B, C, 1080) or None — rhythm context window
      rr       : (B, 6)        — 6 RR ratio features

    Output: logits (B, n_classes) — no softmax (use with focal/CE loss)
    """

    def __init__(self, cfg: dict, n_classes: int = 5):
        super().__init__()
        m = cfg["model"]
        p = cfg["preprocess"]

        d_model     = int(m["d_model"])
        n_heads     = int(m["n_heads"])
        n_layers    = int(m["n_layers"])
        d_ff        = int(m["ffn_hidden"])
        dropout     = float(m["dropout"])
        drop_path   = float(m["stoch_depth"])
        channels    = list(m["cnn_channels"])
        ms_kernels  = list(m["ms_kernels"])
        se_red      = int(m["se_reduction"])
        head_type   = str(m.get("head", "kan"))
        grid_size   = int(m["kan_grid_size"])
        spline_order= int(m["kan_spline_order"])
        grid_range  = tuple(m["kan_grid_range"])
        kan_hidden  = int(m["kan_hidden"])
        head_drop   = float(m["head_dropout"])
        n_rr        = int(cfg["rr"]["n_features"])
        ctx_len     = int(p.get("context_len", 1080))

        self.d_model  = d_model
        self.n_rr     = n_rr
        self.has_ctx  = (ctx_len > 0)

        # Beat branch: 256 / 2^3 = 32 tokens
        in_ch = 1   # MLII only (extend to 2 for ablation with V1/V5)
        self.beat_stem = CNNStem(in_ch, channels, ms_kernels, se_red, d_model)

        # Context branch: 1080 / 2^3 = 135 tokens
        if self.has_ctx:
            self.ctx_stem = CNNStem(in_ch, channels, ms_kernels, se_red, d_model)

        # Beat tokens: 32; context tokens: 135 (or 0); CLS: 1; RR token: 1
        beat_tokens = 256 // (2 ** len(channels))
        ctx_tokens  = (ctx_len // (2 ** len(channels))) if self.has_ctx else 0
        n_tokens    = 1 + beat_tokens + ctx_tokens + 1   # CLS + beat + ctx + RR
        self.n_tokens = n_tokens

        # Learnable CLS token (broadcast over batch dimension in forward)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional + segment embeddings
        # Segment types: 0=CLS, 1=beat, 2=context, 3=rr
        self.pos_embed = nn.Embedding(n_tokens, d_model)
        self.seg_embed = nn.Embedding(4, d_model)
        self.register_buffer("pos_ids", torch.arange(n_tokens))
        seg = torch.tensor([0] + [1]*beat_tokens + [2]*ctx_tokens + [3])
        self.register_buffer("seg_ids", seg)

        # RR token projection
        self.rr_proj = nn.Linear(n_rr, d_model)

        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout, drop_path)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        head_in = d_model * 2 + n_rr   # CLS + mean-pool + RR direct
        self.head_norm = nn.LayerNorm(head_in)

        if head_type == "kan":
            self.head = nn.Sequential(
                KANLinear(head_in, kan_hidden, grid_size, spline_order, grid_range),
                nn.Dropout(head_drop),
                KANLinear(kan_hidden, n_classes, grid_size, spline_order, grid_range),
            )
        else:
            # MLP ablation (A1): same hidden dimension
            self.head = nn.Sequential(
                nn.Linear(head_in, kan_hidden),
                nn.GELU(),
                nn.Dropout(head_drop),
                nn.Linear(kan_hidden, n_classes),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.seg_embed.weight, std=0.02)

    def forward(self, beat: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                rr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        beat    : (B, 1, 256)
        context : (B, 1, 1080) or None
        rr      : (B, 6) or None (zeros used if not provided)

        Returns logits (B, n_classes).
        """
        B = beat.shape[0]
        device = beat.device

        # Beat tokens: (B, 32, d_model)
        beat_tok = self.beat_stem(beat)

        # Context tokens: (B, 135, d_model)
        if self.has_ctx and context is not None:
            ctx_tok = self.ctx_stem(context)
        else:
            # Ablation: pad with zeros if context disabled
            ctx_len_tok = self.n_tokens - 1 - beat_tok.shape[1] - 1
            ctx_tok = torch.zeros(B, max(0, ctx_len_tok),
                                  self.d_model, device=device)

        # RR token: (B, 1, d_model)
        if rr is not None:
            rr_tok = self.rr_proj(rr).unsqueeze(1)
        else:
            rr_tok = torch.zeros(B, 1, self.d_model, device=device)
            rr = torch.zeros(B, self.n_rr, device=device)

        # CLS token: (B, 1, d_model)
        cls = self.cls_token.expand(B, -1, -1)

        # Assemble token sequence: (B, n_tokens, d_model)
        tokens = torch.cat([cls, beat_tok, ctx_tok, rr_tok], dim=1)

        # Add positional and segment embeddings
        pos = self.pos_embed(self.pos_ids[:tokens.shape[1]])
        seg = self.seg_embed(self.seg_ids[:tokens.shape[1]])
        tokens = tokens + pos + seg

        # Transformer
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        # Head inputs: CLS output + mean pool over all tokens + direct RR features
        cls_out  = tokens[:, 0, :]                   # (B, d_model)
        mean_out = tokens.mean(dim=1)                 # (B, d_model)
        head_in  = torch.cat([cls_out, mean_out, rr], dim=1)  # (B, 2*d_model + n_rr)

        # Squash to (-1, 1) for the B-spline grid
        head_in = torch.tanh(self.head_norm(head_in))

        return self.head(head_in)   # (B, n_classes) logits


# %%
def build_model(cfg: dict, n_classes: int) -> KANFormer:
    return KANFormer(cfg, n_classes=n_classes)


def count_parameters(model: nn.Module) -> dict:
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_summary(model: nn.Module) -> str:
    p = count_parameters(model)
    return f"KANFormer: {p['total']:,} total params, {p['trainable']:,} trainable"


def compute_flops(model: nn.Module, cfg: dict, device: str = "cpu") -> Optional[int]:
    """Estimate FLOPs via thop or fvcore (returns None if neither installed)."""
    beat_len = int(cfg["preprocess"]["beat_len"])
    ctx_len  = int(cfg["preprocess"].get("context_len", 1080))
    n_rr     = int(cfg["rr"]["n_features"])

    beat = torch.zeros(1, 1, beat_len)
    ctx  = torch.zeros(1, 1, ctx_len) if ctx_len > 0 else None
    rr   = torch.zeros(1, n_rr)

    try:
        from thop import profile
        macs, _ = profile(model, inputs=(beat, ctx, rr), verbose=False)
        return int(macs * 2)
    except ImportError:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, (beat, ctx, rr))
        return int(flops.total())
    except ImportError:
        return None
