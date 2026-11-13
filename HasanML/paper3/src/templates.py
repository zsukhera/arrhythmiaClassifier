# %% [markdown]
# ## templates.py — Matched-filter kernel initialisation templates
#
# Reference: Farag (2023), Sections 4.2 and 5.2.
#
# A matched filter maximises signal-to-noise ratio when the template exactly
# matches the signal shape.  Here, the class-average waveform is used as the
# template (the "matched" shape), and the Conv1D layer computes the
# cross-correlation between the input beat and each template.
#
# Key implementation notes:
# - Keras Conv1D performs cross-correlation (NOT convolution), so templates
#   must NOT be time-reversed before loading into the kernel weights.
# - Kernel shape in Keras: (kernel_size, in_channels, n_filters) = (NK, C, 13).
# - Template i is placed at kernel[:, :, i].
# - Each template is normalised to unit L2 norm so that no single class
#   dominates GlobalMaxPooling1D due to amplitude differences [assumption].
#
# ASSUMPTION: 13 sub-classes — N, L, R, e, j, A, a, J, S, V, E, F, Q.
# The paper uses these sub-classes (Figure 4); AAMI super-classes (5 templates)
# and per-record templates are implemented as ablations.

# %%
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless rendering for servers / Colab
import matplotlib.pyplot as plt
import numpy as np

from .data import SUB_CLASSES, AAMI_CLASSES, AAMI_MAP

log = logging.getLogger(__name__)


# %%
def compute_subclass_templates(
    segments_by_symbol: dict,
    nk: int,
    sub_classes: list = SUB_CLASSES,
    min_beats: int = 5,
    normalize_l2: bool = True,
    fallback_segments: Optional[dict] = None,
) -> dict:
    """
    Compute one template (class-average waveform) per MIT-BIH sub-class.

    Parameters
    ----------
    segments_by_symbol : dict mapping MIT-BIH symbol -> list of (nk,) or (nk,C)
                         preprocessed beat segments from DS1-train
    nk                 : kernel / segment length
    sub_classes        : ordered list of sub-class symbols (default: 13)
    min_beats          : if fewer than this many beats exist in DS1-train for a
                         sub-class, fall back to `fallback_segments` (ds1_all)
                         [assumption]
    normalize_l2       : scale each template to unit L2 norm [assumption]
    fallback_segments  : segments from all DS1 (used if DS1-train count is low)

    Returns
    -------
    templates : dict mapping symbol -> np.ndarray of shape (nk,) or (nk, C)
    """
    templates = {}
    for sym in sub_classes:
        segs = segments_by_symbol.get(sym, [])
        n = len(segs)
        if n < min_beats:
            fb = (fallback_segments or {}).get(sym, [])
            if len(fb) >= min_beats:
                log.info("template '%s': only %d beats in DS1-train; "
                         "using all-DS1 fallback (%d beats)", sym, n, len(fb))
                segs = fb
                n = len(fb)
            else:
                log.warning("template '%s': only %d beats even in DS1 fallback; "
                            "template will be noisy", sym, n)

        log.info("template '%s': %d beats -> template computed", sym, n)

        if n == 0:
            # Zero template for absent sub-classes
            shape = (nk,) if isinstance(segs, list) else np.array(segs).shape[1:]
            t = np.zeros(nk if not shape else (nk, *shape[1:]), dtype=np.float64)
        else:
            arr = np.array(segs, dtype=np.float64)  # (n, nk) or (n, nk, C)
            t   = arr.mean(axis=0)                   # (nk,) or (nk, C)

        if normalize_l2:
            norm = np.linalg.norm(t)
            if norm > 1e-9:
                t = t / norm

        templates[sym] = t

    return templates


def templates_to_kernel(templates: dict, sub_classes: list, n_channels: int = 1) -> np.ndarray:
    """
    Pack the template dict into a Keras Conv1D kernel array.

    Keras Conv1D kernel shape: (kernel_size, in_channels, n_filters).
    Template i (for sub_classes[i]) is placed at kernel[:, :, i].
    No time-reversal: Keras uses cross-correlation, not convolution [verified].

    Parameters
    ----------
    templates   : dict symbol -> (nk,) or (nk, C) array (from compute_subclass_templates)
    sub_classes : ordered list of symbols, defines filter index order
    n_channels  : number of input channels (C)

    Returns
    -------
    kernel : (nk, C, n_filters) float32 array
    """
    n_filters  = len(sub_classes)
    sample_tpl = next(iter(templates.values()))
    nk = sample_tpl.shape[0]

    kernel = np.zeros((nk, n_channels, n_filters), dtype=np.float32)
    for i, sym in enumerate(sub_classes):
        t = np.asarray(templates[sym], dtype=np.float32)
        if t.ndim == 1:
            # single-channel: broadcast across channels
            for c in range(n_channels):
                kernel[:, c, i] = t
        else:
            # multi-channel: t shape (nk, C)
            kernel[:, :, i] = t[:, :n_channels]

    return kernel


def apply_mf_init(conv_layer, kernel: np.ndarray, frozen: bool = False) -> None:
    """
    Set Conv1D weights from matched-filter templates and optionally freeze them.

    frozen=True replicates Model 3/4 in the paper (mf_frozen): the kernels are
    computed analytically and do not change during training.  The benefit is
    that the network is interpretable: each filter responds to a known sub-class.
    The cost: the kernels cannot adapt to the specific training distribution.

    frozen=False (mf_trainable, paper's Model 6): MF templates initialise the
    kernels but gradient descent can refine them.  This outperforms frozen in
    most configurations (paper Table 4).
    """
    # Get current bias shape
    current_w = conv_layer.get_weights()
    if len(current_w) == 2:
        bias = current_w[1]
    else:
        bias = np.zeros(kernel.shape[-1], dtype=np.float32)
    conv_layer.set_weights([kernel, bias])
    conv_layer.trainable = not frozen


# %%
def plot_templates(templates: dict, sub_classes: list, nk: int,
                   fs_target: float = 128.0, out_path: Optional[Path] = None) -> None:
    """
    Plot the 13 matched-filter templates (replicates paper Figure 4).
    Each subplot shows the class-average waveform that initialises one Conv1D filter.
    """
    n = len(sub_classes)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
    axes = axes.flat

    t = np.arange(nk) / fs_target * 1000  # ms
    for i, sym in enumerate(sub_classes):
        ax = axes[i]
        tpl = np.asarray(templates.get(sym, np.zeros(nk)))
        if tpl.ndim > 1:
            tpl = tpl[:, 0]  # first channel for plotting
        ax.plot(t, tpl, lw=1.2)
        aami = AAMI_MAP.get(sym, "?")
        ax.set_title(f"'{sym}' (AAMI {aami})", fontsize=9)
        ax.set_xlabel("ms", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="k", lw=0.5, ls="--")

    for j in range(i + 1, len(list(axes))):
        axes[j].set_visible(False)

    fig.suptitle(f"Matched-filter templates (NK={nk}, Farag 2023 Figure 4 equivalent)",
                 fontsize=11)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        log.info("template plot saved to %s", out_path)
    plt.close(fig)


# %%
def compute_aami_templates(segments_by_symbol: dict, nk: int,
                            normalize_l2: bool = True) -> dict:
    """
    Ablation: 5 templates (one per AAMI super-class) instead of 13 sub-classes.
    The paper found this performs worse than 13 sub-class templates (Sec 5.2).
    """
    aami_segs = {cls: [] for cls in AAMI_CLASSES}
    for sym, segs in segments_by_symbol.items():
        aami = AAMI_MAP.get(sym)
        if aami and segs:
            aami_segs[aami].extend(segs)

    templates = {}
    for cls, segs in aami_segs.items():
        if segs:
            arr = np.array(segs, dtype=np.float64)
            t   = arr.mean(axis=0)
        else:
            t = np.zeros(nk, dtype=np.float64)
        if normalize_l2:
            norm = np.linalg.norm(t)
            if norm > 1e-9:
                t /= norm
        templates[cls] = t
    return templates
