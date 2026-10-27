"""Figures: baseline removal, fiducial points, F-scores, confusion matrices, ROC curves."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .data import CLASSES  # noqa: E402
from .fiducials import POINTS  # noqa: E402

# Reference categorical palette (light mode), fixed order: class / model identity.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#9cc3ef", "#2a78d6", "#0d3a73"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def plot_baseline(raw, base, clean, fs, path, name, seconds=(0, 10)):
    a, b = (int(s * fs) for s in seconds)
    t = np.arange(a, b) / fs
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    axes[0].plot(t, raw[a:b], color=SERIES[0], lw=1, label="raw MLII")
    axes[0].plot(t, base[a:b], color=SERIES[1], lw=2, label="estimated baseline (elliptic LP 1 Hz)")
    axes[0].set_title(f"Record {name}: before baseline removal")
    axes[0].legend(loc="upper right")
    axes[1].plot(t, clean[a:b], color=SERIES[0], lw=1)
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set_title("after: raw - baseline")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.set_ylabel("mV")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fiducials(beats_by_class: dict, points_by_class: dict, fs, r_index, path, title):
    """One panel per class: a few example beats with their 11 fiducial points labelled."""
    names = [c for c in CLASSES if c in beats_by_class]
    fig, axes = plt.subplots(len(names), 3, figsize=(11, 2.3 * len(names)), squeeze=False)
    t = (np.arange(beats_by_class[names[0]].shape[1]) - r_index) * 1000 / fs
    for i, c in enumerate(names):
        B, P = beats_by_class[c], points_by_class[c]
        for j in range(3):
            ax = axes[i, j]
            if j >= len(B):
                ax.axis("off")
                continue
            ax.plot(t, B[j], color=SERIES[CLASSES.index(c)], lw=1.5)
            for k, name in enumerate(POINTS):
                col = P[j, k]
                ax.plot(t[col], B[j, col], "o", ms=5, color=INK, mec=SURFACE, mew=1.5)
                ax.annotate(name, (t[col], B[j, col]), textcoords="offset points",
                            xytext=(0, 6 if k % 2 else -11), ha="center", fontsize=7, color=INK2)
            ax.set_title(f"class {c}, example {j + 1}", loc="left")
            if j == 0:
                ax.set_ylabel("mV")
            if i == len(names) - 1:
                ax.set_xlabel("ms from R")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fscores(names, scores, selected, path, threshold=1000.0, title="ANOVA F-scores (f_classif)"):
    order = np.argsort(scores)[::-1]
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = [SERIES[0] if selected[i] else MUTED for i in order]
    ax.bar(np.arange(len(order)), np.asarray(scores)[order], color=colors, width=0.8,
           edgecolor=SURFACE, linewidth=1)
    ax.axhline(threshold, color=INK2, lw=1, ls="--")
    ax.text(len(order) - 0.5, threshold, f"F = {threshold:g}", ha="right", va="bottom", color=INK2)
    ax.set_xticks(np.arange(len(order)), [names[i] for i in order], rotation=90, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("F-score (log)")
    ax.set_title(f"{title}: blue = the 20 selected features", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusions(cms: dict, path, title):
    """Row-normalised colour, raw counts as text; one panel per model."""
    fig, axes = plt.subplots(1, len(cms), figsize=(3.3 * len(cms), 3.4), squeeze=False)
    for ax, (model, cm) in zip(axes[0], cms.items()):
        cm = np.asarray(cm)
        rows = cm.sum(axis=1, keepdims=True)
        frac = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
        ax.imshow(frac, cmap=SEQ, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if frac[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(model)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rocs(curves: dict, path, title):
    """curves: model -> {class name or 'micro': (fpr, tpr, auc)}; one panel per model."""
    fig, axes = plt.subplots(1, len(curves), figsize=(3.4 * len(curves), 3.5), squeeze=False)
    for ax, (model, cv) in zip(axes[0], curves.items()):
        for k, (label, (fpr, tpr, auc)) in enumerate(cv.items()):
            color = INK if label == "micro" else SERIES[CLASSES.index(label)]
            ax.plot(fpr, tpr, color=color, lw=2 if label == "micro" else 1.5,
                    label=f"{label} ({100 * auc:.2f})")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(f"{model}: ROC (AUC %)")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
