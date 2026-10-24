"""Per-beat metrics, comparison tables against Table 4.1, and figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Liu (2019) Table 4.1, "Standard ESN" rows, test set, in percent.
PAPER_TABLE = {
    "106": {"accuracy": 98.75, "sensitivity": 97.22, "precision": 97.22, "f1": 97.22},
    "119": {"accuracy": 99.70, "sensitivity": 100.0, "precision": 99.10, "f1": 99.55},
    "200": {"accuracy": 99.07, "sensitivity": 98.24, "precision": 99.40, "f1": 98.82},
    "201": {"accuracy": 99.24, "sensitivity": 100.0, "precision": 97.18, "f1": 98.57},
    "203": {"accuracy": 98.14, "sensitivity": 100.0, "precision": 90.32, "f1": 94.92},
    "223": {"accuracy": 99.07, "sensitivity": 99.05, "precision": 98.11, "f1": 98.58},
    "233": {"accuracy": 99.78, "sensitivity": 100.0, "precision": 99.21, "f1": 99.60},
}
PAPER_METRICS = ("accuracy", "sensitivity", "precision", "f1")
METRICS = ("accuracy", "sensitivity", "precision", "f1", "specificity")
VARIANT_KEYS = ["input", "leak", "lambda_mode", "weighting"]
PRIMARY_VARIANT = {"input": "spikes", "leak": 0.99, "lambda_mode": "cv", "weighting": "none"}
FLAG_PP = 2.0


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def binary_metrics(y_true, y_pred) -> dict:
    """Per-beat metrics with PVC (1) as the positive class; undefined ratios are NaN."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    tp = int((p & y).sum())
    fp = int((p & ~y).sum())
    fn = int((~p & y).sum())
    tn = int((~p & ~y).sum())
    n = tp + fp + fn + tn
    return {
        "accuracy": _ratio(tp + tn, n),
        "sensitivity": _ratio(tp, tp + fn),
        "precision": _ratio(tp, tp + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "specificity": _ratio(tn, tn + fp),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_test_pvc": tp + fn,
        "n_test_other": tn + fp,
        "all_normal_accuracy": _ratio(tn + fp, n),
    }


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean/std over seeds per (variant, record), in percent, next to the paper's numbers."""
    pct = per_seed.copy()
    for col in METRICS + ("all_normal_accuracy",):
        pct[col] = 100 * pct[col]
    keys = VARIANT_KEYS + ["record"]
    grouped = pct.groupby(keys, sort=False)
    out = grouped[["n_test_pvc", "n_test_other", "all_normal_accuracy"]].first()
    for col in METRICS:
        out[f"{col}_mean"] = grouped[col].mean()
        out[f"{col}_std"] = grouped[col].std(ddof=0)
    for col in ("tp", "fp", "fn", "tn"):
        out[f"{col}_mean"] = grouped[col].mean()
    out["n_seeds"] = grouped.size()
    out = out.reset_index()
    for col in PAPER_METRICS:
        out[f"paper_{col}"] = out["record"].map(lambda r: PAPER_TABLE[r][col])
        out[f"diff_{col}"] = out[f"{col}_mean"] - out[f"paper_{col}"]
    return out


def _cell(mean: float, std: float) -> str:
    return "n/a" if np.isnan(mean) else f"{mean:.2f} ± {std:.2f}"


def _diff(d: float) -> str:
    if np.isnan(d):
        return "n/a"
    text = f"{d:+.2f}"
    return f"**{text}**" if abs(d) > FLAG_PP else text


def _variant_table(df: pd.DataFrame) -> list[str]:
    lines = [
        "| Record | Test PVC / other | All-normal acc | Accuracy | Δ | Sensitivity | Δ | Precision | Δ | F1 | Δ | Specificity |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        cells = [r["record"], f"{int(r['n_test_pvc'])} / {int(r['n_test_other'])}",
                 f"{r['all_normal_accuracy']:.2f}"]
        for col in PAPER_METRICS:
            cells += [_cell(r[f"{col}_mean"], r[f"{col}_std"]), _diff(r[f"diff_{col}"])]
        cells.append(_cell(r["specificity_mean"], r["specificity_std"]))
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return lines


def _variant_name(v: dict) -> str:
    return f"input={v['input']}, leak a={v['leak']}, lambda={v['lambda_mode']}, weighting={v['weighting']}"


def write_tables(summary: pd.DataFrame, results_dir: Path, n_seeds: int) -> None:
    summary.to_csv(results_dir / "results_table.csv", index=False, float_format="%.4f")
    lines = [
        "# Results: Standard ESN, per-subject PVC detection (test window 10:00–15:00)",
        "",
        f"Our numbers are mean ± std over {n_seeds} reservoir seed(s), in percent. "
        f"Δ = ours − paper (Liu 2019, Table 4.1, Standard ESN), in percentage points; "
        f"**bold** means |Δ| > {FLAG_PP:g} pp. \"All-normal acc\" is the accuracy of predicting "
        "every beat as non-PVC.",
        "",
        "## Paper (Table 4.1, Standard ESN)",
        "",
        "| Record | Accuracy | Sensitivity | Precision | F1 |",
        "|---|---|---|---|---|",
    ]
    for rec, vals in PAPER_TABLE.items():
        lines.append(f"| {rec} | " + " | ".join(f"{vals[c]:.2f}" for c in PAPER_METRICS) + " |")

    variants = summary[VARIANT_KEYS].drop_duplicates().to_dict("records")
    lines += ["", "## Variant overview (mean over records)", "",
              "| Input | Leak a | Lambda | Weighting | Mean accuracy | Mean F1 | Mean ΔF1 | Records with \\|ΔF1\\| > 2 pp |",
              "|---|---|---|---|---|---|---|---|"]
    for v in variants:
        sel = summary.loc[(summary[VARIANT_KEYS] == pd.Series(v)).all(axis=1)]
        n_flag = int((sel["diff_f1"].abs() > FLAG_PP).sum())
        lines.append(f"| {v['input']} | {v['leak']} | {v['lambda_mode']} | {v['weighting']} | "
                     f"{sel['accuracy_mean'].mean():.2f} | {sel['f1_mean'].mean():.2f} | "
                     f"{sel['diff_f1'].mean():+.2f} | {n_flag} / {len(sel)} |")

    primary_first = sorted(variants, key=lambda v: v != PRIMARY_VARIANT)
    for v in primary_first:
        sel = summary.loc[(summary[VARIANT_KEYS] == pd.Series(v)).all(axis=1)]
        tag = " (primary)" if v == PRIMARY_VARIANT else ""
        lines += ["", f"## {_variant_name(v)}{tag}", ""] + _variant_table(sel)
    (results_dir / "results_table.md").write_text("\n".join(lines) + "\n")


def plot_record(path: Path, fs: float, start: int, ecg: np.ndarray, trace: np.ndarray, threshold: float,
                true_pvc: np.ndarray, pred_pvc: np.ndarray, window: tuple[int, int], title: str) -> None:
    """ECG, readout output and threshold over a test segment, with true/predicted PVC markers."""
    t = (start + np.arange(len(ecg))) / fs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1.2, 1]})
    ax1.plot(t, ecg, color="0.25", lw=0.8)
    ax1.plot(true_pvc / fs, np.full(len(true_pvc), 1.04), "v", color="tab:red", ms=7,
             label="true PVC")
    ax1.plot(pred_pvc / fs, np.full(len(pred_pvc), 1.12), "v", color="tab:blue", ms=7, mfc="none",
             label="predicted PVC")
    ax1.set_ylim(-0.05, 1.2)
    ax1.set_ylabel("ECG (normalised)")
    ax1.legend(loc="lower right", fontsize=8, ncol=2)
    for r in true_pvc:
        ax2.axvspan((r + window[0]) / fs, (r + window[1]) / fs, color="tab:red", alpha=0.12, lw=0)
    ax2.plot(t, trace, color="tab:green", lw=1, label="readout output")
    ax2.axhline(threshold, color="k", ls="--", lw=1, label=f"threshold θ={threshold:.2f}")
    ax2.set_ylabel("readout")
    ax2.set_xlabel("time in record (s)")
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
