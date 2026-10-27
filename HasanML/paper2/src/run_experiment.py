"""Run the Yakut & Dogru Bolat (2020) replication end to end.

    python -m src.run_experiment --config configs/default.yaml
    python -m src.run_experiment --protocols A --models LR LDA --no-ablations   # quick check

Parallelism: per-record feature extraction runs in a process pool. Every model fit
(protocol x fold x feature set x ablation x model x seed) is an independent job in a
second process pool, slowest models first, each process limited to `blas_threads`
BLAS/OpenMP threads. With a CUDA GPU (cuML installed), LR and SVM jobs run on the
device from a small thread pool in parallel with the CPU pool. Finished jobs are
cached, so an interrupted run resumes where it stopped.
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import platform
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml

from .backend import GPU_MODELS, resolve_device
from .data import CLASSES, PAPER_COUNTS, make_splits, used_records
from .features import FEATURE35, FEATURE_NAMES, build_dataset, feature_cache_key, record_features
from .metrics import auc_scores, confusion, metrics_from_cm, paper_metrics_from_error, roc_curves
from .models import fit_predict, make_model
from .plots import plot_baseline, plot_confusions, plot_fiducials, plot_fscores, plot_rocs
from .select import fit_scaler, fit_selector, fscore_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("p2")

PROTOCOL_NAMES = {"A": "inter-patient DS1/DS2", "B": "intra-patient stratified split",
                  "C": "5-fold stratified CV"}
# Paper Tables 3/4 (%): model -> (accuracy, sensitivity, specificity, F1), and ROC AUC.
PAPER_RESULTS = {
    57: {"LR": (99.801, 99.503, 99.876, 99.503), "LDA": (99.339, 98.346, 99.587, 98.963),
         "SVM": (98.806, 97.015, 99.254, 98.122), "MLP": (98.623, 96.557, 99.139, 97.831)},
    20: {"LR": (99.766, 99.416, 99.854, 99.416), "LDA": (99.697, 99.243, 99.811, 99.243),
         "SVM": (99.552, 98.880, 99.720, 99.298), "MLP": (98.776, 96.939, 99.235, 98.074)},
}
PAPER_AUC = {57: {"LR": 99.61, "LDA": 99.31, "SVM": 99.33, "MLP": 99.07},
             20: {"LR": 99.58, "LDA": 99.32, "SVM": 99.31, "MLP": 99.08}}
COST = {"SVM": 4, "MLP": 3, "LR": 1, "LDA": 0}   # scheduling order: slowest first


@dataclass(frozen=True)
class Job:
    protocol: str
    fold: int
    nfeat: int
    variant: str      # default | scaler_all | select_all | no_feature35
    model: str
    seed: int


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _path(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ----------------------------------------------------------------------------- workers

_DATA: dict = {}


def _init_worker(data_path: str, blas_threads: int) -> None:
    """Load the feature matrix and split indices once per process; cap BLAS threads."""
    global _limit
    from threadpoolctl import threadpool_limits
    _limit = threadpool_limits(blas_threads)
    z = np.load(data_path)
    _DATA.update({k: z[k] for k in z.files})


def _design(job: Job, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Scaled (and selected) design matrices for the job; selected mask when nfeat < 57."""
    X, y = _DATA["X"], _DATA["y"]
    tr, te = _DATA[f"{job.protocol}{job.fold}_tr"], _DATA[f"{job.protocol}{job.fold}_te"]
    if job.variant == "no_feature35":
        X = np.delete(X, FEATURE35, axis=1)
    scaler = fit_scaler(X if job.variant == "scaler_all" else X[tr])
    Xs = np.clip(scaler.transform(X), -1e6, 1e6)
    mask = None
    if job.nfeat < Xs.shape[1]:
        sel = fit_selector(Xs if job.variant == "select_all" else Xs[tr],
                           y if job.variant == "select_all" else y[tr], k)
        mask = sel.get_support()
        Xs = Xs[:, mask]
    return Xs[tr], Xs[te], y[tr], mask


def run_job(job: Job, lr_alt_iter: int, device: str) -> list[dict]:
    """Fit one model; an unconverged LR is refit with max_iter=lr_alt_iter as 'LR (max_iter=N)'."""
    Xtr, Xte, ytr, mask = _design(job, job.nfeat)
    out = []
    res = fit_predict(make_model(job.model, job.seed, device=device), Xtr, ytr, Xte)
    out.append({"job": job, "mask": mask, "device": device, **res})
    if job.model == "LR" and not res["converged"] and lr_alt_iter:
        res2 = fit_predict(make_model("LR", job.seed, max_iter=lr_alt_iter, device=device), Xtr, ytr, Xte)
        out.append({"job": replace(job, model=f"LR (max_iter={lr_alt_iter})"), "mask": mask,
                    "device": device, **res2})
    return out


def _job_cache_path(cache_dir: Path, job: Job, data_hash: str, lr_alt: int, device: str) -> Path:
    key = json.dumps([asdict(job), data_hash, lr_alt, device, sklearn.__version__], sort_keys=True)
    return cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.pkl"


def _run_cached(job: Job, lr_alt: int, device: str, path: str) -> list[dict]:
    res = run_job(job, lr_alt, device)
    tmp = Path(path).with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return res


def build_jobs(cfg: dict, n_folds: dict) -> list[Job]:
    m, ab = cfg["models"], cfg["ablations"]
    jobs = []
    for p in cfg["protocols"]:
        for fold in range(n_folds[p]):
            for nfeat in cfg["nfeat"]:
                for model in m["names"]:
                    seeds = m["mlp_seeds"] if model == "MLP" else [cfg["seed"]]
                    jobs += [Job(p, fold, nfeat, "default", model, s) for s in seeds]
                    if not cfg["run_ablations"] or model not in ab["models"]:
                        continue
                    seed = seeds[0]
                    for variant in ("scaler_all", "select_all", "no_feature35"):
                        if not ab.get(variant, False) or (variant == "select_all" and nfeat >= 57):
                            continue
                        jobs.append(Job(p, fold, nfeat, variant, model, seed))
    return jobs


def run_all(jobs: list[Job], cfg: dict, device: str, data_path: Path, data_hash: str,
            cache_dir: Path) -> list[dict]:
    comp, lr_alt = cfg["compute"], int(cfg["models"]["lr_max_iter_alt"])
    results, todo = [], []
    for job in jobs:
        path = _job_cache_path(cache_dir, job, data_hash, lr_alt, device if job.model in GPU_MODELS else "cpu")
        if path.exists():
            with open(path, "rb") as fh:
                results += pickle.load(fh)
        else:
            todo.append((job, path))
    log.info("jobs: %d total, %d cached, %d to run", len(jobs), len(jobs) - len(todo), len(todo))
    if not todo:
        return results
    todo.sort(key=lambda jp: (-COST.get(jp[0].model, 1), -(jp[0].protocol == "C"), jp[0].nfeat))
    gpu = [jp for jp in todo if device == "cuda" and jp[0].model in GPU_MODELS]
    cpu = [jp for jp in todo if jp not in gpu]
    workers = comp["cpu_workers"]
    if workers == "auto":
        workers = max(1, (os.cpu_count() or 2) // max(1, comp["blas_threads"]))
    t0, done = time.perf_counter(), 0
    futures = {}
    with ProcessPoolExecutor(max_workers=int(workers), initializer=_init_worker,
                             initargs=(str(data_path), int(comp["blas_threads"]))) as pex, \
            ThreadPoolExecutor(max_workers=int(comp["gpu_workers"])) as tex:
        if gpu:
            _init_worker(str(data_path), int(comp["blas_threads"]))
            log.info("GPU: %d LR/SVM jobs on %d threads (cuML)", len(gpu), comp["gpu_workers"])
        for job, path in cpu:
            futures[pex.submit(_run_cached, job, lr_alt, "cpu", str(path))] = job
        for job, path in gpu:
            futures[tex.submit(_run_cached, job, lr_alt, "cuda", str(path))] = job
        log.info("CPU: %d jobs on %d processes x %d BLAS threads", len(cpu), workers, comp["blas_threads"])
        for fut in as_completed(futures):
            job = futures[fut]
            res = fut.result()
            results += res
            done += 1
            r = res[0]
            log.info("[%d/%d %.0fs] %s fold %d %dF %-12s %-4s seed %d: fit %.1fs%s", done, len(todo),
                     time.perf_counter() - t0, job.protocol, job.fold, job.nfeat, job.variant, job.model,
                     job.seed, r["fit_seconds"], "" if r["converged"] else " (not converged)")
    return results


# ----------------------------------------------------------------------------- reporting

def aggregate(results: list[dict], y: np.ndarray, splits: dict) -> tuple[pd.DataFrame, dict]:
    """One row per (protocol, nfeat, variant, model, seed); CV folds are pooled."""
    groups = defaultdict(list)
    for r in results:
        j = r["job"]
        groups[(j.protocol, j.nfeat, j.variant, j.model, j.seed)].append(r)
    rows, pooled = [], {}
    for key, rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["job"].fold)
        y_true = np.concatenate([y[splits[(key[0], r["job"].fold)][1]] for r in rs])
        pred = np.concatenate([r["pred"] for r in rs])
        proba = np.concatenate([r["proba"] for r in rs])
        cm = confusion(y_true, pred)
        row = dict(zip(["protocol", "nfeat", "variant", "model", "seed"], key))
        row.update(metrics_from_cm(cm))
        row.update(auc_scores(y_true, proba))
        fold_acc = [metrics_from_cm(confusion(y[splits[(key[0], r["job"].fold)][1]], r["pred"]))["accuracy"]
                    for r in rs]
        row.update(n_folds=len(rs), fold_accuracy_std=float(np.std(fold_acc)),
                   fit_seconds=float(np.sum([r["fit_seconds"] for r in rs])),
                   converged=all(r["converged"] for r in rs), n_iter=max(r["n_iter"] for r in rs),
                   device=rs[0]["device"], cm=json.dumps(cm.tolist()))
        rows.append(row)
        pooled[key] = (y_true, pred, proba, cm)
    df = pd.DataFrame(rows)
    # Multi-seed models: add a mean row ("seed" = -1) plus the spread of plain accuracy.
    extra = []
    for key, g in df.groupby(["protocol", "nfeat", "variant", "model"]):
        if len(g) > 1:
            num = g.select_dtypes("number").mean()
            row = {**dict(zip(["protocol", "nfeat", "variant", "model"], key)), **num.to_dict()}
            row.update(seed=-1, seed_accuracy_std=float(g["accuracy"].std(ddof=0)),
                       converged=bool(g["converged"].all()), device=g["device"].iloc[0],
                       cm=json.dumps(np.sum([json.loads(c) for c in g["cm"]], axis=0).tolist()))
            extra.append(row)
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df, pooled


def _pct(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.3f}"


def summary_rows(df: pd.DataFrame, protocol: str, nfeat: int, variant: str = "default") -> pd.DataFrame:
    """Main row per model: seed -1 (mean) for multi-seed models, the only seed otherwise."""
    g = df[(df.protocol == protocol) & (df.nfeat == nfeat) & (df.variant == variant)]
    keep = [grp[grp.seed == -1] if (grp.seed == -1).any() else grp for _, grp in g.groupby("model", sort=False)]
    return pd.concat(keep) if keep else g


def model_order(models) -> list[str]:
    base = ["LR", "LR (max_iter=5000)", "LDA", "SVM", "MLP"]
    return sorted(models, key=lambda m: (base.index(m) if m in base else 99, m))


def write_counts(ds, splits: dict, cfg: dict, out: Path) -> None:
    lines = ["# Beat counts vs. paper Table 1", "",
             f"Label map: `{cfg['data']['label_map']}`; records: {len(np.unique(ds.record))} "
             f"(excluded: {', '.join(cfg['data']['exclude_records'])}).", "",
             "## All beats", "", "| Class | Paper total | Ours | Diff |", "|---|---:|---:|---:|"]
    counts = np.bincount(ds.y, minlength=5)
    for i, c in enumerate(CLASSES):
        lines.append(f"| {c} | {PAPER_COUNTS[c][0]} | {counts[i]} | {counts[i] - PAPER_COUNTS[c][0]:+d} |")
    lines.append(f"| All | {sum(v[0] for v in PAPER_COUNTS.values())} | {counts.sum()} | "
                 f"{counts.sum() - sum(v[0] for v in PAPER_COUNTS.values()):+d} |")
    for p in cfg["protocols"]:
        folds = sorted(f for (pp, f) in splits if pp == p)
        tr, te = splits[(p, folds[0])]
        title = PROTOCOL_NAMES[p] + (" (fold 1 of 5 shown; folds differ by at most 1 beat per class)" if p == "C" else "")
        lines += ["", f"## Protocol {p}: {title}", "",
                  "| Class | Paper train | Ours train | Diff | Paper test | Ours test | Diff |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        ctr, cte = np.bincount(ds.y[tr], minlength=5), np.bincount(ds.y[te], minlength=5)
        for i, c in enumerate(CLASSES):
            _, ptr, pte = PAPER_COUNTS[c]
            lines.append(f"| {c} | {ptr} | {ctr[i]} | {ctr[i] - ptr:+d} | {pte} | {cte[i]} | {cte[i] - pte:+d} |")
        lines.append(f"| All | 51527 | {ctr.sum()} | {ctr.sum() - 51527:+d} | 49825 | {cte.sum()} | "
                     f"{cte.sum() - 49825:+d} |")
    lines += ["", "Dropped beats are listed in `dropped_beats.csv`."]
    (out / "counts_vs_paper.md").write_text("\n".join(lines) + "\n")


def write_selected(ds, splits: dict, cfg: dict, out: Path) -> None:
    k = cfg["selection"]["k"]
    lines = ["# SelectKBest(f_classif, k=20): F-scores", "",
             "Fit on the min-max-scaled training features of each protocol (CV: fold 1), and on all "
             "beats (the paper-faithful / `select_all` variant). F-scores are invariant to the "
             "min-max scaling. The paper kept the features with F > 1000.", ""]
    tables = {}
    fits = [(f"Protocol {p} train", splits[(p, 0)][0]) for p in cfg["protocols"]]
    fits.append(("All beats", np.arange(len(ds.y))))
    for label, idx in fits:
        sel = fit_selector(fit_scaler(ds.X[idx]).transform(ds.X[idx]), ds.y[idx], k)
        tables[label] = {n: (s, m) for n, s, m in fscore_table(sel, FEATURE_NAMES)}
        tag = label.split()[1] if label.startswith("Protocol") else "all"
        plot_fscores(FEATURE_NAMES, [tables[label][n][0] for n in FEATURE_NAMES],
                     [tables[label][n][1] for n in FEATURE_NAMES], out / f"fscores_{tag}.png",
                     title=f"F-scores, {label}")
    labels = list(tables)
    lines += ["| # | Feature | " + " | ".join(labels) + " |", "|---:|---|" + "---:|" * len(labels)]
    first = tables[labels[0]]
    for name in sorted(FEATURE_NAMES, key=lambda n: -first[n][0]):
        cells = [f"{'**' if t[name][1] else ''}{t[name][0]:.1f}{'**' if t[name][1] else ''}"
                 for t in tables.values()]
        lines.append(f"| {FEATURE_NAMES.index(name) + 1} | {name} | " + " | ".join(cells) + " |")
    lines += ["", "**Bold** = selected (top 20). Features with F > 1000: " +
              "; ".join(f"{lab}: {sum(s > 1000 for s, _ in t.values())}" for lab, t in tables.items()) + "."]
    (out / "selected_features.md").write_text("\n".join(lines) + "\n")


def write_summary(df: pd.DataFrame, cfg: dict, device: str, out: Path, elapsed: float) -> None:
    L = ["# Results summary", "",
         f"scikit-learn {sklearn.__version__}, Python {platform.python_version()}, device `{device}` "
         f"(LR/SVM on GPU only when device is `cuda`). Label map `{cfg['data']['label_map']}`. "
         f"Wall time for the model jobs run this session: {elapsed / 60:.1f} min.", "",
         "Paper columns are Tables 3/4. **Ours (paper-style)** uses the definitions the paper's numbers "
         "satisfy: Accuracy = mean one-vs-rest accuracy, Sensitivity = F1 = micro recall (= plain "
         "accuracy), Specificity = micro specificity. MLP = mean over seeds "
         f"{cfg['models']['mlp_seeds']}. Protocol C pools the out-of-fold predictions of all 5 folds. "
         "AUC: macro one-vs-rest over the classes present in the test set (micro in brackets). "
         "Rows with Δ > 1 pp between our paper-style accuracy and the paper's are marked ⚠.", ""]
    L += ["## Is the micro-averaging hypothesis consistent with the paper's own tables?", "",
          "Take e = 1 - Sensitivity from each paper row and predict the other columns.", "",
          "| Table | Model | e (%) | Acc paper | 1-2e/5 | Spec paper | 1-e/4 | F1 paper | 1-e |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for nfeat, tab in PAPER_RESULTS.items():
        for m, (acc, se, sp, f1) in tab.items():
            e = 1 - se / 100
            pm = paper_metrics_from_error(e)
            L.append(f"| {3 if nfeat == 57 else 4} ({nfeat}F) | {m} | {100 * e:.3f} | {acc:.3f} | "
                     f"{100 * pm['accuracy']:.3f} | {sp:.3f} | {100 * pm['specificity']:.3f} | {f1:.3f} | "
                     f"{100 * pm['f1']:.3f} |")
    for p in cfg["protocols"]:
        for nfeat in cfg["nfeat"]:
            g = summary_rows(df, p, nfeat)
            if g.empty:
                continue
            L += ["", f"## Protocol {p} ({PROTOCOL_NAMES[p]}), {nfeat} features", "",
                  "| Model | Acc paper | Acc ours | Se paper | Se ours | Sp paper | Sp ours | F1 paper | F1 ours "
                  "| AUC paper | AUC ours | Plain acc | Macro Se | Macro F1 | Se S | Se V | Se F | Majority |",
                  "|---|" + "---:|" * 17]
            for m in model_order(g.model.unique()):
                r = g[g.model == m].iloc[0]
                base = m.split(" ")[0]
                pa = PAPER_RESULTS.get(nfeat, {}).get(base, (np.nan,) * 4)
                pauc = PAPER_AUC.get(nfeat, {}).get(base, np.nan)
                flag = " ⚠" if abs(100 * r.ovr_accuracy - pa[0]) > 1 else ""
                std = f" ± {100 * r.seed_accuracy_std:.2f}" if "seed_accuracy_std" in r and not pd.isna(r.get("seed_accuracy_std")) else ""
                L.append(
                    f"| {m}{'' if r.converged else ' (no conv.)'}{flag} | {pa[0]:.3f} | {_pct(r.ovr_accuracy)} | "
                    f"{pa[1]:.3f} | {_pct(r.micro_sensitivity)} | {pa[2]:.3f} | {_pct(r.micro_specificity)} | "
                    f"{pa[3]:.3f} | {_pct(r.micro_f1)} | {pauc:.2f} | {100 * r.auc_macro:.2f} "
                    f"({100 * r.auc_micro:.2f}) | {_pct(r.accuracy)}{std} | {_pct(r.macro_sensitivity)} | "
                    f"{_pct(r.macro_f1)} | {_pct(r.se_S)} | {_pct(r.se_V)} | {_pct(r.se_F)} | "
                    f"{_pct(r.majority_accuracy)} |")
            L += ["", f"Confusion matrices: `confusion_{p}_{nfeat}.png`; ROC: `roc_{p}_{nfeat}.png`; "
                      f"all metrics and averaging schemes: `results_{p}_{nfeat}.csv`."]
    variants = [v for v in ("default", "scaler_all", "select_all", "no_feature35") if v in set(df.variant)]
    if len(variants) > 1:
        L += ["", "## Ablations (plain multiclass accuracy / macro F1 / S sensitivity, %)", "",
              "`scaler_all` = min-max fit on all beats (paper-faithful); `select_all` = SelectKBest fit "
              "on all beats; `no_feature35` = without the R-time feature. Ablations use the first MLP "
              "seed; the `default` column here uses the same seed for comparability.", "",
              "| Protocol | Features | Model | " + " | ".join(variants) + " |",
              "|---|---:|---|" + "---|" * len(variants)]
        first_seed = cfg["models"]["mlp_seeds"][0]
        for p in cfg["protocols"]:
            for nfeat in cfg["nfeat"]:
                g = df[(df.protocol == p) & (df.nfeat == nfeat)]
                for m in model_order(g.model.unique()):
                    cells = []
                    for v in variants:
                        h = g[(g.model == m) & (g.variant == v)]
                        h = h[h.seed == first_seed] if (h.seed == first_seed).any() else h[h.seed != -1]
                        cells.append("" if h.empty else
                                     f"{_pct(h.accuracy.iloc[0])} / {_pct(h.macro_f1.iloc[0])} / {_pct(h.se_S.iloc[0])}")
                    if any(cells[1:]) or len(cells) == 1:
                        L.append(f"| {p} | {nfeat} | {m} | " + " | ".join(cells) + " |")
    (out / "results_summary.md").write_text("\n".join(L) + "\n")


def write_figures_for_results(df: pd.DataFrame, pooled: dict, cfg: dict, out: Path) -> None:
    first_seed = cfg["models"]["mlp_seeds"][0]
    for p in cfg["protocols"]:
        for nfeat in cfg["nfeat"]:
            cms, curves = {}, {}
            for m in model_order({k[3] for k in pooled if k[0] == p and k[1] == nfeat and k[2] == "default"}):
                if m.startswith("LR (max_iter"):
                    continue
                seeds = sorted(k[4] for k in pooled if k[:4] == (p, nfeat, "default", m))
                seed = first_seed if first_seed in seeds else seeds[0]
                y_true, _, proba, cm = pooled[(p, nfeat, "default", m, seed)]
                label = m if len(seeds) == 1 else f"{m} (seed {seed})"
                cms[label] = cm
                curves[label] = roc_curves(y_true, proba)
            if cms:
                ttl = f"Protocol {p} ({PROTOCOL_NAMES[p]}), {nfeat} features"
                plot_confusions(cms, out / f"confusion_{p}_{nfeat}.png", ttl)
                plot_rocs(curves, out / f"roc_{p}_{nfeat}.png", ttl)


def write_signal_figures(ds, cfg: dict, fig_dir: Path) -> None:
    rf, raw, base, clean, _ = record_features(cfg["output"]["baseline_record"], cfg, PROJECT_ROOT, True)
    plot_baseline(raw, base, clean, cfg["data"]["fs"], fig_dir / f"baseline_{rf.name}.png", rf.name)
    rng = np.random.default_rng(cfg["seed"])
    r0 = cfg["preprocess"]["window_before"]

    rf, *_, beats = record_features(cfg["output"]["example_record"], cfg, PROJECT_ROOT, True)
    bb, pp = {}, {}
    for c in np.unique(rf.y):
        i = rng.choice(np.flatnonzero(rf.y == c), min(3, int(np.sum(rf.y == c))), replace=False)
        bb[CLASSES[c]], pp[CLASSES[c]] = beats[i], rf.points[i]
    plot_fiducials(bb, pp, cfg["data"]["fs"], r0, fig_dir / f"fiducials_{rf.name}.png",
                   f"Record {rf.name}: detected fiducial points")

    bb, pp, src = {}, {}, []
    for c, name in enumerate(CLASSES):
        recs, cnt = np.unique(ds.record[ds.y == c], return_counts=True)
        if not len(recs):
            continue
        rec = recs[np.argmax(cnt)]
        rf, *_, beats = record_features(rec, cfg, PROJECT_ROOT, True)
        i = rng.choice(np.flatnonzero(rf.y == c), min(3, int(np.sum(rf.y == c))), replace=False)
        bb[name], pp[name] = beats[i], rf.points[i]
        src.append(f"{name}: {rec}")
    plot_fiducials(bb, pp, cfg["data"]["fs"], r0, fig_dir / "fiducials_by_class.png",
                   "Example beats per class with fiducial points (" + ", ".join(src) + ")")


# ----------------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--protocols", nargs="+", choices=["A", "B", "C"])
    ap.add_argument("--models", nargs="+", choices=["LR", "LDA", "SVM", "MLP"])
    ap.add_argument("--nfeat", nargs="+", type=int, choices=[57, 20])
    ap.add_argument("--mlp-seeds", nargs="+", type=int)
    ap.add_argument("--label-map", choices=["paper", "aami"])
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "gpu"])
    ap.add_argument("--workers", type=int, help="model-fit processes")
    ap.add_argument("--blas-threads", type=int)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--no-figures", action="store_true", help="skip signal/fiducial figures")
    ap.add_argument("--results-dir")
    args = ap.parse_args(argv)

    cfg = load_config(_path(args.config))
    for key, val in (("protocols", args.protocols),):
        if val:
            cfg[key] = val
    if args.models:
        cfg["models"]["names"] = args.models
    if args.mlp_seeds:
        cfg["models"]["mlp_seeds"] = args.mlp_seeds
    if args.label_map:
        cfg["data"]["label_map"] = args.label_map
    if args.device:
        cfg["compute"]["device"] = args.device
    if args.workers:
        cfg["compute"]["cpu_workers"] = args.workers
    if args.blas_threads:
        cfg["compute"]["blas_threads"] = args.blas_threads
    if args.results_dir:
        cfg["output"]["results_dir"] = args.results_dir
    cfg["nfeat"] = args.nfeat or [57, cfg["selection"]["k"]]
    cfg["run_ablations"] = not args.no_ablations
    if cfg["normalization"]["scope"] != "train" or cfg["selection"]["scope"] != "train":
        raise ValueError("set normalization/selection scope to 'train'; the 'all' variants run as ablations")

    out = _path(cfg["output"]["results_dir"])
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_root = _path(cfg["output"]["cache_dir"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log", mode="w")])
    log.info("scikit-learn %s, numpy %s, Python %s", sklearn.__version__, np.__version__,
             platform.python_version())
    with open(out / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    device = resolve_device(cfg["compute"]["device"])
    records = used_records(cfg["data"]["exclude_records"])
    t0 = time.perf_counter()
    ds = build_dataset(records, cfg, PROJECT_ROOT, cache_root, workers=int(cfg["compute"]["feature_workers"]))
    log.info("features: %d beats x %d from %d records in %.1fs (label map %s); counts %s",
             *ds.X.shape, len(records), time.perf_counter() - t0, cfg["data"]["label_map"],
             dict(zip(CLASSES, np.bincount(ds.y, minlength=5).tolist())))
    log.info("T search window clipped at the beat-window end for %d beats", ds.t_clipped)
    pd.DataFrame(ds.drops, columns=["record", "sample", "symbol", "reason"]).to_csv(out / "dropped_beats.csv", index=False)
    for d in ds.drops:
        log.info("dropped: record %s sample %s symbol %s: %s", *d)

    splits = {}
    for p in cfg["protocols"]:
        for f, (tr, te) in enumerate(make_splits(p, ds.y, ds.record, cfg["seed"], cfg["cv_folds"])):
            splits[(p, f)] = (tr, te)
    if "A" in cfg["protocols"]:
        tr, te = splits[("A", 0)]
        assert not set(ds.record[tr]) & set(ds.record[te]), "protocol A: a record is in train and test"
    write_counts(ds, splits, cfg, out)
    write_selected(ds, splits, cfg, out)
    if not args.no_figures:
        write_signal_figures(ds, cfg, fig_dir)

    data_hash = feature_cache_key(cfg) + cfg["data"]["label_map"] + str(cfg["seed"])
    data_path = cache_root / f"design_{data_hash}.npz"
    arrays = {"X": ds.X, "y": ds.y}
    for (p, f), (tr, te) in splits.items():
        arrays[f"{p}{f}_tr"], arrays[f"{p}{f}_te"] = tr, te
    np.savez(data_path, **arrays)
    job_cache = cache_root / "jobs"
    job_cache.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(cfg, {p: sum(1 for (pp, _) in splits if pp == p) for p in cfg["protocols"]})
    t1 = time.perf_counter()
    results = run_all(jobs, cfg, device, data_path, data_hash, job_cache)
    elapsed = time.perf_counter() - t1

    df, pooled = aggregate(results, ds.y, splits)
    cols = [c for c in df.columns if c != "cm"] + ["cm"]
    for p in cfg["protocols"]:
        for nfeat in cfg["nfeat"]:
            g = df[(df.protocol == p) & (df.nfeat == nfeat)]
            g[cols].sort_values(["variant", "model", "seed"]).to_csv(out / f"results_{p}_{nfeat}.csv", index=False)
    write_summary(df, cfg, device, out, elapsed)
    write_figures_for_results(df, pooled, cfg, out)
    log.info("done in %.1f min; summary in %s", (time.perf_counter() - t0) / 60, out / "results_summary.md")


if __name__ == "__main__":
    main()
