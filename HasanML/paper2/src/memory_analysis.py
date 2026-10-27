"""Memory-requirements analysis for the paper-2 5-class Logistic Regression.

    python -m src.memory_analysis                       # writes results/memoryRequirementsAnalysis.txt
    python -m src.memory_analysis --skip-svm --out other.txt

Targets: cloud (Python, batch), Raspberry Pi (Python / NumPy, streaming) and ESP32
(C, float32, static buffers, streaming). It also sizes LR as a knowledge-distillation
student or teacher.

What is measured here (on this host) and what is analytic:
  measured  model sizes (sklearn pickle, raw arrays), peak RSS of each runtime stack
            (separate child processes), LR training memory and time, per-beat latency,
            and the accuracy cost of float32 / float16 / int8 weights on the
            patient-level (DS1/DS2) test set
  analytic  the streaming on-device budget (every buffer listed with its size), compute
            per beat, and device fit against nominal device memory (DEVICES below;
            confirm against the actual board)

The deployed model is the default pipeline of the replication: min-max scaler and
SelectKBest(k=20) fit on DS1, LogisticRegression(max_iter=5000) (the default 100
iterations do not converge), for both the 57- and 20-feature variants.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .data import CLASSES, make_splits, used_records
from .features import FEATURE_NAMES, build_dataset
from .metrics import confusion, metrics_from_cm
from .select import fit_scaler, fit_selector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
K = len(CLASSES)
KB, MB = 1024, 1024 ** 2

# Nominal device memory, bytes. Confirm against the board actually used.
DEVICES = {
    "ESP32 (WROOM-32, no PSRAM)": dict(ram=520 * KB, usable=160 * KB, flash=4 * MB,
        note="520 KB SRAM total; ~160 KB is a conservative heap budget with Wi-Fi/BLE "
             "running; FPU is single-precision only (double is software-emulated)"),
    "ESP32-S3 (8 MB PSRAM)": dict(ram=512 * KB + 8 * MB, usable=256 * KB, flash=8 * MB,
        note="512 KB SRAM + PSRAM; budget counts internal SRAM only"),
    "Raspberry Pi Zero 2 W": dict(ram=512 * MB, usable=300 * MB, flash=None,
        note="512 MB RAM shared with the OS; ~300 MB left for an application"),
    "Raspberry Pi 4/5 (2 GB)": dict(ram=2048 * MB, usable=1500 * MB, flash=None,
        note="smallest common Pi 4/5 RAM size"),
    "Cloud (small instance, 1 GB)": dict(ram=1024 * MB, usable=700 * MB, flash=None,
        note="e.g. the smallest general-purpose VM / container size"),
}


def fmt(n: float) -> str:
    """Human-readable bytes."""
    n = float(n)
    for unit, size in (("GB", 1024 ** 3), ("MB", MB), ("KB", KB)):
        if abs(n) >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n:.0f} B"


def table(rows: list[list], header: list[str]) -> str:
    """Plain-text table; columns whose values all start with a digit are right-aligned."""
    cells = [[str(c) for c in r] for r in [header] + rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(header))]
    num = [i > 0 and all(r[i][:1].isdigit() for r in cells[1:] if r[i]) for i in range(len(header))]
    line = lambda r: "  ".join(c.rjust(w[i]) if num[i] else c.ljust(w[i]) for i, c in enumerate(r))
    return "\n".join([line(cells[0]), "  ".join("-" * x for x in w)] + [line(r) for r in cells[1:]])


# ----------------------------------------------------------------------------- data / model

def prepare(cfg: dict, work: Path) -> dict:
    """Fit the deployed pipelines on DS1 and save everything the child processes need."""
    ds = build_dataset(used_records(cfg["data"]["exclude_records"]), cfg, PROJECT_ROOT,
                       PROJECT_ROOT / cfg["output"]["cache_dir"], workers=int(cfg["compute"]["feature_workers"]))
    tr, te = make_splits("A", ds.y, ds.record, cfg["seed"], cfg["cv_folds"])[0]
    scaler = fit_scaler(ds.X[tr])
    Xs = scaler.transform(ds.X)
    selector = fit_selector(Xs[tr], ds.y[tr], cfg["selection"]["k"])
    out = {"n_train": len(tr), "n_test": len(te), "n_all": len(ds.y), "models": {}}
    np.savez(work / "train.npz", X57=Xs[tr], X20=Xs[tr][:, selector.get_support()], y=ds.y[tr],
             Xall=Xs, yall=ds.y)
    for nfeat in (57, 20):
        cols = np.arange(57) if nfeat == 57 else np.flatnonzero(selector.get_support())
        lr = LogisticRegression(max_iter=5000, random_state=cfg["seed"]).fit(Xs[tr][:, cols], ds.y[tr])
        pipe = {"scaler": scaler, "selector": selector if nfeat == 20 else None, "lr": lr}
        joblib.dump(pipe, work / f"lr_{nfeat}.joblib")
        # Minimal float32 artifact: everything inference needs, nothing else.
        np.savez(work / f"lr_{nfeat}_f32.npz", coef=lr.coef_.astype(np.float32),
                 intercept=lr.intercept_.astype(np.float32),
                 data_min=scaler.data_min_[cols].astype(np.float32),
                 scale=scaler.scale_[cols].astype(np.float32), cols=cols.astype(np.uint8))
        out["models"][nfeat] = dict(lr=lr, cols=cols, Xte_raw=ds.X[te], Xte=Xs[te][:, cols], yte=ds.y[te],
                                    joblib_bytes=(work / f"lr_{nfeat}.joblib").stat().st_size,
                                    npz_bytes=(work / f"lr_{nfeat}_f32.npz").stat().st_size,
                                    n_iter=int(np.max(lr.n_iter_)))
    out["dataset"] = ds
    out["tr"], out["te"] = tr, te
    out["selected"] = [FEATURE_NAMES[i] for i in np.flatnonzero(selector.get_support())]
    return out


def precision_study(m: dict) -> list[list]:
    """Accuracy of the same LR with float64 / float32 / float16 / int8 weights (and int8 inputs)."""
    lr, X, y = m["lr"], m["Xte"], m["yte"]
    W, b = lr.coef_, lr.intercept_
    ref = np.argmax(X @ W.T + b, axis=1)

    def row(name, pred, weight_bytes):
        mt = metrics_from_cm(confusion(y, pred))
        return [name, fmt(weight_bytes), f"{100 * mt['accuracy']:.2f}", f"{100 * mt['macro_f1']:.2f}",
                f"{100 * mt['se_S']:.1f}", f"{100 * mt['se_V']:.1f}", f"{100 * np.mean(pred == ref):.3f}"]

    n = W.size + b.size
    rows = [row("float64 (sklearn)", ref, 8 * n)]
    X32, W32, b32 = X.astype(np.float32), W.astype(np.float32), b.astype(np.float32)
    rows.append(row("float32", np.argmax(X32 @ W32.T + b32, axis=1), 4 * n))
    W16, b16 = W.astype(np.float16), b.astype(np.float16)
    rows.append(row("float16 weights, fp32 math", np.argmax(X32 @ W16.astype(np.float32).T + b16.astype(np.float32), axis=1), 2 * n))
    # Symmetric per-class int8 weights; bias kept in float32 (5 values).
    s = np.abs(W).max(axis=1, keepdims=True) / 127.0
    Wq = np.clip(np.round(W / s), -127, 127).astype(np.int8)
    rows.append(row("int8 weights (per-class scale), fp32 input",
                    np.argmax(X32 @ (Wq.astype(np.float32) * s.astype(np.float32)).T + b32, axis=1),
                    W.size + 4 * K + 4 * K))
    # Fully integer dot product: inputs in [0, 1] -> uint8 (/255), int32 accumulators.
    xq = np.clip(np.round(np.clip(X, 0, 1) * 255), 0, 255).astype(np.int32)
    acc = xq @ Wq.astype(np.int32).T
    logits = acc.astype(np.float64) * (s.T / 255.0) + b
    rows.append(row("int8 weights + uint8 inputs (integer MACs)", np.argmax(logits, axis=1),
                    W.size + 4 * K + 4 * K))
    return rows


# ----------------------------------------------------------------------------- child-process measurements

# Peak RSS of this process image. VmHWM resets on exec; ru_maxrss would carry over the forked
# parent's peak on Linux, which is why it is not used.
CHILD_PRELUDE = ("import json, time, sys\n"
                 "def rss():\n"
                 "    for line in open('/proc/self/status'):\n"
                 "        if line.startswith('VmHWM:'):\n"
                 "            return int(line.split()[1]) * 1024\n")


def run_child(code: str) -> dict:
    """Run `code` in a fresh interpreter (project root as cwd); it must print one JSON line."""
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    res = subprocess.run([sys.executable, "-c", CHILD_PRELUDE + textwrap.dedent(code)], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, env=env, timeout=3600)
    if res.returncode:
        raise RuntimeError(f"child failed:\n{res.stderr[-2000:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def runtime_stacks(work: Path) -> list[list]:
    """Peak RSS of each way to run inference, each in its own fresh process."""
    w = str(work)
    stages = [
        ("Python interpreter only", "print(json.dumps({'rss': rss()}))"),
        ("+ NumPy", "import numpy\nprint(json.dumps({'rss': rss()}))"),
        ("+ NumPy + scikit-learn (LR, MinMaxScaler imported)",
         "import numpy\nfrom sklearn.linear_model import LogisticRegression\n"
         "from sklearn.preprocessing import MinMaxScaler\nprint(json.dumps({'rss': rss()}))"),
        ("sklearn pipeline: load joblib (20 F) + predict 1 beat",
         f"import numpy as np, joblib\np=joblib.load('{w}/lr_20.joblib')\n"
         "x=np.random.default_rng(0).random((1,57))\n"
         "z=p['selector'].transform(p['scaler'].transform(x)); p['lr'].predict(z)\nprint(json.dumps({'rss': rss()}))"),
        ("NumPy-only: load float32 npz (20 F) + predict 1 beat",
         f"import numpy as np\nm=np.load('{w}/lr_20_f32.npz')\n"
         "x=np.random.default_rng(0).random(57).astype(np.float32)[m['cols']]\n"
         "z=(x-m['data_min'])*m['scale']; int(np.argmax(m['coef']@z+m['intercept']))\nprint(json.dumps({'rss': rss()}))"),
        ("Batch cloud inference: 49,825 test beats, sklearn (57 F)",
         f"import numpy as np, joblib\np=joblib.load('{w}/lr_57.joblib')\n"
         "x=np.random.default_rng(0).random((49825,57))\nr0=rss()\n"
         "p['lr'].predict_proba(p['scaler'].transform(x))\nprint(json.dumps({'rss': rss(), 'r0': r0}))"),
    ]
    rows, raw = [], {}
    base = None
    for key, (name, code) in zip(("bare", "numpy", "sklearn_import", "sklearn", "numpy_model", "batch"), stages):
        r = run_child(code)
        base = base or r["rss"]
        raw[key] = r["rss"]
        rows.append([name, fmt(r["rss"]), fmt(r["rss"] - base)])
    return rows, raw


def training_memory(work: Path) -> list[list]:
    """Peak RSS and time of LR training (cloud) in fresh processes, 1 BLAS thread."""
    w = str(work)
    jobs = [("DS1 train, 57 features", "X57", "y"), ("DS1 train, 20 features", "X20", "y"),
            ("all 44 records, 57 features", "Xall", "yall")]
    rows = []
    for name, xk, yk in jobs:
        r = run_child(f"""
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            d=np.load('{w}/train.npz'); X=d['{xk}']; y=d['{yk}']; r0=rss()
            t=time.perf_counter(); m=LogisticRegression(max_iter=5000, random_state=0).fit(X,y)
            print(json.dumps({{'r0':r0,'r1':rss(),'t':time.perf_counter()-t,'n':X.shape[0],'f':X.shape[1],
                               'xb':X.nbytes,'it':int(np.max(m.n_iter_))}}))""")
        rows.append([name, f"{r['n']:,} x {r['f']}", fmt(r["xb"]), fmt(r["r1"]), fmt(r["r1"] - r["r0"]),
                     f"{r['t']:.1f} s", r["it"]])
    return rows


def latency(m: dict, repeats: int = 2000) -> dict:
    """Per-beat inference time on this host: NumPy float32 (no sklearn) and sklearn."""
    lr, X = m["lr"], m["Xte"]
    W, b = lr.coef_.astype(np.float32), lr.intercept_.astype(np.float32)
    x = np.ascontiguousarray(X[:repeats], dtype=np.float32)
    best = []
    for _ in range(3):                                     # first pass warms caches
        t = time.perf_counter()
        for i in range(repeats):
            int(np.argmax(W @ x[i] + b))
        best.append((time.perf_counter() - t) / repeats)
    t_np = min(best)
    t = time.perf_counter()
    for i in range(200):
        lr.predict(X[i:i + 1])
    t_sk = (time.perf_counter() - t) / 200
    t = time.perf_counter()
    lr.predict(X)
    t_batch = (time.perf_counter() - t) / len(X)
    return {"numpy": t_np, "sklearn": t_sk, "batch": t_batch}


def feature_extraction_cost(cfg: dict) -> dict:
    """Peak RSS and per-beat time of offline (whole-record) feature extraction, record 208."""
    cfg_json = json.dumps(cfg)
    r = run_child(f"""
        import json
        from pathlib import Path
        from src.features import record_features
        cfg=json.loads({cfg_json!r}); r0=rss(); t=time.perf_counter()
        rf=record_features('208', cfg, Path('.'))
        print(json.dumps({{'r0':r0,'r1':rss(),'t':time.perf_counter()-t,'n':len(rf.y)}}))""")
    return r


# ----------------------------------------------------------------------------- analytic streaming budget

def streaming_budget(nfeat: int, cfg: dict) -> list[tuple[str, int, str]]:
    """Static RAM for causal, beat-by-beat inference in C with float32 (ESP32 / bare-metal)."""
    f4 = 4
    win = cfg["preprocess"]["window_before"] + cfg["preprocess"]["window_after"] + 1     # 256
    n_sos = (cfg["preprocess"]["baseline"]["order"] + 1) // 2                            # biquads
    pend = cfg["features"]["rr_local_n"] + 1          # beats waiting for Post-RR / 5 future RRs
    rr_ring = 2 * cfg["features"]["rr_local_n"] + 1
    return [
        ("ADC sample ring (int16, 2 x window: R-peak search + 128-sample look-ahead)", 2 * win * 2,
         "raw 11-bit samples"),
        ("Baseline filter: 2nd-order sections, coefficients + state (float32)", n_sos * (6 + 2) * f4,
         f"{n_sos} biquads; causal sosfilt"),
        ("Baseline-corrected sample ring (float32, 2 x window)", 2 * win * f4, ""),
        ("R-peak detector state (Pan-Tompkins-style: band-pass, derivative, 150 ms MWI)",
         (54 + 32) * f4 + 64, "needed on device; the paper used annotations"),
        ("Beat window copy for fiducial detection (float32)", win * f4, ""),
        ("Fiducial scratch: |x - iso|, |dx|, median sort buffer (float32)", 3 * win * f4, ""),
        ("Fiducial indices (11 x int16) + iso level", 11 * 2 + f4, ""),
        (f"Pending feature vectors: {pend} beats x 57 float32 (wait for Post-RR + 5 future RRs)",
         pend * 57 * f4, "morphology features computed once, RR features filled in later"),
        (f"R-time ring for RR features ({rr_ring} x uint32) + running record RR mean", rr_ring * 4 + 2 * 8,
         "'RR average' becomes a running mean"),
        (f"Scaler for the {nfeat} model inputs: min + scale (float32)", 2 * nfeat * f4, ""),
        ("Selected-feature index list (uint8)" if nfeat < 57 else "Selected-feature index list (none)",
         nfeat if nfeat < 57 else 0, ""),
        (f"LR weights {K} x {nfeat} + {K} bias (float32)", (K * nfeat + K) * f4, ""),
        ("Scaled input vector + 5 logits (float32)", (nfeat + K) * f4, ""),
    ]


# ----------------------------------------------------------------------------- distillation

def distillation(models: dict, n_train: int, n_all: int, svm_info: dict | None) -> list[str]:
    L = []
    rows = []
    for nfeat in (57, 20):
        lr_p = K * nfeat + K
        mlp_p = nfeat * 100 + 100 + 100 * K + K     # sklearn default hidden_layer_sizes=(100,)
        rows.append([f"LR, {nfeat} F (student or teacher)", f"{lr_p:,}", fmt(4 * lr_p), fmt(lr_p),
                     f"{K * nfeat:,}"])
        rows.append([f"MLP (100 hidden, sklearn default), {nfeat} F", f"{mlp_p:,}", fmt(4 * mlp_p), fmt(mlp_p),
                     f"{nfeat * 100 + 100 * K:,}"])
    if svm_info:
        f = svm_info["nfeat"]
        n_sv = svm_info["n_sv"]
        sv_p = n_sv * f + n_sv * (K - 1) + K * (K - 1) // 2
        rows.append([f"SVM RBF (measured on DS1), {f} F: {n_sv:,} support vectors", f"{sv_p:,}",
                     fmt(4 * sv_p), fmt(sv_p), f"{n_sv * (f + 1) * K * (K - 1) // 2:,} (OvO, worst case)"])
    L.append(table(rows, ["Model", "Parameters", "float32", "int8", "MACs / beat"]))
    L += ["",
          "Soft-label (teacher output) storage for a distillation transfer set, float32 x 5 classes:",
          f"  DS1 training set ({n_train:,} beats): {fmt(n_train * K * 4)};"
          f"  all 44 records ({n_all:,} beats): {fmt(n_all * K * 4)}",
          "  (float16 halves this; storing top-1 + max prob only needs 5 bytes per beat)."]
    return L


def svm_support_vectors(m20: dict, X20: np.ndarray, y: np.ndarray) -> dict:
    t = time.perf_counter()
    svm = SVC(random_state=0).fit(X20, y)       # probability=False: same support vectors, 5x faster
    return {"nfeat": X20.shape[1], "n_sv": int(svm.n_support_.sum()),
            "per_class": dict(zip([CLASSES[c] for c in svm.classes_], svm.n_support_.tolist())),
            "fit_s": time.perf_counter() - t}


# ----------------------------------------------------------------------------- report

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/memoryRequirementsAnalysis.txt")
    ap.add_argument("--skip-svm", action="store_true", help="skip the SVM teacher fit (~1-3 min)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(PROJECT_ROOT / args.config))
    work = PROJECT_ROOT / cfg["output"]["cache_dir"] / "memory_analysis"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("fitting deployed pipelines ...", flush=True)
    P = prepare(cfg, work)
    print("measuring runtime stacks, training, latency ...", flush=True)
    stacks, stack_raw = runtime_stacks(work)
    stack_rss = {"numpy": stack_raw["numpy_model"], "sklearn": stack_raw["sklearn"]}
    train_rows = training_memory(work)
    lat = {nf: latency(P["models"][nf]) for nf in (57, 20)}
    fx = feature_extraction_cost(cfg)
    svm_info = None
    if not args.skip_svm:
        print("fitting SVM teacher for support-vector count ...", flush=True)
        d = np.load(work / "train.npz")
        svm_info = svm_support_vectors(P["models"][20], d["X20"], d["y"])

    L = []
    hr = lambda title: L.extend(["", "=" * 100, title, "=" * 100])
    L += ["MEMORY REQUIREMENTS ANALYSIS: paper-2 5-class Logistic Regression (Yakut & Dogru Bolat 2020)",
          f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python -m src.memory_analysis` "
          f"({time.perf_counter() - t0:.0f} s before writing)",
          f"Host: {platform.machine()}, {platform.processor() or platform.platform()}, Python "
          f"{platform.python_version()}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}",
          "Deployed model: min-max scaler + [SelectKBest k=20] + LogisticRegression(max_iter=5000), fit on DS1 "
          "(22 records, patient-level split); evaluated on DS2 (22 other records).",
          "Numbers marked (measured) come from this host; (analytic) are computed from the algorithm; device "
          "memory figures are nominal and should be confirmed for the exact board."]

    hr("1. MODEL SIZE (measured)")
    rows = []
    for nf in (57, 20):
        m = P["models"][nf]
        p = K * nf + K
        sc = 2 * nf
        rows.append([f"{nf} features", f"{p:,}", f"{sc:,}", fmt(8 * (p + sc)), fmt(4 * (p + sc)),
                     fmt(K * nf + 4 * K + 4 * K + 4 * sc), fmt(m["joblib_bytes"]), fmt(m["npz_bytes"]), m["n_iter"]])
    L.append(table(rows, ["Variant", "LR params", "Scaler params", "float64", "float32",
                          "int8 W + fp32 scaler", "sklearn joblib", "fp32 .npz", "LR iters"]))
    L += ["",
          "The trained classifier itself is tiny: 290 parameters (57 F) or 105 (20 F), i.e. 1.2 KB / 0.4 KB in",
          "float32. The sklearn pickle is larger only because of Python object overhead. Model storage is not a",
          "constraint on any target; the runtime around it (Python, feature extraction buffers) is.",
          f"Selected 20 features (fit on DS1): {', '.join(P['selected'])}."]

    hr("2. NUMERIC PRECISION AND QUANTISATION (measured on the DS2 test set)")
    for nf in (57, 20):
        L += [f"\n{nf} features:", table(precision_study(P["models"][nf]),
              ["Weights / arithmetic", "Weight bytes", "Acc %", "Macro F1 %", "Se S %", "Se V %",
               "Agree w/ fp64 %"])]
    L += ["",
          "float32 is exact for practical purposes, so ESP32 (single-precision FPU only) loses nothing by avoiding",
          "double. int8 rows show what a fully integer-quantised LR would cost; use them to decide whether int8",
          "is worth it. Since the float32 model is only ~1 KB, quantisation is optional for memory and mainly",
          "useful for speed on MCUs without an FPU, or for the student in a distillation setup."]

    hr("3. RUNTIME MEMORY OF EACH SOFTWARE STACK (measured, peak RSS, fresh process each, x86_64)")
    L.append(table(stacks, ["Stack / stage", "Peak RSS", "Above bare Python"]))
    L += ["",
          "Inference latency on this host (per beat, model only, features already computed):"]
    for nf in (57, 20):
        l = lat[nf]
        L.append(f"  {nf} F: NumPy float32 dot {1e6 * l['numpy']:.1f} us | sklearn predict() one beat "
                 f"{1e6 * l['sklearn']:.1f} us | sklearn batch {1e6 * l['batch']:.2f} us/beat")
    L += [f"Offline feature extraction (whole record 208, {fx['n']:,} beats, vectorised NumPy): "
          f"{1e3 * fx['t'] / fx['n']:.3f} ms/beat, peak RSS {fmt(fx['r1'])} "
          f"(+{fmt(fx['r1'] - fx['r0'])} above the imported stack).",
          "On a Raspberry Pi (ARM64) RSS figures are of the same order; latency is ~3-10x higher than on this",
          "x86 host, still far below the >= 300 ms between heartbeats."]

    hr("4. TRAINING MEMORY (cloud; measured, 1 BLAS thread)")
    L.append(table(train_rows, ["Training set", "Beats x features", "X (float64)", "Peak RSS", "RSS added by fit",
                                "Fit time", "Iterations"]))
    L += ["",
          "LR training needs roughly the design matrix plus a few copies of it. A full replication run",
          "(all models, 5-fold CV) needs more because SVM/MLP run in parallel processes, each holding its own",
          "copy of the data. Training belongs in the cloud or on a workstation; a Pi 4/5 could retrain LR",
          "(tens of MB), an ESP32 cannot (the training matrix alone is ~10-25 MB)."]

    hr("5. STREAMING ON-DEVICE INFERENCE BUDGET (analytic, C, float32, static buffers)")
    budgets = {}
    for nf in (57, 20):
        items = streaming_budget(nf, cfg)
        total = sum(b for _, b, _ in items)
        budgets[nf] = total
        L += [f"\n{nf}-feature model:",
              table([[n, f"{b:,} B", note] for n, b, note in items] + [["TOTAL static RAM", f"{total:,} B", fmt(total)]],
                    ["Buffer", "Bytes", "Note"])]
    L += ["",
          "Assumptions behind this budget:",
          "  - The paper filtered offline with MATLAB; zero-phase filtering (sosfiltfilt) needs the future signal,",
          "    so the device must use a causal filter (sosfilt). That changes the signal slightly; re-validate",
          "    accuracy with causal filtering before deploying.",
          "  - The paper used the database's R-peak annotations. A device needs its own R-peak detector",
          "    (budgeted above); detector errors are not reflected in the replication's accuracy.",
          "  - 'Post-RR' and 'RR local average' (5 RR intervals after the beat) mean each beat is classified",
          "    ~6 beats (~5 s) after it occurs; the pending-vector queue above holds those beats.",
          "  - 'RR average' (whole-record mean) becomes a running mean; 'IR time' (feature 35, R time in the",
          "    record) is not meaningful in continuous monitoring and should be dropped (the no_feature35",
          "    ablation shows <= 0.2 pp change for LR).",
          "  - Program code (filter + detector + fiducials + features + LR) is estimated at 20-60 KB of flash.",
          "",
          "Compute per beat (analytic): baseline filter 2 biquads x 5 MACs = 10 MACs/sample (3,600 MAC/s at",
          "360 Hz); fiducial detection + features ~10-20 k simple ops per beat; LR "
          f"{K * 57} MACs (57 F) / {K * 20} MACs (20 F) + 5 exp. At 240 MHz this is well under 1 ms per beat."]

    hr("6. TARGET-BY-TARGET VERDICT")
    rows = []
    for name, d in DEVICES.items():
        if name.startswith("ESP32"):
            need, how = budgets[20], "C, float32, static buffers (20 F)"
        elif name.startswith("Raspberry Pi Zero"):
            need, how = stack_rss["numpy"], "Python + NumPy only"
        else:
            need, how = stack_rss["sklearn"], "Python + scikit-learn"
        fit = f"{fmt(need)} of {fmt(d['usable'])} budget ({100 * need / d['usable']:.1f} %)"
        rows.append([name, how, fit, "fits" if need < d["usable"] else "does NOT fit"])
    L.append(table(rows, ["Target", "Recommended runtime", "Memory need vs budget", "Verdict"]))
    L += ["",
          "Notes per target:"]
    L += [f"  - {n}: {d['note']}." for n, d in DEVICES.items()]
    L += ["",
          "  Cloud: memory is a non-issue for LR. Batch inference on the whole 49,825-beat test set and LR",
          "  training both stay within the RSS figures of sections 3-4. The cloud is the natural place to train the teacher/student",
          "  and to run batch re-analysis.",
          "  Raspberry Pi: any model in this study fits. Prefer the NumPy-only float32 runtime (no sklearn import)",
          "  on a Pi Zero 2 W to keep RSS and start-up time low; feature extraction then dominates CPU time, not",
          "  memory.",
          "  ESP32: LR fits with large headroom (<10 % of a conservative SRAM budget) even with all 57 features.",
          "  The real porting work is in the signal chain (causal filter, R-peak detector, fiducial rules in C),",
          "  not the classifier."]

    hr("7. KNOWLEDGE DISTILLATION SIZING")
    L += distillation(P["models"], P["n_train"], P["n_all"], svm_info)
    if svm_info:
        L.append(f"SVM support vectors per class: {svm_info['per_class']} (fit {svm_info['fit_s']:.0f} s).")
    L += ["",
          "LR as STUDENT: the obvious role. It is the cheapest model that fits every target, and a teacher can",
          "  pass on soft class probabilities. Caveat: sklearn's LogisticRegression cannot train on soft labels;",
          "  a small custom softmax-regression trainer (NumPy or PyTorch, same 5 x F weights, cross-entropy",
          "  against teacher probabilities, temperature T) is needed. Memory for that training is the same as",
          "  in section 4 plus the soft-label array above.",
          "LR as TEACHER: only sensible for an even smaller student (e.g. int8 LR, fewer features, or a",
          "  distilled rule set). Note that in the replication LR is weaker than MLP/SVM when patients overlap,",
          "  and all models are weak on unseen patients (DS2: S sensitivity < 15 %). Distillation transfers the",
          "  teacher's mistakes too, so the teacher should be the best patient-level model available.",
          "Teacher candidates on the targets: an MLP teacher (~25 KB float32 for 57 F) would itself fit an",
          "  ESP32; an RBF-SVM teacher is dominated by its support vectors (see the table) and is cloud/Pi only.",
          "",
          "Accuracy context (patient-level DS1/DS2 test, this exact model): see section 2, float64 row. The",
          "memory analysis shows LR is deployable everywhere; whether it is accurate enough to deploy is a",
          "separate question (see README, 'Feasibility')."]
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
