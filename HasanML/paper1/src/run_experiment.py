"""Run the Standard-ESN PVC replication end to end.

    python -m src.run_experiment --config configs/default.yaml
    python -m src.run_experiment --records 119 --seeds 0 --input spikes --leak 0.99   # quick check

Parallelism: on CPU every reservoir job runs in its own process (the
per-timestep loop holds the GIL, so threads would serialise it); on a CUDA
GPU, jobs batch many reservoirs into one simulation and a small thread pool
keeps the device busy. Record loading and filtering use a thread pool.
"""

import argparse
import dataclasses
import hashlib
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .backend import resolve_device
from .data import load_record
from .encode import build_input
from .esn import JobSpec, Member, run_job
from .evaluate import binary_metrics, plot_record, summarize, write_tables
from .preprocess import filter_without_leakage, highpass_sos, minmax_normalize
from .readout import select_operating_point, threshold_grid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("esn")


@dataclass
class RecordData:
    name: str
    fs: float
    x_norm: np.ndarray       # filtered, normalised ECG over [0, t_end + window end)
    beat_start: np.ndarray   # beat-window start of every kept beat
    beat_fold: np.ndarray    # CV fold (training beats) or -1 (test beats)
    labels: np.ndarray       # 1 = PVC
    target: np.ndarray       # readout target y(t) on [0, t_train)
    window: tuple[int, int]  # beat window offsets around R, samples
    plot_start: int
    plot_len: int


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _path(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def samples(seconds: float, fs: float) -> int:
    return int(round(seconds * fs))


def prepare_record(name: str, cfg: dict) -> RecordData:
    d, sp_, pp, rd = cfg["data"], cfg["split"], cfg["preprocess"], cfg["readout"]
    rec = load_record(name, _path(d["data_dir"]), [_path(m) for m in d.get("local_mirrors", [])], d["lead"])
    fs = rec.fs
    if fs != d["fs"]:
        raise ValueError(f"record {name}: fs={fs}, config expects {d['fs']}")
    # Crop so that the train window starts at `start_minutes` (per-record override
    # possible); everything below is relative to the cropped signal.
    start_min = sp_.get("start_minutes_per_record", {}).get(name, sp_.get("start_minutes", 0))
    off = samples(start_min * 60, fs)
    R, labels = rec.beat_samples - off, rec.labels
    signal = rec.signal[off:]
    if off:
        log.info("record %s: train window starts at %g min (test %g-%g min)", name, start_min,
                 start_min + sp_["train_minutes"], start_min + sp_["train_minutes"] + sp_["test_minutes"])
    t_train = samples(sp_["train_minutes"] * 60, fs)
    t_end = t_train + samples(sp_["test_minutes"] * 60, fs)
    w_lo, w_hi = (samples(ms / 1000, fs) for ms in rd["beat_window_ms"])
    t_ext = t_end + max(w_hi, 0)
    washout = samples(cfg["esn"]["washout_seconds"], fs)
    nf = sp_["n_cv_folds"]
    fold_len = t_train // nf

    sos = highpass_sos(pp["highpass_order"], pp["highpass_cutoff_hz"], fs)
    x = filter_without_leakage(signal, sos, t_train, t_ext, pp["zero_phase"])
    x_norm = minmax_normalize(x, t_train, pp["normalization_scope"])

    train = (R >= 0) & (R < t_train) & (R + w_lo >= washout) & (R + w_hi <= t_train)
    test = (R >= t_train) & (R < t_end)
    keep = train | test
    fold = np.where(train, np.minimum(R // fold_len, nf - 1), -1)

    # Only PVCs whose R peak lies in the training window define the training target.
    target = np.zeros(t_train, dtype=bool)
    for r in R[(labels == 1) & (R >= 0) & (R < t_train)]:
        target[max(r + w_lo, 0):min(r + w_hi, t_train)] = True

    # Figure segment: the test stretch with the most PVCs (labels are only used for display).
    plot_len = samples(cfg["output"]["plot_seconds"], fs)
    test_pvc = R[test & (labels == 1)]
    starts = np.arange(t_train, max(t_end - plot_len, t_train) + 1, int(fs))
    counts = [np.count_nonzero((test_pvc >= s) & (test_pvc < s + plot_len)) for s in starts]
    plot_start = int(starts[int(np.argmax(counts))])

    log.info("record %s: train beats PVC=%d other=%d | test beats PVC=%d other=%d | "
             "target PVC samples=%d", name,
             int(labels[train].sum()), int((labels[train] == 0).sum()),
             int(labels[test].sum()), int((labels[test] == 0).sum()), int(target.sum()))
    return RecordData(name=name, fs=fs, x_norm=x_norm, beat_start=R[keep] + w_lo,
                      beat_fold=fold[keep], labels=labels[keep], target=target,
                      window=(w_lo, w_hi), plot_start=plot_start, plot_len=plot_len)


def make_spec(cfg: dict, fs: float, device: str) -> JobSpec:
    e, rd = cfg["esn"], cfg["readout"]
    t_train = samples(cfg["split"]["train_minutes"] * 60, fs)
    w_lo, w_hi = (samples(ms / 1000, fs) for ms in rd["beat_window_ms"])
    lambdas = sorted(set(float(x) for x in rd["lambdas"]) | {float(rd["fixed_lambda"])})
    return JobSpec(
        seed=0,
        n_reservoir=int(e["n_reservoir"]),
        connectivity=float(e["connectivity"]),
        spectral_radius=float(e["spectral_radius"]),
        bias_scaling=float(e["bias_scaling"]),
        decays=tuple(float(np.exp(-1000.0 / (tau * fs))) for tau in cfg["smoothing"]["tau_ms"]),
        lambdas=tuple(lambdas),
        weightings=tuple(rd["class_weightings"]),
        t_train=t_train,
        washout=samples(e["washout_seconds"], fs),
        n_folds=int(cfg["split"]["n_cv_folds"]),
        window_len=w_hi - w_lo,
        chunk=int(cfg["compute"]["chunk_size"]),
        aggregations=(rd["aggregation"],),
        device=device,
    )


def build_members(records: list[RecordData], cfg: dict) -> list[Member]:
    plot_seed = cfg["output"]["plot_seed"]
    members = []
    for rd in records:
        for inp in cfg["encoding"]["inputs"]:
            u = build_input(rd.x_norm, inp, cfg["encoding"]["delta_threshold"])
            for leak in cfg["esn"]["leak_rates"]:
                for seed in cfg["seeds"]:
                    for s in cfg["esn"]["input_scalings"]:
                        members.append(Member(
                            key=(rd.name, inp, float(leak), int(seed), float(s)),
                            u=u, input_scaling=float(s), leak=float(leak), target=rd.target,
                            beat_start=rd.beat_start, beat_fold=rd.beat_fold,
                            trace_start=rd.plot_start if seed == plot_seed else -1,
                            trace_len=rd.plot_len if seed == plot_seed else 0,
                        ))
    return members


def group_jobs(members: list[Member], batch_size: int) -> list[tuple[int, list[Member]]]:
    """Batch members that share a seed (one W) and an input encoding (same K)."""
    groups: dict[tuple, list[Member]] = {}
    for m in members:
        groups.setdefault((m.key[3], m.key[1]), []).append(m)
    jobs = []
    for (seed, _), ms in groups.items():
        for i in range(0, len(ms), batch_size):
            jobs.append((seed, ms[i:i + batch_size]))
    return jobs


def auto_cpu_workers(spec: JobSpec, batch_size: int) -> int:
    n1 = spec.n_reservoir + 1
    per_job = batch_size * (2 * spec.n_folds * len(spec.decays) * n1 * n1 * 8 + 150e6)
    avail = 4e9
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    return max(1, min(os.cpu_count() or 1, int(0.75 * avail // per_job)))


_thread_limit = None


def _init_cpu_worker() -> None:
    """One BLAS thread per process: the pool already uses every core."""
    global _thread_limit
    from threadpoolctl import threadpool_limits
    _thread_limit = threadpool_limits(1)


def member_cache_key(spec: JobSpec, seed: int, m: Member) -> str:
    """Hash of everything that determines a member's result (spec, seed, inputs, labels)."""
    h = hashlib.sha1()
    h.update(repr(dataclasses.replace(spec, seed=seed, device="")).encode())
    h.update(repr((m.key, m.input_scaling, m.leak, m.trace_start, m.trace_len)).encode())
    for a in (m.u, m.target, m.beat_start, m.beat_fold):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _run_cached(spec: JobSpec, members: list[Member], paths: list[Path]) -> list[dict]:
    """Run one job in a worker and write each member's result to its cache file."""
    results = run_job(spec, members)
    for res, path in zip(results, paths):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    return results


def run_all(jobs, spec: JobSpec, device: str, cfg: dict, workers: int | None,
            cache_dir: Path | None = None) -> dict:
    """Run every job; with `cache_dir`, finished members are reused and new ones saved,
    so an interrupted run resumes where it stopped."""
    comp = cfg["compute"]
    results, todo = {}, []
    for seed, ms in jobs:
        paths = [cache_dir / f"{member_cache_key(spec, seed, m)}.pkl" for m in ms] if cache_dir else []
        if paths and all(p.exists() for p in paths):
            for m, p in zip(ms, paths):
                with open(p, "rb") as fh:
                    results[m.key] = pickle.load(fh)
        else:
            todo.append((seed, ms, paths))
    if cache_dir:
        log.info("%d/%d jobs already cached in %s", len(jobs) - len(todo), len(jobs), cache_dir)
    if not todo:
        return results
    if device == "cpu":
        n = workers or (comp["cpu_workers"] if comp["cpu_workers"] != "auto"
                        else auto_cpu_workers(spec, comp["cpu_batch_size"]))
        pool = ProcessPoolExecutor(max_workers=int(n), initializer=_init_cpu_worker)
        log.info("running %d jobs on CPU with %d worker processes", len(todo), n)
    else:
        n = workers or comp["gpu_workers"]
        pool = ThreadPoolExecutor(max_workers=int(n))
        log.info("running %d batched jobs on CUDA with %d host threads", len(todo), n)
    t0 = time.time()
    with pool:
        futures = {}
        for seed, ms, paths in todo:
            job_spec = dataclasses.replace(spec, seed=seed)
            fut = (pool.submit(_run_cached, job_spec, ms, paths) if paths
                   else pool.submit(run_job, job_spec, ms))
            futures[fut] = ms
        for done, fut in enumerate(as_completed(futures), 1):
            for m, res in zip(futures[fut], fut.result()):
                results[m.key] = res
            elapsed = time.time() - t0
            log.info("job %d/%d done | %.0fs elapsed, ~%.0fs left", done, len(todo), elapsed,
                     elapsed / done * (len(todo) - done))
    return results


def evaluate_all(records, results, cfg, spec, results_dir: Path) -> pd.DataFrame:
    rd_cfg = cfg["readout"]
    thresholds = threshold_grid(**rd_cfg["thresholds"])
    taus, lambdas, scalings = cfg["smoothing"]["tau_ms"], spec.lambdas, cfg["esn"]["input_scalings"]
    fixed_idx = lambdas.index(float(rd_cfg["fixed_lambda"]))
    agg, plot_seed = rd_cfg["aggregation"], cfg["output"]["plot_seed"]
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rd in records:
        is_train = rd.beat_fold >= 0
        y_val, y_test = rd.labels[is_train], rd.labels[~is_train]
        r_test = rd.beat_start[~is_train] - rd.window[0]
        for inp in cfg["encoding"]["inputs"]:
            for leak in cfg["esn"]["leak_rates"]:
                for seed in cfg["seeds"]:
                    res = [results[(rd.name, inp, float(leak), int(seed), float(s))] for s in scalings]
                    val = np.stack([r["val"][agg] for r in res])     # (s, tau, w, lam, beats)
                    test = np.stack([r["test"][agg] for r in res])
                    for wi, weighting in enumerate(spec.weightings):
                        for mode in ("cv", "fixed"):
                            sel = select_operating_point(val[:, :, wi], y_val, thresholds,
                                                         None if mode == "cv" else fixed_idx)
                            s, t, l = sel["scaling_index"], sel["tau_index"], sel["lambda_index"]
                            pred = test[s, t, wi, l] > sel["threshold"]
                            rows.append({
                                "record": rd.name, "input": inp, "leak": float(leak),
                                "lambda_mode": mode, "weighting": weighting, "seed": int(seed),
                                "input_scaling": scalings[s], "tau_ms": taus[t], "lambda": lambdas[l],
                                "threshold": sel["threshold"], "cv_f1": sel["cv_f1"],
                                **binary_metrics(y_test, pred),
                            })
                            if seed == plot_seed and mode == "cv" and weighting == "none":
                                seg = slice(rd.plot_start, rd.plot_start + rd.plot_len)
                                in_seg = (r_test >= seg.start) & (r_test < seg.stop)
                                plot_record(
                                    fig_dir / f"record_{rd.name}_{inp}_leak{leak}.png", rd.fs,
                                    rd.plot_start, rd.x_norm[seg], res[s]["trace"][t, wi, l],
                                    sel["threshold"], r_test[in_seg & (y_test == 1)],
                                    r_test[in_seg & pred], rd.window,
                                    f"Record {rd.name} | input={inp}, a={leak}, seed={seed} | "
                                    f"s={scalings[s]}, tau={taus[t]} ms, lambda={lambdas[l]:g}, "
                                    f"θ={sel['threshold']:.2f} (all CV-selected)")
    return pd.DataFrame(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--records", nargs="+", help="override record list, e.g. --records 119")
    p.add_argument("--input", choices=["spikes", "raw", "both"], help="ESN input (default: config)")
    p.add_argument("--leak", nargs="+", type=float, help="leak rate(s) a")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--start-minutes", nargs="+", metavar="[REC=]MIN",
                   help="train-window start in minutes, globally (e.g. 5) or per record (e.g. 200=5 119=10)")
    p.add_argument("--n-reservoir", type=int)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "gpu"])
    p.add_argument("--workers", type=int, help="CPU processes or GPU host threads")
    p.add_argument("--batch-size", type=int, help="reservoirs per job (CPU or GPU)")
    p.add_argument("--results-dir")
    p.add_argument("--no-cache", action="store_true", help="ignore and do not write results/cache")
    return p.parse_args(argv)


def apply_overrides(cfg: dict, args) -> dict:
    if args.records:
        cfg["data"]["records"] = [str(r) for r in args.records]
    if args.input:
        cfg["encoding"]["inputs"] = ["spikes", "raw"] if args.input == "both" else [args.input]
    if args.leak:
        cfg["esn"]["leak_rates"] = args.leak
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.start_minutes:
        per_rec = cfg["split"].setdefault("start_minutes_per_record", {})
        for item in args.start_minutes:
            rec, _, minutes = item.rpartition("=")
            if rec:
                per_rec[rec] = float(minutes)
            else:
                cfg["split"]["start_minutes"] = float(minutes)
    if args.n_reservoir:
        cfg["esn"]["n_reservoir"] = args.n_reservoir
    if args.device:
        cfg["compute"]["device"] = args.device
    if args.batch_size:
        cfg["compute"]["cpu_batch_size"] = cfg["compute"]["gpu_batch_size"] = args.batch_size
    if args.results_dir:
        cfg["output"]["results_dir"] = args.results_dir
    if cfg["output"]["plot_seed"] not in cfg["seeds"]:
        cfg["output"]["plot_seed"] = cfg["seeds"][0]
    return cfg


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = apply_overrides(load_config(Path(args.config)), args)
    results_dir = _path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(results_dir / "run.log", "w")])
    t_start = time.time()
    device = resolve_device(cfg["compute"]["device"])
    log.info("device: %s (requested %s)", device, cfg["compute"]["device"])

    names = [str(r) for r in cfg["data"]["records"]]
    with ThreadPoolExecutor(max_workers=min(8, len(names))) as pool:
        records = list(pool.map(lambda n: prepare_record(n, cfg), names))

    spec = make_spec(cfg, records[0].fs, device)
    members = build_members(records, cfg)
    batch = cfg["compute"]["cpu_batch_size" if device == "cpu" else "gpu_batch_size"]
    jobs = group_jobs(members, int(batch))
    log.info("%d reservoir runs (records x inputs x leaks x seeds x scalings) in %d jobs",
             len(members), len(jobs))
    cache_dir = None
    if cfg["compute"].get("cache", True) and not args.no_cache:
        cache_dir = results_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
    results = run_all(jobs, spec, device, cfg, args.workers, cache_dir)

    per_seed = evaluate_all(records, results, cfg, spec, results_dir)
    per_seed.to_csv(results_dir / "per_seed.csv", index=False, float_format="%.6f")
    summary = summarize(per_seed)
    write_tables(summary, results_dir, len(cfg["seeds"]))
    with open(results_dir / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump({**cfg, "resolved_device": device}, fh, sort_keys=False)
    log.info("done in %.1f min; tables in %s", (time.time() - t_start) / 60, results_dir)


if __name__ == "__main__":
    main()
