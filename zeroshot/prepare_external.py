"""
Step 1/2 for external data — label mapping, cohort selection, lead II extraction,
500 -> 360 Hz resampling (scipy resample_poly 18/25).  No filtering here: each model
applies its own pipeline (LGBM 0.5–40 Hz band-pass; Farag none) in featurize.py.

Usage:  python -I prepare_external.py ptbxl|chapman
Output: out/<ds>_manifest.csv (one row per cohort record, truth columns t_N,t_L,t_R,t_V,t_S)
        out/<ds>_leadII_360.npz (X (n,3600) float32, ids)

Label maps (record level):
  PTB-XL (scp_codes keys):
    N = keys ⊆ {NORM, SR} with NORM likelihood 100      L = CLBBB     R = CRBBB
    V = PVC                                              S = PAC
  Chapman-Shaoxing (SNOMED-CT #Dx):
    N = Dx == {426783006 SR}   L = 164909002 LBBB | 733534002 CLBBB   R = 59118001 RBBB | 713427006 CRBBB
    V = 17338001 VPB | 11157007 ventricular bigeminy | 427172004 PVC
    S = 284470004 APB | 251173003 atrial bigeminy | 251164006 junctional premature beat
                       | 63593006 SVPB
Cohort = N records ∪ records with ≥1 target label, EXCLUDING records that also carry a
non-sinus tachy/fibrillation rhythm (AF, AFL, SVT, AT, VT ...) or pacing, which lie
outside the MIT-BIH 5-class label space (decided a priori, not tuned on the targets).
"""
import ast
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXT, OUT, resample_to_360, STRIP_LEN  # noqa

PTB_DIR = next(p.parent for p in (EXT / "ptbxl").glob("*/ptbxl_database.csv")) if (EXT / "ptbxl").exists() else None
CH_DIR = next((EXT / "chapman").glob("*/WFDBRecords")).parent if (EXT / "chapman").exists() else None

PTB_EXCL = {"AFIB", "AFLT", "SVTAC", "PSVT", "PACE", "STACH_X"}
CH_EXCL = {"164889003", "164890007", "426761007", "713422000", "233896004", "233897008",
           "164895002", "195101003", "10370003"}   # ... + 10370003 pacing rhythm
CH_L = {"164909002", "733534002"}   # LBBB, complete LBBB (code used in headers, absent from CSV)
CH_R = {"59118001", "713427006"}    # RBBB, complete RBBB (code used in headers, absent from CSV)
CH_V = {"17338001", "11157007", "427172004"}
CH_S = {"284470004", "251173003", "251164006", "63593006"}


def ptbxl_manifest():
    db = pd.read_csv(PTB_DIR / "ptbxl_database.csv")
    rows = []
    for _, r in db.iterrows():
        codes = ast.literal_eval(r.scp_codes)
        keys = set(codes)
        t = dict(t_L="CLBBB" in keys, t_R="CRBBB" in keys, t_V="PVC" in keys, t_S="PAC" in keys,
                 t_N=keys <= {"NORM", "SR"} and codes.get("NORM", 0) == 100 and "SR" in keys)
        if not any(t.values()) or (keys & PTB_EXCL) or str(r.pacemaker) not in ("nan", ""):
            continue
        rows.append(dict(id=int(r.ecg_id), path=str(PTB_DIR / r.filename_hr), patient=r.patient_id,
                         codes=";".join(sorted(keys)), **t))
    return pd.DataFrame(rows)


def chapman_manifest():
    rows = []
    for hea in sorted(CH_DIR.glob("WFDBRecords/*/*/*.hea")):
        dx = set()
        for line in hea.read_text(errors="ignore").splitlines():
            if line.startswith("#Dx:"):
                dx = {c.strip() for c in line[4:].split(",") if c.strip()}
        t = dict(t_L=bool(dx & CH_L), t_R=bool(dx & CH_R), t_V=bool(dx & CH_V), t_S=bool(dx & CH_S),
                 t_N=dx == {"426783006"})
        if not any(t.values()) or (dx & CH_EXCL):
            continue
        rows.append(dict(id=hea.stem, path=str(hea.with_suffix("")), patient=hea.stem,
                         codes=";".join(sorted(dx)), **t))
    return pd.DataFrame(rows)


def load_lead2(path):
    try:
        rec = wfdb.rdrecord(path)
        names = [n.strip().upper() for n in rec.sig_name]
        x = rec.p_signal[:, names.index("II")].astype(np.float64)
        fs = rec.fs
        if not np.isfinite(x).all():
            x = pd.Series(x).interpolate(limit_direction="both").fillna(0).to_numpy()
        y = resample_to_360(x, fs)[:STRIP_LEN]
        if len(y) < STRIP_LEN:
            y = np.pad(y, (0, STRIP_LEN - len(y)), mode="edge")
        return y.astype(np.float32), "", fs
    except Exception as e:  # noqa
        return np.zeros(STRIP_LEN, np.float32), repr(e)[:120], 0


def all_codes_ptbxl():
    db = pd.read_csv(PTB_DIR / "ptbxl_database.csv")
    from collections import Counter
    c = Counter(k for s in db.scp_codes for k in ast.literal_eval(s))
    return c


if __name__ == "__main__":
    ds = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    man = ptbxl_manifest() if ds == "ptbxl" else chapman_manifest()
    print(ds, "cohort", len(man), {c: int(man[c].sum()) for c in ["t_N", "t_L", "t_R", "t_V", "t_S"]}, flush=True)
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(load_lead2, man.path.tolist(), chunksize=64))
    X = np.stack([r[0] for r in res])
    man["load_error"] = [r[1] for r in res]
    man["fs_in"] = [r[2] for r in res]
    man["flat"] = np.ptp(X, axis=1) < 0.05
    print("load errors", int((man.load_error != "").sum()), "flat", int(man.flat.sum()), flush=True)
    man.to_csv(OUT / f"{ds}_manifest.csv", index=False)
    np.savez_compressed(OUT / f"{ds}_leadII_360.npz", X=X, ids=man.id.astype(str).to_numpy())
    print("saved", flush=True)
