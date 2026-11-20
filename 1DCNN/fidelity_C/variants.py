"""Variant definitions for fid.py (config overrides on configs/default.yaml).

Keys starting with '_' are driver-only flags.
ORIG  = the replication as it stood (configs/default.yaml, class_weight none).
PAPER = paper-faithful reading of Farag 2023 Table 3 / Sec 3 / Sec 5.2:
        Table-3 architecture (64-sample input, Conv1D(NK, same) -> BN -> Tanh
        -> GMP; RR (4,1) -> Dense 32/16/8 per feature -> Flatten), 11 templates
        for 3 classes (F, Q excluded), templates from all of DS1, no class
        weights (Model 6).
"""
ORIG = {}
PAPER = {
    "model.arch": "table3",
    "model.padding": "same",
    "input.crop_input": False,
    "templates.match_classes": True,
    "templates.source": "ds1_all",
    "training.class_weight": "none",
}


def P(**kw):
    d = dict(PAPER)
    for k, v in kw.items():
        d[k.replace("__", ".")] = v
    return d


V = {}
# ---- ladder ORIG -> PAPER (cumulative)
V["L0_orig"] = dict(ORIG)
V["L1_in64_same"] = {"input.crop_input": False, "model.padding": "same"}
V["L2_table3"] = {**V["L1_in64_same"], "model.arch": "table3"}
V["L3_11tpl"] = {**V["L2_table3"], "templates.match_classes": True}
V["L4_paper"] = dict(PAPER)          # + templates from all DS1
# ---- leave-one-out / window ablations around PAPER
V["P_crop32"] = P(input__crop_input=True)               # input cropped to NK=32
V["P_valid"] = P(model__padding="valid")                # 64 input, valid -> 33 positions
V["P_13tpl"] = P(templates__match_classes=False)
V["P_tplTrain"] = P(templates__source="ds1_train")
V["P_aamiTpl"] = {**P(), "_aami_templates": True}       # 3 AAMI super-class templates
V["P_nk64"] = P(input__nk=64)
V["P_win48"] = P(preprocess__segment_samples=48, preprocess__before_r=24, preprocess__after_r=24)
V["P_win96"] = P(preprocess__segment_samples=96, preprocess__before_r=48, preprocess__after_r=48)
V["P_shift"] = P(preprocess__before_r=24, preprocess__after_r=40)   # asymmetric (more T wave)
# ---- resampling
V["P_rs_fft"] = P(preprocess__resample="fft")
V["P_rs_linear"] = P(preprocess__resample="linear")
V["P_rs_naive"] = P(preprocess__resample="decimate_naive")
# ---- normalisation
V["P_norm_z"] = P(preprocess__normalization="z-score")
V["P_norm_none"] = P(preprocess__normalization="none")
V["P_norm_minmax"] = P(preprocess__normalization="minmax")
# ---- RR features
V["P_rr_40_200"] = P(rr__local_window=40, rr__global_window=200)
V["P_rr_160_800"] = P(rr__local_window=160, rr__global_window=800)
V["P_rr_10_80"] = P(rr__local_window=10, rr__global_window=80)
V["P_rr_past"] = P(rr__mean_includes_current=False)
V["P_rr_preonly"] = P(rr__use_post_rr=False)
# ---- input type
V["P_in_signal"] = P(input__mode="signal")
V["P_in_both"] = P(input__mode="both")
# ---- C4 per-patient RR normalisation
V["C4_append_full"] = P(rr__record_norm="append", rr__record_mean_type="full")
V["C4_replace_full"] = P(rr__record_norm="replace_global", rr__record_mean_type="full")
V["C4_append_exp"] = P(rr__record_norm="append", rr__record_mean_type="expanding")
V["C4_replace_exp"] = P(rr__record_norm="replace_global", rr__record_mean_type="expanding")
V["C4_orig_append_full"] = {"rr.record_norm": "append", "rr.record_mean_type": "full"}
# ---- C2 training hyper-parameters
V["T_es_kerasdef"] = P(training__early_stop_patience=0, training__restore_best_weights=False,
                       training__lr_schedule="none")      # bare Keras EarlyStopping(), Adam 1e-3
V["T_es_p5"] = P(training__early_stop_patience=5)
V["T_es_p50"] = P(training__early_stop_patience=50)
V["T_lr_const"] = P(training__lr_schedule="none")
V["T_bs64"] = P(training__batch_size=64)
V["T_trainDS1all"] = P(data__train_records="ds1_all")     # ">50K examples" reading; val overlaps train
V["T_orig_trainDS1all"] = {"data.train_records": "ds1_all"}
# ---- paper Table 4 Models 1-9 on the PAPER base
_cw = "balanced"
V["M1"] = P(task__init="default", input__nk=64, training__class_weight=_cw)
V["M2"] = P(task__init="mf_trainable", training__class_weight=_cw)
V["M3"] = P(task__init="mf_frozen", training__class_weight=_cw)
V["M4"] = P(task__init="default", input__mode="both", training__class_weight=_cw)
V["M4_frozen"] = P(task__init="mf_frozen", input__mode="both", training__class_weight=_cw)
V["M5"] = P(task__init="mf_trainable", input__mode="signal", training__class_weight=_cw)
V["M6"] = P()
V["M7"] = P(input__nk=64)
V["M8"] = P(task__classes=4)
V["M9"] = P(task__classes=5)
V["M2_esloss"] = P(training__class_weight=_cw, training__early_stop_monitor="val_loss")
# Table 4 on the ORIG (compact) base, for comparison of orderings
for m in ("M1", "M2", "M3", "M5", "M6"):
    o = {k: v for k, v in V[m].items() if k not in PAPER or k in ("training.class_weight",)}
    V["O" + m] = o

# ---- extra: raw amplitude / template scaling / train-on-all-DS1 combos
V["P_tpl_noL2"] = P(templates__normalize_l2=False)
V["P_raw"] = P(preprocess__normalization="none", templates__normalize_l2=False)
V["P_trainAll_cw"] = P(data__train_records="ds1_all", training__class_weight="balanced")
V["P_trainAll_z"] = P(data__train_records="ds1_all", preprocess__normalization="z-score")
V["P_trainAll_raw"] = P(data__train_records="ds1_all", preprocess__normalization="none",
                        templates__normalize_l2=False)
V["M6_trainAll_p50"] = P(data__train_records="ds1_all", training__early_stop_patience=50)

# =====================================================================
# Round 2 (2026-11-19): base F = PAPER + train on ALL DS1 (paper Sec 6.1:
# "training set size of more than 50 K examples"; DS1-val (>=200) stays the
# early-stopping set, so it overlaps training exactly as the paper's text
# implies) + max 100 epochs (paper: "does not exceed 100 epochs").
# FZ = F + per-segment z-score (best normalisation in round 1).
F = {**PAPER, "data.train_records": "ds1_all", "training.max_epochs": 100}
FZ = {**F, "preprocess.normalization": "z-score"}


def FZv(**kw):
    d = dict(FZ)
    for k, v in kw.items():
        d[k.replace("__", ".")] = v
    return d


V["F_mean"] = dict(F)
V["FZ"] = dict(FZ)
V["FZ_minmax"] = FZv(preprocess__normalization="minmax")
V["FZ_ep500"] = FZv(training__max_epochs=500)
# ladder on full DS1 + z-score + 100 epochs
V["FL0_orig"] = {"data.train_records": "ds1_all", "training.max_epochs": 100,
                 "preprocess.normalization": "z-score"}
V["FL1_in64_same"] = {**V["FL0_orig"], "input.crop_input": False, "model.padding": "same"}
V["FL2_table3"] = {**V["FL1_in64_same"], "model.arch": "table3"}
V["FL3_11tpl"] = {**V["FL2_table3"], "templates.match_classes": True}   # FL3 + tpl ds1_all = FZ
# window / templates
V["FZ_crop32"] = FZv(input__crop_input=True)
V["FZ_valid"] = FZv(model__padding="valid")
V["FZ_13tpl"] = FZv(templates__match_classes=False)
V["FZ_tplTrain"] = FZv(templates__source="ds1_train")
V["FZ_aamiTpl"] = {**FZ, "_aami_templates": True}
V["FZ_win48"] = FZv(preprocess__segment_samples=48, preprocess__before_r=24, preprocess__after_r=24)
V["FZ_win96"] = FZv(preprocess__segment_samples=96, preprocess__before_r=48, preprocess__after_r=48)
V["FZ_shift"] = FZv(preprocess__before_r=24, preprocess__after_r=40)
# resampling
V["FZ_rs_fft"] = FZv(preprocess__resample="fft")
V["FZ_rs_naive"] = FZv(preprocess__resample="decimate_naive")
# RR
V["FZ_rr_40_200"] = FZv(rr__local_window=40, rr__global_window=200)
V["FZ_rr_160_800"] = FZv(rr__local_window=160, rr__global_window=800)
V["FZ_rr_past"] = FZv(rr__mean_includes_current=False)
V["FZ_rr_preonly"] = FZv(rr__use_post_rr=False)
# C4
V["FZ_C4_append_full"] = FZv(rr__record_norm="append", rr__record_mean_type="full")
V["FZ_C4_replace_full"] = FZv(rr__record_norm="replace_global", rr__record_mean_type="full")
V["FZ_C4_append_exp"] = FZv(rr__record_norm="append", rr__record_mean_type="expanding")
V["FZ_C4_replace_exp"] = FZv(rr__record_norm="replace_global", rr__record_mean_type="expanding")
# training
V["FZ_es_kerasdef"] = FZv(training__early_stop_patience=0, training__restore_best_weights=False,
                          training__lr_schedule="none")
V["FZ_lr_const"] = FZv(training__lr_schedule="none")
V["FZ_cw"] = FZv(training__class_weight="balanced")
# Table 4 on FZ base (all DS1, z-score, 100 epochs)
V["FM1"] = FZv(task__init="default", input__nk=64, training__class_weight="balanced")
V["FM2"] = FZv(task__init="mf_trainable", training__class_weight="balanced")
V["FM3"] = FZv(task__init="mf_frozen", training__class_weight="balanced")
V["FM4"] = FZv(task__init="default", input__mode="both", training__class_weight="balanced")
V["FM5"] = FZv(task__init="mf_trainable", input__mode="signal", training__class_weight="balanced")
V["FM6"] = dict(FZ)
V["FM7"] = FZv(input__nk=64)
V["FM8"] = FZv(task__classes=4)
V["FM9"] = FZv(task__classes=5)
V["FM6_signal"] = FZv(input__mode="signal")
V["FM6_both"] = FZv(input__mode="both")
V["FM6_frozen"] = FZv(task__init="mf_frozen")
V["FM6_default"] = FZv(task__init="default")
V["FZ_tplS_aami"] = FZv(templates__empty_fallback="aami")   # 'S' template = mean of A/a/J

# ---- farag-improve recipe on the paper architecture: GroupKFold-by-record epoch
# selection (their 5 folds, 1DCNN/improve_B/cache/folds.json), class weight
# power 0.75, then refit on all DS1 for the median best epoch (constant LR).
import json as _json
from pathlib import Path as _Path
_FOLDS = _json.loads((_Path(__file__).resolve().parents[1] / "improve_B" / "cache" /
                      "folds.json").read_text())["folds"]
_DS1 = sorted({r for f in _FOLDS for r in f})
for _k, _f in enumerate(_FOLDS):
    V[f"CV_FZ075_f{_k}"] = FZv(training__class_weight="balanced", training__class_weight_power=0.75,
                               training__max_epochs=500, training__early_stop_monitor="val_macro_f1")
    V[f"CV_FZ075_f{_k}"]["_train_recs"] = [r for r in _DS1 if r not in _f]
    V[f"CV_FZ075_f{_k}"]["_val_recs"] = list(_f)


def refit(base, epochs):
    d = dict(base)
    d.update({"data.train_records": "ds1_all", "training.max_epochs": int(epochs),
              "training.early_stop_patience": 100000, "training.restore_best_weights": False,
              "training.lr_schedule": "none"})
    for k in ("_train_recs", "_val_recs"):
        d.pop(k, None)
    return d

VARIANTS = V
GROUPS = {
    "cv_fz075": [f"CV_FZ075_f{k}" for k in range(5)],
    "r2_core": ["FZ", "F_mean", "FL0_orig", "FL1_in64_same", "FL2_table3", "FL3_11tpl"],
    "r2_table4": ["FM1", "FM2", "FM3", "FM4", "FM5", "FM7", "FM8", "FM9",
                  "FM6_signal", "FM6_both", "FM6_frozen", "FM6_default"],
    "r2_abl": ["FZ_crop32", "FZ_valid", "FZ_13tpl", "FZ_tplTrain", "FZ_aamiTpl", "FZ_win48",
               "FZ_win96", "FZ_shift", "FZ_rs_fft", "FZ_rs_naive", "FZ_rr_40_200",
               "FZ_rr_160_800", "FZ_rr_past", "FZ_rr_preonly", "FZ_minmax", "FZ_tplS_aami"],
    "r2_c4": ["FZ_C4_append_full", "FZ_C4_replace_full", "FZ_C4_append_exp", "FZ_C4_replace_exp"],
    "r2_train": ["FZ_es_kerasdef", "FZ_lr_const", "FZ_cw", "FZ_ep500"],
    "ladder": ["L0_orig", "L1_in64_same", "L2_table3", "L3_11tpl", "L4_paper"],
    "window": ["P_crop32", "P_valid", "P_13tpl", "P_tplTrain", "P_aamiTpl", "P_nk64",
               "P_win48", "P_win96", "P_shift"],
    "resample": ["P_rs_fft", "P_rs_linear", "P_rs_naive"],
    "norm": ["P_norm_z", "P_norm_none", "P_norm_minmax"],
    "rr": ["P_rr_40_200", "P_rr_160_800", "P_rr_10_80", "P_rr_past", "P_rr_preonly"],
    "input": ["P_in_signal", "P_in_both"],
    "c4": ["C4_append_full", "C4_replace_full", "C4_append_exp", "C4_replace_exp",
           "C4_orig_append_full"],
    "train": ["T_es_kerasdef", "T_es_p5", "T_es_p50", "T_lr_const", "T_bs64",
              "T_trainDS1all", "T_orig_trainDS1all"],
    "table4": ["M1", "M2", "M3", "M4", "M4_frozen", "M5", "M6", "M7", "M8", "M9", "M2_esloss"],
    "extra": ["P_tpl_noL2", "P_raw", "P_trainAll_cw", "P_trainAll_z", "P_trainAll_raw",
              "M6_trainAll_p50"],
    "table4_orig": ["OM1", "OM2", "OM3", "OM5", "OM6"],
}
GROUPS.setdefault("refit_fz075", ["R_FZ075"])
# refit variant (epochs = median GroupKFold best epoch, cv_epochs.json written by run_r3.sh)
_cvf = _Path(__file__).resolve().parent / "cv_epochs.json"
if _cvf.exists():
    V["R_FZ075"] = refit(V["CV_FZ075_f0"], _json.loads(_cvf.read_text())["FZ075"])
