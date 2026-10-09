#!/usr/bin/env python3
"""Decide whether the E+L feature-fusion collapse is representational interference
or a fixed-hyper-parameter conditioning artefact.

Controls, all on the identical frozen banks and identical ordered samples:
  E, M, L                 native d=2048, protocol probe (Ridge a=1 / LogReg C=1)
  E+L, M+L                protocol balanced_concat -> d=4096, protocol probe
  E|E, L|L                balanced_concat of a bank WITH ITSELF -> d=4096.
                          Carries exactly the information of one checkpoint, so any
                          drop versus E / L is purely a dimensionality/regularisation
                          effect and cannot be interference.
  *_cv                    every arm re-fit with a train-only cross-validated
                          regularisation strength over one shared grid.

If E|E collapses like E+L, the reported fusion losses are an artefact of holding the
regularisation fixed while the feature dimension doubles past the sample count.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/mnt/huawei_deepcad/dinov3")
sys.path.insert(0, str(REPO))

FROZEN = REPO / "outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921/frozen"
ANCHOR = {"E": 12687, "M": 20007, "L": 29279}


def unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / n


def balanced_concat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return unit_rows(np.concatenate((unit_rows(a), unit_rows(b)), axis=1))


def load(ds: str, role: str, split: str | None):
    import glob
    pat = f"{FROZEN}/{ds}/{role}/features/{ds}/*"
    pat += f"_{split}.npz" if split else ".npz"
    fs = sorted(glob.glob(pat))
    if len(fs) != 1:
        raise FileNotFoundError(f"{ds}/{role}/{split}: {fs}")
    d = np.load(fs[0], allow_pickle=True)
    return np.asarray(d["features"]), np.asarray(d["labels"]), np.asarray(d["paths"])


def reg_eval(xtr, ytr, xte, yte, alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    m.fit(xtr, ytr.astype(float))
    return float(r2_score(yte.astype(float), m.predict(xte)))


def reg_eval_cv(xtr, ytr, xte, yte, grid, folds=5):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=grid, cv=folds))
    m.fit(xtr, ytr.astype(float))
    a = float(m[-1].alpha_)
    return float(r2_score(yte.astype(float), m.predict(xte))), a


def cls_eval(xtr, ytr, xte, yte, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=10000, class_weight="balanced", C=C,
                                         random_state=0, n_jobs=1))
    m.fit(xtr, ytr.astype(int))
    return float(balanced_accuracy_score(yte.astype(int), m.predict(xte)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="conic-cell-count,livecell-cell-count")
    ap.add_argument("--task", choices=["regression", "classification"], default="regression")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/00_reports/hs6_l5_fusion_conditioning_control_20260923")
    ap.add_argument("--subsample-train", type=int, default=0,
                    help="cap train size (classification only, for cost); 0 = use all")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = np.logspace(-1, 5, 13)
    report = {}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        banks = {}
        for role in ("E", "M", "L"):
            xtr, ytr, ptr = load(ds, role, "train")
            xte, yte, pte = load(ds, role, "test")
            banks[role] = (xtr, ytr, ptr, xte, yte, pte)
        # identity check across roles
        for role in ("M", "L"):
            assert np.array_equal(banks["E"][2], banks[role][2]), f"{ds}: train paths differ"
            assert np.array_equal(banks["E"][5], banks[role][5]), f"{ds}: test paths differ"
            assert np.array_equal(banks["E"][1], banks[role][1]), f"{ds}: train labels differ"
        ytr, yte = banks["E"][1], banks["E"][4]
        n_tr, d = banks["E"][0].shape
        arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for role in ("E", "M", "L"):
            arms[role] = (banks[role][0], banks[role][3])
        arms["E+L"] = (balanced_concat(banks["E"][0], banks["L"][0]),
                       balanced_concat(banks["E"][3], banks["L"][3]))
        arms["M+L"] = (balanced_concat(banks["M"][0], banks["L"][0]),
                       balanced_concat(banks["M"][3], banks["L"][3]))
        arms["E|E"] = (balanced_concat(banks["E"][0], banks["E"][0]),
                       balanced_concat(banks["E"][3], banks["E"][3]))
        arms["L|L"] = (balanced_concat(banks["L"][0], banks["L"][0]),
                       balanced_concat(banks["L"][3], banks["L"][3]))

        rows = {}
        print(f"\n=== {ds}  n_train={n_tr} d={d} (concat d={2*d}) task={args.task} ===", flush=True)
        for name, (xtr, xte) in arms.items():
            if args.task == "regression":
                fixed = reg_eval(xtr, ytr, xte, yte, alpha=1.0)
                cv, a = reg_eval_cv(xtr, ytr, xte, yte, grid)
                rows[name] = {"protocol_fixed": fixed, "train_cv": cv, "cv_alpha": a,
                              "dim": int(xtr.shape[1])}
                print(f"  {name:5s} d={xtr.shape[1]:5d}  R2 fixed(a=1)={fixed:.4f}   "
                      f"R2 train-CV={cv:.4f} (a*={a:g})", flush=True)
            else:
                fixed = cls_eval(xtr, ytr, xte, yte, C=1.0)
                rows[name] = {"protocol_fixed": fixed, "dim": int(xtr.shape[1])}
                print(f"  {name:5s} d={xtr.shape[1]:5d}  balAcc fixed(C=1)={fixed:.4f}", flush=True)
        report[ds] = {"n_train": int(n_tr), "n_test": int(len(yte)), "native_dim": int(d),
                      "task": args.task, "arms": rows}
    out = args.out / f"control_{args.task}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
