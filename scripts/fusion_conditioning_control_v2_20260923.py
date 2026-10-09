#!/usr/bin/env python3
"""Separate the two confounds hidden in the protocol's feature-fusion arm.

The protocol scores a single checkpoint on RAW features, but builds E+L as
   L2-normalise each row of E, L2-normalise each row of L, concatenate, L2 again.
So the fusion arm differs from the single arms in TWO ways at once:
   (a) per-sample row normalisation, which a per-feature StandardScaler does not undo;
   (b) doubled feature dimension against a fixed Ridge alpha / LogReg C.
Arms below vary one at a time:
   raw        E, L                      protocol single-arm
   unit       unit(E), unit(L)          confound (a) only, same dimension
   selfcat    E|E, L|L                  confound (a)+(b) with no new information
   fusion     E+L, M+L                  protocol fusion
and every arm is additionally refit with a train-only cross-validated penalty so the
comparison no longer depends on a hyper-parameter chosen for d = 2048.
"""
from __future__ import annotations

import argparse, glob, json, time
from pathlib import Path
import numpy as np

REPO = Path("/mnt/huawei_deepcad/dinov3")
FROZEN = REPO / "outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921/frozen"
REG = {"conic-cell-count", "livecell-cell-count"}


def load(ds, role, split):
    fs = sorted(glob.glob(f"{FROZEN}/{ds}/{role}/features/{ds}/*_{split}.npz"))
    if len(fs) != 1:
        raise FileNotFoundError(f"{ds}/{role}/{split}: {fs}")
    d = np.load(fs[0], allow_pickle=True)
    return np.asarray(d["features"], np.float32), np.asarray(d["labels"]), np.asarray(d["paths"])


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def bcat(a, b):
    return unit(np.concatenate((unit(a), unit(b)), axis=1))


def run(task, xtr, ytr, xte, yte, cv, grid):
    from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression, LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, balanced_accuracy_score
    if task == "regression":
        est = RidgeCV(alphas=grid, cv=5) if cv else Ridge(alpha=1.0)
        m = make_pipeline(StandardScaler(), est); m.fit(xtr, ytr.astype(float))
        return float(r2_score(yte.astype(float), m.predict(xte))), (float(m[-1].alpha_) if cv else 1.0)
    est = (LogisticRegressionCV(Cs=grid, cv=3, max_iter=4000, class_weight="balanced",
                                scoring="balanced_accuracy", random_state=0, n_jobs=8)
           if cv else LogisticRegression(max_iter=10000, class_weight="balanced", C=1.0, random_state=0))
    m = make_pipeline(StandardScaler(), est); m.fit(xtr, ytr.astype(int))
    c = float(np.mean(list(m[-1].C_))) if cv else 1.0
    return float(balanced_accuracy_score(yte.astype(int), m.predict(xte))), c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/00_reports/hs6_l5_fusion_conditioning_control_20260923")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        t0 = time.time()
        task = "regression" if ds in REG else "classification"
        grid = np.logspace(-1, 5, 13) if task == "regression" else np.logspace(-4, 2, 7)
        b = {r: dict(zip(("xtr", "ytr", "ptr"), load(ds, r, "train"))) |
                dict(zip(("xte", "yte", "pte"), load(ds, r, "test"))) for r in ("E", "M", "L")}
        for r in ("M", "L"):
            assert np.array_equal(b["E"]["ptr"], b[r]["ptr"]) and np.array_equal(b["E"]["pte"], b[r]["pte"])
        ytr, yte = b["E"]["ytr"], b["E"]["yte"]
        n, d = b["E"]["xtr"].shape
        arms = {
            "E_raw": (b["E"]["xtr"], b["E"]["xte"]),
            "L_raw": (b["L"]["xtr"], b["L"]["xte"]),
            "M_raw": (b["M"]["xtr"], b["M"]["xte"]),
            "E_unit": (unit(b["E"]["xtr"]), unit(b["E"]["xte"])),
            "L_unit": (unit(b["L"]["xtr"]), unit(b["L"]["xte"])),
            "E|E": (bcat(b["E"]["xtr"], b["E"]["xtr"]), bcat(b["E"]["xte"], b["E"]["xte"])),
            "L|L": (bcat(b["L"]["xtr"], b["L"]["xtr"]), bcat(b["L"]["xte"], b["L"]["xte"])),
            "E+L": (bcat(b["E"]["xtr"], b["L"]["xtr"]), bcat(b["E"]["xte"], b["L"]["xte"])),
            "M+L": (bcat(b["M"]["xtr"], b["L"]["xtr"]), bcat(b["M"]["xte"], b["L"]["xte"])),
        }
        print(f"\n=== {ds} [{task}] n_train={n} n_test={len(yte)} d={d}  "
              f"cv={'on' if a.cv else 'off'} ===", flush=True)
        rows = {}
        for name, (xtr, xte) in arms.items():
            s, hp = run(task, xtr, ytr, xte, yte, a.cv, grid)
            rows[name] = {"score": s, "dim": int(xtr.shape[1]), "hyper": hp}
            print(f"  {name:7s} d={xtr.shape[1]:5d}  score={s:.4f}  hp={hp:g}", flush=True)
        rep[ds] = {"task": task, "n_train": int(n), "n_test": int(len(yte)), "cv": bool(a.cv),
                   "arms": rows, "seconds": time.time() - t0}
        (a.out / f"control_{a.tag}{'_cv' if a.cv else ''}.json").write_text(json.dumps(rep, indent=1))
    print("\nwrote", a.out / f"control_{a.tag}{'_cv' if a.cv else ''}.json")


if __name__ == "__main__":
    main()
