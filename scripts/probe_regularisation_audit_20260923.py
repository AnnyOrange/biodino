#!/usr/bin/env python3
"""Is the measured early-to-late gap a loss of information, or a probe-regularisation
mismatch between representations of different effective rank?

The protocol fixes LogisticRegression C = 1.0 and Ridge alpha = 1.0 for every arm. The
late checkpoint has a higher effective rank than the early one on several datasets, so
at a fixed penalty it is effectively less regularised and can overfit the training
split harder - which costs the most on the datasets whose test split is a different
patient/plate population.

Here every arm additionally gets a train-only cross-validated penalty over one shared
grid. If the gap collapses under matched tuning, the capability was never lost.
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
    d = np.load(fs[0], allow_pickle=True)
    return np.asarray(d["features"], np.float32), np.asarray(d["labels"]), np.asarray(d["paths"])


def fit_score(task, xtr, ytr, xte, yte, hp):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, r2_score
    if task == "regression":
        m = make_pipeline(StandardScaler(), Ridge(alpha=hp)); m.fit(xtr, ytr.astype(float))
        return float(r2_score(yte.astype(float), m.predict(xte)))
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=6000,
                      class_weight="balanced", C=hp, random_state=0))
    m.fit(xtr, ytr.astype(int))
    return float(balanced_accuracy_score(yte.astype(int), m.predict(xte)))


def cv_pick(task, xtr, ytr, grid, folds=3, seed=0):
    """Train-only K-fold selection of the penalty. Never sees the test split."""
    from sklearn.model_selection import StratifiedKFold, KFold
    y = ytr.astype(int) if task != "regression" else ytr.astype(float)
    kf = (StratifiedKFold(folds, shuffle=True, random_state=seed) if task != "regression"
          else KFold(folds, shuffle=True, random_state=seed))
    best, best_s = grid[0], -1e18
    for hp in grid:
        s = []
        for tr, va in kf.split(xtr, y):
            s.append(fit_score(task, xtr[tr], ytr[tr], xtr[va], ytr[va], hp))
        m = float(np.mean(s))
        if m > best_s:
            best, best_s = hp, m
    return best, best_s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--roles", default="E,M,L")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/00_reports/hs6_l5_probe_regularisation_20260923")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        t0 = time.time()
        task = "regression" if ds in REG else "classification"
        grid = list(np.logspace(-1, 5, 7)) if task == "regression" else list(np.logspace(-5, 1, 7))
        rows = {}
        print(f"\n=== {ds} [{task}] ===", flush=True)
        for role in a.roles.split(","):
            xtr, ytr, _ = load(ds, role, "train"); xte, yte, _ = load(ds, role, "test")
            n = len(ytr); idx = np.arange(n)
            if n > a.max_train:
                rng = np.random.default_rng(0)
                idx = rng.choice(n, a.max_train, replace=False); idx.sort()
            fixed = fit_score(task, xtr[idx], ytr[idx], xte, yte, 1.0)
            hp, cvs = cv_pick(task, xtr[idx], ytr[idx], grid)
            tuned = fit_score(task, xtr[idx], ytr[idx], xte, yte, hp)
            # effective rank of the probed train features
            c = xtr[idx].astype(np.float64); c = c - c.mean(0)
            ev = np.linalg.eigvalsh(c.T @ c / max(1, len(c) - 1))[::-1]
            p = np.clip(ev, 0, None); p = p / p.sum(); nz = p[p > 0]
            rankme = float(np.exp(-(nz * np.log(nz)).sum()))
            rows[role] = {"protocol_fixed": fixed, "tuned": tuned, "hp": float(hp),
                          "cv_score": cvs, "rankme": rankme, "n_probe": int(len(idx))}
            print(f"  {role}: protocol(hp=1)={fixed:.4f}   train-CV tuned={tuned:.4f} "
                  f"(hp*={hp:g})   rankme={rankme:.1f}", flush=True)
        if "E" in rows and "L" in rows:
            gf = rows["E"]["protocol_fixed"] - rows["L"]["protocol_fixed"]
            gt = rows["E"]["tuned"] - rows["L"]["tuned"]
            print(f"  E-L gap: protocol {gf:+.4f}  ->  tuned {gt:+.4f}   "
                  f"({100*(1-gt/gf) if abs(gf)>1e-9 else float('nan'):.0f}% of the gap "
                  f"was regularisation)", flush=True)
            rows["gap_protocol"] = gf; rows["gap_tuned"] = gt
        rep[ds] = {"task": task, "roles": rows, "seconds": time.time() - t0}
        (a.out / "probe_regularisation.json").write_text(json.dumps(rep, indent=1))
    print("\nwrote", a.out / "probe_regularisation.json")


if __name__ == "__main__":
    main()
