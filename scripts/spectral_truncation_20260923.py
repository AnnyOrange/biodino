#!/usr/bin/env python3
"""How many of the anchor's own eigen-directions does the linear probe actually need?

Probe each checkpoint on its features truncated to their own top-k principal
directions (fit on TRAIN only).  Together with the per-direction decodability profile
this closes the argument: if accuracy needs the low-variance tail, and the late model
reproduces only the high-variance head, then any variance-weighted retention loss is
optimising the wrong subspace.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

REPO = Path("/mnt/huawei_deepcad/dinov3")
FROZEN = REPO / "outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921/frozen"
REG = {"conic-cell-count", "livecell-cell-count"}


def load(ds, role, split):
    fs = sorted(glob.glob(f"{FROZEN}/{ds}/{role}/features/{ds}/*_{split}.npz"))
    d = np.load(fs[0], allow_pickle=True)
    return np.asarray(d["features"], np.float32), np.asarray(d["labels"]), np.asarray(d["paths"])


def probe(task, xtr, ytr, xte, yte):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, r2_score
    if task == "regression":
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)); m.fit(xtr, ytr.astype(float))
        return float(r2_score(yte.astype(float), m.predict(xte)))
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000,
                      class_weight="balanced", C=1.0, random_state=0))
    m.fit(xtr, ytr.astype(int))
    return float(balanced_accuracy_score(yte.astype(int), m.predict(xte)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--roles", default="E,L")
    ap.add_argument("--ks", default="8,32,128,512,1024,2048")
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/00_reports/hs6_l5_spectral_truncation_20260923")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in a.ks.split(",")]
    rep = {}
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        task = "regression" if ds in REG else "classification"
        rep[ds] = {"task": task, "ks": ks, "roles": {}}
        print(f"\n=== {ds} [{task}] ===", flush=True)
        for role in a.roles.split(","):
            xtr, ytr, _ = load(ds, role, "train"); xte, yte, _ = load(ds, role, "test")
            n = len(ytr); idx = np.arange(n)
            if n > a.max_train:
                idx = np.random.default_rng(0).choice(n, a.max_train, replace=False); idx.sort()
            c = xtr[idx].astype(np.float64); mu = c.mean(0); c = c - mu
            ev, U = np.linalg.eigh(c.T @ c / max(1, len(c) - 1))
            o = np.argsort(ev)[::-1]; U = U[:, o]; ev = ev[o]
            row = {}
            for k in ks:
                V = U[:, :k]
                row[str(k)] = probe(task, (xtr[idx].astype(np.float64) - mu) @ V, ytr[idx],
                                    (xte.astype(np.float64) - mu) @ V, yte)
            rep[ds]["roles"][role] = {"scores": row,
                                      "var_frac": {str(k): float(ev[:k].sum() / ev.sum()) for k in ks}}
            print(f"  {role}: " + "  ".join(f"k={k}:{row[str(k)]:.4f}" for k in ks), flush=True)
            print(f"     var: " + "  ".join(
                f"k={k}:{ev[:k].sum()/ev.sum():.3f}" for k in ks), flush=True)
        (a.out / "truncation.json").write_text(json.dumps(rep, indent=1))
    print("\nwrote", a.out / "truncation.json")


if __name__ == "__main__":
    main()
