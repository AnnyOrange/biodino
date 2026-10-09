#!/usr/bin/env python3
"""Mechanism probe: does prolonged SSL contract the representation spectrum, and is
the early representation still linearly decodable from the late one?

Per dataset and per checkpoint arm we report
  rankme      exp(entropy of the normalised singular-value spectrum) of the features
  pr          participation ratio (sum l)^2 / sum l^2 of the covariance eigenvalues
  d95         number of principal components holding 95% of the variance
each computed on the train bank, separately for the whole 2048-d readout and for its
two halves (final CLS block and final patch-mean block).

Decodability R2(A <- B) fits a ridge map on TRAIN and scores on TEST:
  1 - ||A_te - B_te W||^2 / ||A_te - mean_tr(A)||^2
An asymmetry R2(E<-L) << R2(L<-E) indicates asymmetric recoverability under this
finite-sample ridge probe. It does not establish information destruction:
conditioning, nonlinear recoding, and distribution shift are alternative causes.
"""
from __future__ import annotations

import argparse, glob, json
from pathlib import Path
import numpy as np

REPO = Path("/mnt/huawei_deepcad/dinov3")
FROZEN = REPO / "outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921/frozen"
ROLES = ("E", "M", "L")


def load(ds, role, split):
    fs = sorted(glob.glob(f"{FROZEN}/{ds}/{role}/features/{ds}/*_{split}.npz"))
    if len(fs) != 1:
        raise FileNotFoundError(f"{ds}/{role}/{split}")
    d = np.load(fs[0], allow_pickle=True)
    return np.asarray(d["features"], np.float32), np.asarray(d["paths"])


def spectrum_stats(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    x = x - x.mean(0)
    n = len(x)
    cov = x.T @ x / max(1, n - 1)
    ev = np.linalg.eigvalsh(cov)[::-1]
    ev = np.clip(ev, 0, None)
    tot = ev.sum()
    p = ev / tot
    nz = p[p > 0]
    rankme = float(np.exp(-(nz * np.log(nz)).sum()))
    pr = float(tot ** 2 / (ev ** 2).sum())
    c = np.cumsum(p)
    return {"rankme": rankme, "participation_ratio": pr,
            "d95": int(np.searchsorted(c, 0.95) + 1), "d99": int(np.searchsorted(c, 0.99) + 1),
            "top1_frac": float(p[0]), "dim": int(x.shape[1])}


def decode_r2(src_tr, dst_tr, src_te, dst_te, ridge_frac=1e-3) -> float:
    s = src_tr.astype(np.float64); t = dst_tr.astype(np.float64)
    ms, mt = s.mean(0), t.mean(0)
    sc, tc = s - ms, t - mt
    g = sc.T @ sc
    g.flat[:: g.shape[0] + 1] += ridge_frac * np.trace(g) / g.shape[0]
    w = np.linalg.solve(g, sc.T @ tc)
    pred = (src_te.astype(np.float64) - ms) @ w + mt
    num = ((dst_te.astype(np.float64) - pred) ** 2).sum()
    den = ((dst_te.astype(np.float64) - mt) ** 2).sum()
    return float(1 - num / den)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/00_reports/hs6_l5_spectral_decodability_20260923")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        tr, te, paths = {}, {}, {}
        for r in ROLES:
            xtr, ptr = load(ds, r, "train"); xte, pte = load(ds, r, "test")
            tr[r], te[r], paths[r] = xtr, xte, (ptr, pte)
        for r in ROLES[1:]:
            assert np.array_equal(paths["E"][0], paths[r][0]) and np.array_equal(paths["E"][1], paths[r][1])
        n = len(tr["E"])
        idx = np.arange(n)
        if n > a.max_train:
            idx = np.random.default_rng(0).choice(n, a.max_train, replace=False); idx.sort()
        d = tr["E"].shape[1]; h = d // 2
        blocks = {"all": slice(0, d), "cls": slice(0, h), "patchmean": slice(h, d)}
        print(f"\n=== {ds}  n_train={n} (used {len(idx)})  n_test={len(te['E'])}  d={d} ===", flush=True)
        # half-norm sanity: are the two halves separately unit-normalised?
        hn = {r: (float(np.linalg.norm(tr[r][0, :h])), float(np.linalg.norm(tr[r][0, h:]))) for r in ROLES}
        print(f"  first-row half norms (cls, patchmean): "
              + "  ".join(f"{r}=({x:.3f},{y:.3f})" for r, (x, y) in hn.items()), flush=True)
        spec = {}
        for b, sl in blocks.items():
            spec[b] = {r: spectrum_stats(tr[r][idx, sl]) for r in ROLES}
            print(f"  [{b:9s}] " + "  ".join(
                f"{r}: rankme={spec[b][r]['rankme']:7.1f} d95={spec[b][r]['d95']:4d}" for r in ROLES), flush=True)
        dec = {}
        for b, sl in blocks.items():
            dec[b] = {}
            for src in ROLES:
                for dst in ROLES:
                    if src == dst:
                        continue
                    dec[b][f"{dst}<-{src}"] = decode_r2(tr[src][idx, sl], tr[dst][idx, sl],
                                                        te[src][:, sl], te[dst][:, sl])
            print(f"  [{b:9s}] decodability R2: " + "  ".join(
                f"{k}={v:.3f}" for k, v in dec[b].items()), flush=True)
        rep[ds] = {"n_train": int(n), "n_used": int(len(idx)), "n_test": int(len(te["E"])),
                   "half_norms": hn, "spectrum": spec, "decodability": dec}
        (a.out / "spectral_decodability.json").write_text(json.dumps(rep, indent=1))
    print("\nwrote", a.out / "spectral_decodability.json")


if __name__ == "__main__":
    main()
