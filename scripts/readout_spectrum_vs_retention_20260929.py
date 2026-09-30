#!/usr/bin/env python3
"""Where does each task's readout live in E's spectrum, and which E-directions did Adaptive keep?

Mechanism analysis on cached frozen features (CPU only, no training, no new test runs).

Space: E (ck12687) features standardised per coordinate with TRAIN statistics. This is exactly
the space in which Adaptive's recovery loss is (approximately) an unweighted MSE, because its
per-channel weights are 1/Var(anchor_j). So the eigen-directions of the standardised E covariance
are the directions ordered by how much gradient the Adaptive loss can put on them.

Per classification dataset:
  A. protocol probe (LogisticRegression C=1, balanced) on standardised E train -> class weights W.
     Signal energy of the readout per eigen-band b:   S_b = sum_{k in b} lambda_k * ||U_k^T W||^2
     (fraction of the E-logit variance carried by band b).  Readout-norm fraction N_b likewise.
  B. ridge map student(ck16103) -> E, fit on train, scored on test, for Adaptive and no-GRAM:
     per-band R2, readout-weighted R2 (how well E's logits are recoverable), and E-prediction
     agreement (fraction of test samples whose recovered-logit argmax equals E's own argmax).
Joined with the capability-regret deltas at ck16103 from regret_v4_20260929.

Train subsampled to <= MAX_TRAIN per dataset (stratified, seed 0) for the probe/ridge fits: this
is a mechanism estimate of the readout direction, not a re-scoring of the protocol.
"""
from __future__ import annotations
import csv, glob, json, os, time, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = Path("/mnt/huawei_deepcad/dinov3")
E_ROOT = REPO / "outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921/frozen"
AD_ROOT = REPO / "outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927/frozen"
NG_ROOT = REPO / "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908/point_16103"
REGRET = REPO / "outputs/00_reports/deepcad_method_20260927/regret_v4_20260929/per_capability_ck16103.csv"
OUT = REPO / "outputs/00_reports/deepcad_method_20260927/readout_spectrum_20260929"
BANDS = [(0, 8), (8, 32), (32, 128), (128, 512), (512, 2048)]
MAX_TRAIN, MAX_TEST, SEED, RIDGE_FRAC = 20000, 20000, 0, 1e-3

def load(pattern):
    fs = sorted(glob.glob(pattern)); assert len(fs) == 1, (pattern, fs)
    d = np.load(fs[0], allow_pickle=True)
    return np.asarray(d["features"], np.float32), np.asarray(d["labels"]).astype(int)

def sources(ds):
    return {"E": (f"{E_ROOT}/{ds}/E/features/{ds}/*_{{s}}.npz"),
            "Adaptive": (f"{AD_ROOT}/{ds}/adaptive_continue_v2_gpu2_ck16103/features/{ds}/*_{{s}}.npz"),
            "no-GRAM": (f"{NG_ROOT}/classification_*/bio_classification/{ds}/16103/features/{ds}/*_{{s}}.npz")}

def strat_sub(y, n, seed):
    if len(y) <= n: return np.arange(len(y))
    rng = np.random.default_rng(seed); idx = []
    classes, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(classes, counts):
        take = max(1, int(round(n * cnt / len(y))))
        idx.append(rng.choice(np.where(y == c)[0], min(take, cnt), replace=False))
    return np.sort(np.concatenate(idx))

def standardise(tr, te):
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    return (tr - mu) / sd, (te - mu) / sd

def ridge_fit(x, y, frac):
    xc, yc = x - x.mean(0), y - y.mean(0)
    g = xc.T @ xc; g.flat[:: g.shape[0] + 1] += frac * np.trace(g) / g.shape[0]
    w = np.linalg.solve(g, xc.T @ yc)
    return w, x.mean(0), y.mean(0)

def band_fracs(vec_per_k):
    tot = vec_per_k.sum()
    return {f"{a}-{b}": float(vec_per_k[a:b].sum() / tot) for a, b in BANDS}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    regret = {r["metric_key"].split(":")[1]: r for r in csv.DictReader(REGRET.open()) if r["metric_key"].startswith("classification:")}
    ds_list = sorted(d for d in os.listdir(AD_ROOT) if d in regret and d != "chestmnist"
                     and glob.glob(sources(d)["Adaptive"].format(s="train")) and glob.glob(sources(d)["no-GRAM"].format(s="train")))
    print(f"{len(ds_list)} datasets: {ds_list}", flush=True)
    rows = []
    for ds in ds_list:
        t0 = time.time(); src = sources(ds)
        Etr, ytr = load(src["E"].format(s="train")); Ete, yte = load(src["E"].format(s="test"))
        itr, ite = strat_sub(ytr, MAX_TRAIN, SEED), strat_sub(yte, MAX_TEST, SEED)
        Etr, ytr, Ete, yte = Etr[itr], ytr[itr], Ete[ite], yte[ite]
        Ze_tr, Ze_te = standardise(Etr, Ete)
        cov = Ze_tr.T @ Ze_tr / (len(Ze_tr) - 1)
        lam, U = np.linalg.eigh(cov.astype(np.float64)); lam, U = lam[::-1].clip(0), U[:, ::-1]
        # ---- A. readout of the protocol probe on E ----------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, n_jobs=1).fit(Ze_tr, ytr)
        W = clf.coef_.astype(np.float64)                       # (C, d); binary -> (1, d)
        Wc = W - W.mean(0, keepdims=True) if W.shape[0] > 1 else W
        alpha = (Wc @ U) ** 2                                  # (C, d) squared coefficient per eigen-direction
        signal_k = lam * alpha.sum(0); norm_k = alpha.sum(0)
        e_ba = float(np.mean([np.mean(clf.predict(Ze_te)[yte == c] == c) for c in np.unique(yte)]))
        e_logit_te = Ze_te @ W.T + clf.intercept_; e_pred = clf.classes_[e_logit_te.argmax(1)] if W.shape[0] > 1 else clf.classes_[(e_logit_te[:, 0] > 0).astype(int)]
        row = {"dataset": ds, "n_train_used": int(len(ytr)), "n_test_used": int(len(yte)), "n_classes": int(len(np.unique(ytr))),
               "E_probe_BA_subsample": e_ba, "E_probe_BA_recorded": float(regret[ds]["no-GRAM"]) if False else None,
               "signal_frac": band_fracs(signal_k), "readout_norm_frac": band_fracs(norm_k), "var_frac": band_fracs(lam),
               "signal_top32": float(signal_k[:32].sum() / signal_k.sum()), "signal_top128": float(signal_k[:128].sum() / signal_k.sum()),
               "signal_tail512": float(signal_k[512:].sum() / signal_k.sum()),
               "effective_rank_of_signal": float(np.exp(-(lambda p: (p[p > 0] * np.log(p[p > 0])).sum())(signal_k / signal_k.sum()))),
               "arms": {}}
        # ---- B. recoverability of E from each ck16103 student -----------------------
        for arm in ("Adaptive", "no-GRAM"):
            Str, _ = load(src[arm].format(s="train")); Ste, _ = load(src[arm].format(s="test"))
            Zs_tr, Zs_te = standardise(Str[itr], Ste[ite])
            Wr, mx, my = ridge_fit(Zs_tr.astype(np.float64), Ze_tr.astype(np.float64), RIDGE_FRAC)
            pred = (Zs_te - mx) @ Wr + my
            resid = pred - Ze_te; target = Ze_te - my
            rU, tU = resid @ U, target @ U
            band_r2 = {f"{a}-{b}": float(1 - (rU[:, a:b] ** 2).sum() / (tU[:, a:b] ** 2).sum()) for a, b in BANDS}
            overall_r2 = float(1 - (resid ** 2).sum() / (target ** 2).sum())
            # readout-weighted: how well are E's task logits recoverable from the student?
            rl, tl = resid @ Wc.T, target @ Wc.T
            readout_r2 = float(1 - (rl ** 2).sum() / (tl ** 2).sum())
            rec_logit = pred @ W.T + clf.intercept_
            rec_pred = clf.classes_[rec_logit.argmax(1)] if W.shape[0] > 1 else clf.classes_[(rec_logit[:, 0] > 0).astype(int)]
            agree = float((rec_pred == e_pred).mean())
            rec_ba = float(np.mean([np.mean(rec_pred[yte == c] == c) for c in np.unique(yte)]))
            row["arms"][arm] = {"overall_r2": overall_r2, "band_r2": band_r2, "readout_r2": readout_r2,
                                "E_prediction_agreement": agree, "recovered_logit_BA": rec_ba}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                own = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, n_jobs=1).fit(Zs_tr, ytr)
            op = own.predict(Zs_te)
            row["arms"][arm]["own_probe_BA_subsample"] = float(np.mean([np.mean(op[yte == c] == c) for c in np.unique(yte)]))
        a, n = row["arms"]["Adaptive"], row["arms"]["no-GRAM"]
        row["delta"] = {"band_r2": {k: a["band_r2"][k] - n["band_r2"][k] for k in a["band_r2"]},
                        "readout_r2": a["readout_r2"] - n["readout_r2"], "agreement": a["E_prediction_agreement"] - n["E_prediction_agreement"],
                        "recovered_logit_BA": a["recovered_logit_BA"] - n["recovered_logit_BA"]}
        rg = regret[ds]
        row["regret"] = {"set": rg["set"], "tau_ck": int(rg["tau_ck"]), "delta_r": float(rg["delta_r"]),
                         "raw_delta_pp": (float(rg["Adaptive"]) - float(rg["no-GRAM"])) * 100,
                         "r_noGRAM": float(rg["r_no-GRAM"]), "r_Adaptive": float(rg["r_Adaptive"])}
        rows.append(row)
        print(f"{ds:22s} {time.time()-t0:5.0f}s  E-BA {e_ba:.3f} | signal top32 {row['signal_top32']:.2f} top128 {row['signal_top128']:.2f} tail512 {row['signal_tail512']:.2f} "
              f"| R2 bands Ad {' '.join(f'{v:.2f}' for v in a['band_r2'].values())} | nG {' '.join(f'{v:.2f}' for v in n['band_r2'].values())} "
              f"| readoutR2 Ad {a['readout_r2']:.3f} nG {n['readout_r2']:.3f} | agree Ad {a['E_prediction_agreement']:.3f} nG {n['E_prediction_agreement']:.3f} "
              f"| Δr {row['regret']['delta_r']:+.2f} raw {row['regret']['raw_delta_pp']:+.2f}pp", flush=True)
        (OUT / "results.json").write_text(json.dumps({"bands": BANDS, "max_train": MAX_TRAIN, "ridge_frac": RIDGE_FRAC, "rows": rows}, indent=1))
    # flat csv
    flat = []
    for r in rows:
        f = {"dataset": r["dataset"], "set": r["regret"]["set"], "tau_ck": r["regret"]["tau_ck"], "delta_r": r["regret"]["delta_r"], "raw_delta_pp": r["regret"]["raw_delta_pp"],
             "E_BA_sub": r["E_probe_BA_subsample"], "signal_top32": r["signal_top32"], "signal_top128": r["signal_top128"], "signal_tail512": r["signal_tail512"], "eff_rank_signal": r["effective_rank_of_signal"]}
        for k, v in r["signal_frac"].items(): f[f"signal_{k}"] = v
        for arm in ("Adaptive", "no-GRAM"):
            f[f"{arm}_overall_r2"] = r["arms"][arm]["overall_r2"]; f[f"{arm}_readout_r2"] = r["arms"][arm]["readout_r2"]; f[f"{arm}_agree"] = r["arms"][arm]["E_prediction_agreement"]; f[f"{arm}_recBA"] = r["arms"][arm]["recovered_logit_BA"]; f[f"{arm}_ownBA_sub"] = r["arms"][arm]["own_probe_BA_subsample"]
            for k, v in r["arms"][arm]["band_r2"].items(): f[f"{arm}_r2_{k}"] = v
        for k, v in r["delta"]["band_r2"].items(): f[f"delta_r2_{k}"] = v
        f["delta_readout_r2"] = r["delta"]["readout_r2"]; f["delta_agree"] = r["delta"]["agreement"]; f["delta_recBA"] = r["delta"]["recovered_logit_BA"]
        flat.append(f)
    with (OUT / "results_flat.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys())); w.writeheader(); w.writerows(flat)
    print("done", flush=True)

if __name__ == "__main__":
    main()
