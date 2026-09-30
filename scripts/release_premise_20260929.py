#!/usr/bin/env python3
"""Premise check for a maturation-aware release rule (zero GPU, cached classification features).

In the E-eigenbasis (E standardised with train stats; per dataset), track per-direction recoverability
R2_k(t) of E from the ADAPTIVE student along 13175..17567. Directions whose R2_k keeps falling despite the
constraint = "drifting" set D (label-free in principle: it only uses anchor-vs-student residuals).
Then ask, with labels, whether D carries (a) the early readout (E protocol probe energy) or (b) the late
readout (L=29279 probe transported into E-space via ridge L-logits<-E). A release rule is admissible only if
D is poor in (a) and rich in (b).
"""
from __future__ import annotations
import glob, json, time, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
REPO = Path("/mnt/huawei_deepcad/dinov3")
CO = REPO / "outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921/frozen"
SR = REPO / "outputs/02_eval_runs/hs6_l5_selective_retention_v4_20260923/frozen"
DM = REPO / "outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927/frozen"
OUT = REPO / "outputs/00_reports/deepcad_method_20260927/release_premise_20260929"; OUT.mkdir(parents=True, exist_ok=True)
ADAPTIVE = {13175: SR / "{ds}/adaptive_formal_ck13175", 13663: SR / "{ds}/adaptive_formal_ck13663", 14151: SR / "{ds}/adaptive_formal_ck14151",
            14639: SR / "{ds}/adaptive_formal_ck14639", 15127: SR / "{ds}/adaptive_formal_ck15127", 15615: DM / "{ds}/adaptive_continue_v2_gpu2_ck15615",
            16103: DM / "{ds}/adaptive_continue_v2_gpu2_ck16103", 16591: DM / "{ds}/adaptive_continue_resume16103_gpu3_formal_ck16591",
            17079: DM / "{ds}/adaptive_continue_resume16103_gpu3_formal_ck17079", 17567: DM / "{ds}/adaptive_continue_resume16103_gpu3_formal_ck17567"}
DS = ["bloodmnist","breastmnist","chammi-allen-task1","chammi-allen-task2","chammi-cp-task1","chammi-cp-task2","chammi-cp-task3","chammi-hpa-task1","chammi-hpa-task2","dermamnist","nct-crc-he","octmnist","organamnist","organcmnist","organsmnist","pathmnist","pcam","pneumoniamnist","retinamnist","tissuemnist"]
MAXN, SEED, RIDGE, BANDS = 20000, 0, 1e-3, [(0,8),(8,32),(32,128),(128,512),(512,2048)]
def load(pat):
    fs = sorted(glob.glob(pat)); assert len(fs) == 1, (pat, fs); d = np.load(fs[0], allow_pickle=True)
    return np.asarray(d["features"], np.float32), np.asarray(d["labels"]).astype(int)
def sub(y, n):
    if len(y) <= n: return np.arange(len(y))
    rng = np.random.default_rng(SEED); idx = []
    for c, cnt in zip(*np.unique(y, return_counts=True)): idx.append(rng.choice(np.where(y == c)[0], min(max(1, int(round(n * cnt / len(y)))), cnt), replace=False))
    return np.sort(np.concatenate(idx))
def std(tr, te): mu, sd = tr.mean(0), tr.std(0) + 1e-6; return (tr - mu) / sd, (te - mu) / sd
def ridge(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0); g = xc.T @ xc; g.flat[:: g.shape[0] + 1] += RIDGE * np.trace(g) / g.shape[0]
    return np.linalg.solve(g, xc.T @ yc), x.mean(0), y.mean(0)
res = []
for ds in DS:
    t0 = time.time()
    Etr, ytr = load(f"{CO}/{ds}/E/features/{ds}/*_train.npz"); Ete, yte = load(f"{CO}/{ds}/E/features/{ds}/*_test.npz")
    itr, ite = sub(ytr, MAXN), sub(yte, MAXN); Etr, ytr, Ete, yte = Etr[itr], ytr[itr], Ete[ite], yte[ite]
    Ze_tr, Ze_te = std(Etr, Ete)
    lam, U = np.linalg.eigh((Ze_tr.T @ Ze_tr / (len(Ze_tr) - 1)).astype(np.float64)); lam, U = lam[::-1].clip(1e-8), U[:, ::-1]
    tU = (Ze_te - Ze_tr.mean(0)) @ U; tvar = (tU ** 2).sum(0)                     # per-direction test variance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); eclf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Ze_tr, ytr)
    We = eclf.coef_.astype(np.float64); We = We - We.mean(0, keepdims=True) if We.shape[0] > 1 else We
    e_energy = lam * ((We @ U) ** 2).sum(0); e_energy /= e_energy.sum()           # early readout energy per direction
    # late readout (L probe) transported into E-space: ridge E -> L-logits, energy of that map per E-direction
    Ltr, _ = load(f"{CO}/{ds}/L/features/{ds}/*_train.npz"); Lte, _ = load(f"{CO}/{ds}/L/features/{ds}/*_test.npz")
    Zl_tr, Zl_te = std(Ltr[itr], Lte[ite])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); lclf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Zl_tr, ytr)
    Wl = lclf.coef_.astype(np.float64); Wl = Wl - Wl.mean(0, keepdims=True) if Wl.shape[0] > 1 else Wl
    A, _, _ = ridge(Ze_tr.astype(np.float64), Zl_tr @ Wl.T)                      # (d, C): E-space readout that best predicts L logits
    l_energy = lam * ((A.T @ U) ** 2).sum(0); l_energy /= l_energy.sum()
    # per-direction recoverability trajectory under Adaptive
    traj = {}
    for ck, pat in ADAPTIVE.items():
        p = str(pat).format(ds=ds)
        fs = glob.glob(f"{p}/features/{ds}/*_train.npz")
        if not fs: continue
        Str, _ = load(f"{p}/features/{ds}/*_train.npz"); Ste, _ = load(f"{p}/features/{ds}/*_test.npz")
        Zs_tr, Zs_te = std(Str[itr], Ste[ite]); B, mx, my = ridge(Zs_tr.astype(np.float64), Ze_tr.astype(np.float64))
        rU = ((Zs_te - mx) @ B + my - Ze_te) @ U
        traj[ck] = 1 - (rU ** 2).sum(0) / tvar                                     # R2_k(ck)
    cks = sorted(traj); R = np.stack([traj[c] for c in cks])                       # (T, d)
    slope = np.polyfit(np.array(cks, float) / 1000, R, 1)[0]                        # per-direction R2 slope per 1k updates
    drop = R[0] - R[-1]
    # drifting set D = worst 20% by drop (largest fall in recoverability under the constraint)
    k20 = int(0.2 * len(drop)); D = np.argsort(-drop)[:k20]; mask = np.zeros(len(drop), bool); mask[D] = True
    row = {"dataset": ds, "checkpoints": cks, "R2_first_mean": float(R[0].mean()), "R2_last_mean": float(R[-1].mean()),
           "band_R2_first": {f"{a}-{b}": float(R[0][a:b].mean()) for a, b in BANDS}, "band_R2_last": {f"{a}-{b}": float(R[-1][a:b].mean()) for a, b in BANDS},
           "D_size": int(k20), "D_variance_share": float(lam[mask].sum() / lam.sum()), "D_mean_rank": float(np.mean(D)),
           "early_readout_energy_in_D": float(e_energy[mask].sum()), "late_readout_energy_in_D": float(l_energy[mask].sum()),
           "early_energy_per_var_D_over_rest": float((e_energy[mask].sum() / lam[mask].sum()) / (e_energy[~mask].sum() / lam[~mask].sum())),
           "late_energy_per_var_D_over_rest": float((l_energy[mask].sum() / lam[mask].sum()) / (l_energy[~mask].sum() / lam[~mask].sum())),
           "corr_drop_vs_early_energy": float(np.corrcoef(drop, e_energy)[0, 1]), "corr_drop_vs_late_energy": float(np.corrcoef(drop, l_energy)[0, 1]),
           "corr_drop_vs_log_lambda": float(np.corrcoef(drop, np.log(lam))[0, 1])}
    res.append(row)
    print(f"{ds:20s} {time.time()-t0:4.0f}s R2 mean {row['R2_first_mean']:.3f}->{row['R2_last_mean']:.3f} | D(20%) var share {row['D_variance_share']:.2f} mean rank {row['D_mean_rank']:.0f} | early energy in D {row['early_readout_energy_in_D']:.2f} late {row['late_readout_energy_in_D']:.2f} | per-var ratio early {row['early_energy_per_var_D_over_rest']:.2f} late {row['late_energy_per_var_D_over_rest']:.2f} | corr(drop,early) {row['corr_drop_vs_early_energy']:+.2f} corr(drop,late) {row['corr_drop_vs_late_energy']:+.2f} corr(drop,logλ) {row['corr_drop_vs_log_lambda']:+.2f}", flush=True)
    json.dump(res, (OUT / "results.json").open("w"), indent=1)
print("done", flush=True)
