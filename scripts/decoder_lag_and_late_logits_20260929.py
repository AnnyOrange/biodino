#!/usr/bin/env python3
"""(B) Decoder-lag cost upper bound and (D) whether late gains are new information. CPU only, cached features.

(B) Along the original no-GRAM trajectory (fullregistry point_XXXX, every 488 updates), fit the ridge map
    student(t) -> E on TRAIN and score on TEST (R2 in E-standardised space, and readout-weighted R2 through
    E's protocol probe). Then apply the map fitted at t-488 to student(t). The drop is an UPPER bound on the
    online decoder's lag cost (its EMA window is ~50 updates, refit every 8).
(D) At M=20007 and L=29279 (coexistence tree, sample-aligned with E): fit the protocol probe on the late
    features; regress its TRAIN logits on E features (ridge) and report TEST R2 = fraction of the late readout
    that is a linear function of E. Low R2 = the late capability is new information not linearly in E.
"""
from __future__ import annotations
import csv, glob, json, time, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = Path("/mnt/huawei_deepcad/dinov3")
FR = REPO / "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908"
CO = REPO / "outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921/frozen"
OUT = REPO / "outputs/00_reports/deepcad_method_20260927/decoder_lag_20260929"; OUT.mkdir(parents=True, exist_ok=True)
LAG_DS = ["nct-crc-he", "bloodmnist", "chammi-cp-task3", "pathmnist", "dermamnist", "breastmnist", "chammi-allen-task2"]
LAG_CKS = [12687, 13175, 13663, 14151, 14639, 15127, 15615, 16103]
LATE_DS = ["bloodmnist","breastmnist","chammi-allen-task1","chammi-allen-task2","chammi-cp-task1","chammi-cp-task2","chammi-cp-task3","chammi-hpa-task1","chammi-hpa-task2","dermamnist","nct-crc-he","octmnist","organamnist","organcmnist","organsmnist","pathmnist","pcam","pneumoniamnist","retinamnist","tissuemnist"]
MAXN, SEED, RIDGE = 20000, 0, 1e-3

def load(pat):
    fs = sorted(glob.glob(pat)); assert len(fs) == 1, (pat, fs); d = np.load(fs[0], allow_pickle=True)
    return np.asarray(d["features"], np.float32), np.asarray(d["labels"]).astype(int)
def fr(ds, ck, s): return f"{FR}/point_{ck}/classification_*/bio_classification/{ds}/{ck}/features/{ds}/*_{s}.npz"
def co(ds, role, s): return f"{CO}/{ds}/{role}/features/{ds}/*_{s}.npz"
def sub(y, n):
    if len(y) <= n: return np.arange(len(y))
    rng = np.random.default_rng(SEED); idx = []
    for c, cnt in zip(*np.unique(y, return_counts=True)):
        idx.append(rng.choice(np.where(y == c)[0], min(max(1, int(round(n * cnt / len(y)))), cnt), replace=False))
    return np.sort(np.concatenate(idx))
def std(tr, te): mu, sd = tr.mean(0), tr.std(0) + 1e-6; return (tr - mu) / sd, (te - mu) / sd
def ridge(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0); g = xc.T @ xc; g.flat[:: g.shape[0] + 1] += RIDGE * np.trace(g) / g.shape[0]
    return np.linalg.solve(g, xc.T @ yc), x.mean(0), y.mean(0)
def r2(pred, target): return float(1 - ((pred - target) ** 2).sum() / ((target - target.mean(0)) ** 2).sum())

# ---------------- (B) lag -----------------
lag = []
for ds in LAG_DS:
    t0 = time.time()
    Etr, ytr = load(fr(ds, 12687, "train")); Ete, yte = load(fr(ds, 12687, "test"))
    itr, ite = sub(ytr, MAXN), sub(yte, MAXN); Etr, ytr, Ete, yte = Etr[itr], ytr[itr], Ete[ite], yte[ite]
    Ze_tr, Ze_te = std(Etr, Ete)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Ze_tr, ytr)
    W = clf.coef_.astype(np.float64); Wc = W - W.mean(0, keepdims=True) if W.shape[0] > 1 else W
    maps = {}; rows = []
    for ck in LAG_CKS:
        Str, _ = load(fr(ds, ck, "train")); Ste, _ = load(fr(ds, ck, "test"))
        Zs_tr, Zs_te = std(Str[itr], Ste[ite]); A, mx, my = ridge(Zs_tr.astype(np.float64), Ze_tr.astype(np.float64)); maps[ck] = (A, mx, my)
        pred = (Zs_te - mx) @ A + my
        row = {"dataset": ds, "checkpoint": ck, "r2_own": r2(pred, Ze_te), "readout_r2_own": r2(pred @ Wc.T, Ze_te @ Wc.T)}
        prev = LAG_CKS[LAG_CKS.index(ck) - 1] if ck != LAG_CKS[0] else None
        if prev in maps:
            Ap, mxp, myp = maps[prev]; pl = (Zs_te - mxp) @ Ap + myp   # previous-checkpoint map, current features (own standardisation)
            row.update(r2_lag488=r2(pl, Ze_te), readout_r2_lag488=r2(pl @ Wc.T, Ze_te @ Wc.T))
            row["lag_cost_r2"] = row["r2_own"] - row["r2_lag488"]; row["lag_cost_readout"] = row["readout_r2_own"] - row["readout_r2_lag488"]
        rows.append(row)
    lag += rows
    print(f"[lag] {ds:20s} {time.time()-t0:4.0f}s  " + " ".join(f"{r['checkpoint']}:own {r['r2_own']:.3f}/{r['readout_r2_own']:.3f}" + (f" lag {r['lag_cost_r2']:+.3f}/{r['lag_cost_readout']:+.3f}" if 'lag_cost_r2' in r else "") for r in rows), flush=True)
    json.dump(lag, (OUT / "lag_rows.json").open("w"), indent=1)

# ---------------- (D) late logits explained by E -----------------
late = []
for ds in LATE_DS:
    t0 = time.time()
    Etr, ytr = load(co(ds, "E", "train")); Ete, yte = load(co(ds, "E", "test"))
    itr, ite = sub(ytr, MAXN), sub(yte, MAXN); Etr, ytr, Ete, yte = Etr[itr], ytr[itr], Ete[ite], yte[ite]
    Ze_tr, Ze_te = std(Etr, Ete)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); eclf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Ze_tr, ytr)
    e_ba = float(np.mean([np.mean(eclf.predict(Ze_te)[yte == c] == c) for c in np.unique(yte)]))
    row = {"dataset": ds, "E_BA": e_ba, "roles": {}}
    for role in ("M", "L"):
        Str, _ = load(co(ds, role, "train")); Ste, _ = load(co(ds, role, "test"))
        Zs_tr, Zs_te = std(Str[itr], Ste[ite])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); lclf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Zs_tr, ytr)
        l_ba = float(np.mean([np.mean(lclf.predict(Zs_te)[yte == c] == c) for c in np.unique(yte)]))
        Wl = lclf.coef_.astype(np.float64); Wl = Wl - Wl.mean(0, keepdims=True) if Wl.shape[0] > 1 else Wl
        lg_tr, lg_te = Zs_tr @ Wl.T, Zs_te @ Wl.T                      # late readout logits
        A, mx, my = ridge(Ze_tr.astype(np.float64), lg_tr); expl = r2((Ze_te - mx) @ A + my, lg_te)   # late logits <- E
        # symmetric: E logits <- late features (recoverability of E's readout)
        We = eclf.coef_.astype(np.float64); We = We - We.mean(0, keepdims=True) if We.shape[0] > 1 else We
        B, mx2, my2 = ridge(Zs_tr.astype(np.float64), Ze_tr @ We.T); rec = r2((Zs_te - mx2) @ B + my2, Ze_te @ We.T)
        # BA achievable on the E-explained part of the late logits alone (probe = argmax of predicted late logits)
        pl = (Ze_te - mx) @ A + my; pred = lclf.classes_[pl.argmax(1)] if Wl.shape[0] > 1 else lclf.classes_[(pl[:, 0] > 0).astype(int)]
        ba_from_E = float(np.mean([np.mean(pred[yte == c] == c) for c in np.unique(yte)]))
        row["roles"][role] = {"late_BA": l_ba, "late_gain_over_E_pp": (l_ba - e_ba) * 100, "late_logits_explained_by_E_r2": expl,
                              "E_logits_recoverable_from_late_r2": rec, "late_probe_BA_using_only_E_explained_logits": ba_from_E}
    late.append(row)
    m, l = row["roles"]["M"], row["roles"]["L"]
    print(f"[late] {ds:20s} {time.time()-t0:4.0f}s E {e_ba*100:5.2f} | M: BA {m['late_BA']*100:5.2f} gain {m['late_gain_over_E_pp']:+5.2f} lateLogit<-E {m['late_logits_explained_by_E_r2']:.3f} Elogit<-M {m['E_logits_recoverable_from_late_r2']:.3f} BA(E-part) {m['late_probe_BA_using_only_E_explained_logits']*100:5.2f} | L: BA {l['late_BA']*100:5.2f} gain {l['late_gain_over_E_pp']:+5.2f} lateLogit<-E {l['late_logits_explained_by_E_r2']:.3f} Elogit<-L {l['E_logits_recoverable_from_late_r2']:.3f} BA(E-part) {l['late_probe_BA_using_only_E_explained_logits']*100:5.2f}", flush=True)
    json.dump(late, (OUT / "late_rows.json").open("w"), indent=1)
print("done", flush=True)
