#!/usr/bin/env python3
"""Readout-vs-geometry diagnostic on cached frozen features (fleet-worker npz: features [N,2048] fp16, labels).

For a checkpoint's cached train/test features of a dataset, report:
  * protocol probe: StandardScaler + balanced LogisticRegression(C=1)        (what the benchmark scores)
  * tuned probe:    same, C chosen by 3-fold CV on a 20k train subsample      (is the information linearly there?)
  * kNN (k=20, cosine) balanced accuracy                                       (is the information in the geometry?)
  * effective rank (RankMe) of the train features, CLS-half anisotropy          (why a fixed penalty drifts)
Usage: readout_vs_geometry_20261014.py <features_dir_glob_root> <dataset> [<dataset> ...]
  root = .../results/point_<ck>  (searches */bio_classification/<ds>/<ck>/features/<ds>/*_{train,test}.npz)
"""
import glob, json, os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load(root, ds):
    tr = glob.glob(f"{root}/*/bio_classification/{ds}/*/features/{ds}/*_train.npz")
    te = glob.glob(f"{root}/*/bio_classification/{ds}/*/features/{ds}/*_test.npz")
    if not tr or not te: return None
    a, b = np.load(tr[0], allow_pickle=True), np.load(te[0], allow_pickle=True)
    return a["features"].astype(np.float32), a["labels"], b["features"].astype(np.float32), b["labels"]

def rankme(x, n=4000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.choice(len(x), min(n, len(x)), replace=False)
    z = x[idx] - x[idx].mean(0); s = np.linalg.svd(z, compute_uv=False); p = s / s.sum() + 1e-12
    return float(np.exp(-(p * np.log(p)).sum()))

def knn_bacc(xtr, ytr, xte, yte, k=20, chunk=2048):
    a = xtr / (np.linalg.norm(xtr, axis=1, keepdims=True) + 1e-8); b = xte / (np.linalg.norm(xte, axis=1, keepdims=True) + 1e-8)
    classes = np.unique(ytr); pred = np.empty(len(yte), dtype=ytr.dtype)
    for i in range(0, len(b), chunk):
        sim = b[i:i+chunk] @ a.T; nn = np.argpartition(-sim, k, axis=1)[:, :k]
        votes = np.zeros((len(nn), len(classes)))
        for j, c in enumerate(classes): votes[:, j] = (ytr[nn] == c).sum(1)
        pred[i:i+chunk] = classes[votes.argmax(1)]
    return balanced_accuracy_score(yte, pred)

def probe(xtr, ytr, xte, yte, C):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", C=C, n_jobs=8))
    clf.fit(xtr, ytr); return balanced_accuracy_score(yte, clf.predict(xte))

def main():
    root = sys.argv[1]; out = {}
    for ds in sys.argv[2:]:
        d = load(root, ds)
        if d is None: print(f"{ds}: no cached features under {root}"); continue
        xtr, ytr, xte, yte = d; t0 = time.time()
        rng = np.random.default_rng(0); sub = rng.choice(len(xtr), min(20000, len(xtr)), replace=False)
        cv_scores = {}
        for C in (0.001, 0.01, 0.1, 1.0, 10.0):
            sc = []
            for trn, val in StratifiedKFold(3, shuffle=True, random_state=0).split(xtr[sub], ytr[sub]):
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", C=C, n_jobs=8))
                clf.fit(xtr[sub][trn], ytr[sub][trn]); sc.append(balanced_accuracy_score(ytr[sub][val], clf.predict(xtr[sub][val])))
            cv_scores[C] = float(np.mean(sc))
        Cbest = max(cv_scores, key=cv_scores.get)
        res = dict(n_train=int(len(xtr)), n_test=int(len(xte)), dim=int(xtr.shape[1]),
                   protocol_C1=float(probe(xtr, ytr, xte, yte, 1.0)), tuned_C=Cbest, tuned=float(probe(xtr, ytr, xte, yte, Cbest)),
                   cv_curve=cv_scores, knn20=float(knn_bacc(xtr, ytr, xte, yte)),
                   rankme_full=rankme(xtr), rankme_cls=rankme(xtr[:, :xtr.shape[1]//2]), rankme_patchmean=rankme(xtr[:, xtr.shape[1]//2:]),
                   seconds=round(time.time()-t0, 1))
        out[ds] = res
        print(f"{ds:22s} protocol(C=1) {res['protocol_C1']:.4f} | tuned C={Cbest:<5g} {res['tuned']:.4f} | kNN20 {res['knn20']:.4f} | rankme full {res['rankme_full']:.0f} cls {res['rankme_cls']:.0f} patch {res['rankme_patchmean']:.0f} | {res['seconds']}s", flush=True)
    tag = os.path.basename(root.rstrip("/"))
    json.dump(out, open(f"/tmp/readout_vs_geometry_{tag}_{os.path.basename(os.path.dirname(os.path.dirname(root.rstrip('/'))))}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
