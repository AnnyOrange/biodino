from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def l2_normalize(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def encode_strings(values: Iterable[Any]) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    mapping: dict[str, int] = {}
    labels: list[int] = []
    for value in values:
        key = str(value)
        if key not in mapping:
            mapping[key] = len(names)
            names.append(key)
        labels.append(mapping[key])
    return np.asarray(labels, dtype=np.int64), names


def mean_by_group(features: np.ndarray, groups: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    groups_arr = np.asarray([str(g) for g in groups])
    unique = np.asarray(sorted(set(groups_arr.tolist())))
    out = []
    for group in unique:
        out.append(np.asarray(features)[groups_arr == group].mean(axis=0))
    return np.asarray(out, dtype=np.float32), unique


def first_by_group(values: np.ndarray, groups: Iterable[Any]) -> np.ndarray:
    groups_arr = np.asarray([str(g) for g in groups])
    values = np.asarray(values)
    out = []
    for group in sorted(set(groups_arr.tolist())):
        out.append(values[np.flatnonzero(groups_arr == group)[0]])
    return np.asarray(out)


def retrieval_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 512,
) -> dict[str, float]:
    if os.environ.get("DINOV3_OOD_RETRIEVAL_BACKEND", "").lower() == "torch":
        try:
            return _retrieval_metrics_torch(features, labels, k_values=k_values, chunk_size=chunk_size)
        except Exception:
            # Fall back to the NumPy implementation so metric computation is
            # robust on machines where CUDA is not available.
            pass
    x = l2_normalize(features)
    y = np.asarray(labels)
    n = x.shape[0]
    if n < 2:
        return {f"recall_at_{k}": float("nan") for k in k_values}
    max_k = min(max(k_values), n - 1)
    hits = {k: 0 for k in k_values}
    ap_sum = {k: 0.0 for k in k_values}
    reciprocal_sum = 0.0
    valid_ap = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = x[start:end] @ x.T
        row_ids = np.arange(start, end)
        sim[np.arange(end - start), row_ids] = -np.inf
        idx_part = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
        scores = np.take_along_axis(sim, idx_part, axis=1)
        order = np.argsort(-scores, axis=1)
        top_idx = np.take_along_axis(idx_part, order, axis=1)
        rel = y[top_idx] == y[row_ids, None]
        for local_i in range(end - start):
            rel_i = rel[local_i]
            if np.any(rel_i):
                reciprocal_sum += 1.0 / float(np.flatnonzero(rel_i)[0] + 1)
            total_relevant = int((y == y[row_ids[local_i]]).sum()) - 1
            if total_relevant > 0:
                valid_ap += 1
            for k in k_values:
                kk = min(k, max_k)
                rel_k = rel_i[:kk]
                hits[k] += int(np.any(rel_k))
                denom = min(total_relevant, kk)
                if denom > 0:
                    precisions = np.cumsum(rel_k) / np.arange(1, kk + 1)
                    ap_sum[k] += float((precisions * rel_k).sum() / denom)

    out: dict[str, float] = {"mrr": reciprocal_sum / n}
    for k in k_values:
        out[f"recall_at_{k}"] = hits[k] / n
        out[f"map_at_{k}"] = ap_sum[k] / max(1, valid_ap)
    return out


def _retrieval_metrics_torch(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 512,
) -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA backend requested but CUDA is not available")
    y_np = np.asarray(labels)
    n = int(len(y_np))
    if n < 2:
        return {f"recall_at_{k}": float("nan") for k in k_values}
    max_k = min(max(k_values), n - 1)

    device = torch.device("cuda")
    x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
    x = torch.nn.functional.normalize(x, dim=1)
    y = torch.as_tensor(y_np, device=device)
    _, inverse, counts = np.unique(y_np, return_inverse=True, return_counts=True)
    relevant_counts = counts[inverse] - 1

    hits = {k: 0 for k in k_values}
    ap_sum = {k: 0.0 for k in k_values}
    reciprocal_sum = 0.0
    valid_ap = 0

    with torch.inference_mode():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sim = x[start:end] @ x.T
            rows = torch.arange(start, end, device=device)
            sim[torch.arange(end - start, device=device), rows] = -torch.inf
            top_idx = torch.topk(sim, k=max_k, dim=1, largest=True, sorted=True).indices
            rel = (y[top_idx] == y[rows, None]).cpu().numpy()
            row_ids = np.arange(start, end)
            for local_i, row_id in enumerate(row_ids):
                rel_i = rel[local_i]
                if np.any(rel_i):
                    reciprocal_sum += 1.0 / float(np.flatnonzero(rel_i)[0] + 1)
                total_relevant = int(relevant_counts[row_id])
                if total_relevant > 0:
                    valid_ap += 1
                for k in k_values:
                    kk = min(k, max_k)
                    rel_k = rel_i[:kk]
                    hits[k] += int(np.any(rel_k))
                    denom = min(total_relevant, kk)
                    if denom > 0:
                        precisions = np.cumsum(rel_k) / np.arange(1, kk + 1)
                        ap_sum[k] += float((precisions * rel_k).sum() / denom)

    out: dict[str, float] = {"mrr": reciprocal_sum / n}
    for k in k_values:
        out[f"recall_at_{k}"] = hits[k] / n
        out[f"map_at_{k}"] = ap_sum[k] / max(1, valid_ap)
    return out


def clustering_metrics(features: np.ndarray, labels: np.ndarray, *, seed: int = 0) -> dict[str, float]:
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    x = l2_normalize(features)
    y = np.asarray(labels).astype(int)
    classes = np.unique(y)
    n_clusters = len(classes)
    if n_clusters < 2 or len(y) <= n_clusters:
        return {
            "cluster_accuracy": float("nan"),
            "ari": float("nan"),
            "nmi": float("nan"),
            "silhouette_cosine": float("nan"),
        }
    remap = {int(v): i for i, v in enumerate(classes)}
    y_dense = np.asarray([remap[int(v)] for v in y], dtype=np.int64)
    pred = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(2048, max(64, len(y))),
        n_init="auto",
    ).fit_predict(x)
    table = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for true, got in zip(y_dense, pred):
        table[int(true), int(got)] += 1
    rows, cols = linear_sum_assignment(-table)
    sample_size = min(5000, len(y))
    sil = silhouette_score(x, pred, metric="cosine", sample_size=sample_size, random_state=seed)
    return {
        "cluster_accuracy": float(table[rows, cols].sum() / len(y)),
        "ari": float(adjusted_rand_score(y_dense, pred)),
        "nmi": float(normalized_mutual_info_score(y_dense, pred)),
        "silhouette_cosine": float(sil),
    }


def classification_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels).astype(np.int64)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan"), "macro_f1": float("nan")}
    test_size = max(1.0 - train_fraction, 1.0 / len(y))
    if counts.min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, y))
    else:
        train_idx, test_idx = train_test_split(np.arange(len(y)), train_size=train_fraction, random_state=seed)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=1,
        ),
    )
    clf.fit(x[train_idx], y[train_idx])
    pred = clf.predict(x[test_idx])
    return {
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
        "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def binary_classification_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels).astype(np.int64)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or counts.min() < 2:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "auroc": float("nan"),
            "average_precision": float("nan"),
        }
    train_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed).split(x, y))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", class_weight="balanced", n_jobs=1),
    )
    clf.fit(x[train_idx], y[train_idx])
    probs = clf.predict_proba(x[test_idx])[:, 1]
    pred = (probs >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
        "auroc": float(roc_auc_score(y[test_idx], probs)),
        "average_precision": float(average_precision_score(y[test_idx], probs)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def regression_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    mask = np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(y) < 8 or np.unique(y).size < 3:
        return {"mae": float("nan"), "r2": float("nan"), "spearman": float("nan")}
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, train_size=train_fraction, random_state=seed)
    reg = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 13)),
    )
    reg.fit(x[train_idx], y[train_idx])
    pred = reg.predict(x[test_idx])
    rho = spearmanr(y[test_idx], pred).correlation
    return {
        "mae": float(mean_absolute_error(y[test_idx], pred)),
        "r2": float(r2_score(y[test_idx], pred)),
        "spearman": float(rho) if np.isfinite(rho) else float("nan"),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def binary_auc_ap(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.isfinite(scores)
    scores = scores[mask]
    labels = labels[mask]
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "average_precision": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def id_vs_ood_knn(
    id_features: np.ndarray,
    ood_features: np.ndarray,
    *,
    k: int = 10,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    id_x = l2_normalize(id_features)
    ood_x = l2_normalize(ood_features)
    idx = np.arange(len(id_x))
    rng.shuffle(idx)
    n_train = max(1, min(len(idx) - 1, int(round(len(idx) * train_fraction))))
    bank = id_x[idx[:n_train]]
    id_test = id_x[idx[n_train:]]
    k = max(1, min(k, len(bank)))

    def score(query: np.ndarray) -> np.ndarray:
        sim = query @ bank.T
        part = np.partition(sim, kth=sim.shape[1] - k, axis=1)[:, -k:]
        # Low similarity to the ID bank should mean more OOD-like.
        return 1.0 - part.mean(axis=1)

    scores = np.concatenate([score(id_test), score(ood_x)])
    labels = np.concatenate([np.zeros(len(id_test), dtype=np.int64), np.ones(len(ood_x), dtype=np.int64)])
    out = binary_auc_ap(scores, labels)
    out.update({"id_bank": int(len(bank)), "id_test": int(len(id_test)), "ood_test": int(len(ood_x))})
    return out


def xray_pair_retrieval(volume_features: np.ndarray, tomo_ids: np.ndarray, variants: np.ndarray) -> dict[str, float]:
    x = l2_normalize(volume_features)
    tomo_ids = np.asarray([str(v) for v in tomo_ids])
    variants = np.asarray([str(v) for v in variants])
    hits = 0
    total = 0
    ranks: list[int] = []
    for i in range(len(x)):
        opposite = variants != variants[i]
        if not opposite.any():
            continue
        candidates = np.flatnonzero(opposite)
        sims = x[candidates] @ x[i]
        order = candidates[np.argsort(-sims)]
        match = np.flatnonzero(tomo_ids[order] == tomo_ids[i])
        if len(match) == 0:
            continue
        rank = int(match[0]) + 1
        ranks.append(rank)
        hits += int(rank == 1)
        total += 1
    if total == 0:
        return {"pair_recall_at_1": float("nan"), "pair_mrr": float("nan"), "pair_queries": 0}
    return {
        "pair_recall_at_1": float(hits / total),
        "pair_mrr": float(np.mean([1.0 / r for r in ranks])),
        "pair_queries": int(total),
    }


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
