# Retrieval + clustering datasets and metrics.
#
# Vendored from `benchmark_model/benchmark_eval/retrieval_clustering.py` (the
# source of `benchmark_results_retrieval_clustering.md`). The only change vs the
# reference is that `build_retrieval_dataset` takes an explicit `benchmark_root`
# (the reference hard-coded `/mnt/huawei_deepcad/benchmark`). Metric functions
# are byte-for-byte identical.
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .datasets import IMAGE_EXTS, load_image

DEFAULT_BENCHMARK_ROOT = Path("/mnt/huawei_deepcad/benchmark")


@dataclass
class RetrievalSpec:
    name: str
    labels: list[str]


class PathLabelDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], classes: list[str]):
        if not samples:
            raise ValueError("No image samples found")
        self.samples = samples
        self.classes = classes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return load_image(path), int(label), str(path)


class ParquetImageDataset(Dataset):
    def __init__(self, rows: list[tuple[bytes, int, str]], classes: list[str]):
        if not rows:
            raise ValueError("No parquet image rows found")
        self.rows = rows
        self.classes = classes

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        image_bytes, label, sample_id = self.rows[idx]
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.convert("RGB"), int(label), sample_id


def _class_folder_samples(class_dirs: list[Path]) -> tuple[list[tuple[Path, int]], list[str]]:
    classes = [p.name for p in class_dirs]
    samples: list[tuple[Path, int]] = []
    for label, class_dir in enumerate(class_dirs):
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        samples.extend((p, label) for p in files)
    return samples, classes


def _read_nct_parquet(paths: list[Path], max_samples: int | None = None) -> list[tuple[bytes, int, str]]:
    import pyarrow.parquet as pq

    rows: list[tuple[bytes, int, str]] = []
    for parquet_path in paths:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(columns=["image", "label"], batch_size=256):
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for idx, (image, label) in enumerate(zip(images, labels)):
                if image is None or image.get("bytes") is None:
                    continue
                sample_id = image.get("path") or f"{parquet_path.stem}:{len(rows)}"
                rows.append((image["bytes"], int(label), sample_id))
                if max_samples is not None and len(rows) >= max_samples:
                    return rows
    return rows


def build_retrieval_dataset(
    name: str,
    max_samples: int | None = None,
    benchmark_root: str | Path | None = None,
) -> tuple[Dataset, list[str]]:
    retrieval_root = Path(benchmark_root or DEFAULT_BENCHMARK_ROOT) / "Retrieval_Clustering"
    if name == "lc25000":
        root = retrieval_root / "LC25000/images/lung_colon_image_set"
        class_dirs = [
            root / "colon_image_sets/colon_aca",
            root / "colon_image_sets/colon_n",
            root / "lung_image_sets/lung_aca",
            root / "lung_image_sets/lung_n",
            root / "lung_image_sets/lung_scc",
        ]
        samples, classes = _class_folder_samples(class_dirs)
        if max_samples is not None:
            # Keep class balance when a cap is requested.
            per_class = max(1, max_samples // len(classes))
            capped: list[tuple[Path, int]] = []
            for label in range(len(classes)):
                capped.extend([s for s in samples if s[1] == label][:per_class])
            samples = capped[:max_samples]
        return PathLabelDataset(samples, classes), classes

    nct_root = retrieval_root / "NCT-CRC-HE/owkin_hf_parquet/data"
    nct_classes = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    if name == "nct-crc-he-1k":
        paths = sorted(nct_root.glob("nct_crc_he_1k-*.parquet"))
        return ParquetImageDataset(_read_nct_parquet(paths, max_samples), nct_classes), nct_classes
    if name == "crc-val-he-7k":
        paths = sorted(nct_root.glob("crc_val_he_7k-*.parquet"))
        return ParquetImageDataset(_read_nct_parquet(paths, max_samples), nct_classes), nct_classes
    if name == "nct-crc-he-100":
        paths = sorted(nct_root.glob("nct_crc_he_100-*.parquet"))
        return ParquetImageDataset(_read_nct_parquet(paths, max_samples), nct_classes), nct_classes

    raise KeyError(f"Unknown retrieval/clustering dataset: {name}")


def retrieval_metrics(features: np.ndarray, labels: np.ndarray, k_values: tuple[int, ...] = (1, 5, 10), chunk_size: int = 512) -> dict[str, float]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    n = x.shape[0]
    max_k = min(max(k_values), max(1, n - 1))
    hits = {k: 0 for k in k_values}
    ap_sum = {k: 0.0 for k in k_values}
    reciprocal_sum = 0.0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = x[start:end] @ x.T
        row_ids = np.arange(start, end)
        sim[np.arange(end - start), row_ids] = -np.inf
        # argpartition is much faster than a full sort for top-k retrieval.
        idx_part = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
        scores = np.take_along_axis(sim, idx_part, axis=1)
        order = np.argsort(-scores, axis=1)
        top_idx = np.take_along_axis(idx_part, order, axis=1)
        rel = y[top_idx] == y[row_ids, None]
        for local_i in range(end - start):
            rel_i = rel[local_i]
            if np.any(rel_i):
                reciprocal_sum += 1.0 / float(np.flatnonzero(rel_i)[0] + 1)
            for k in k_values:
                kk = min(k, max_k)
                rel_k = rel_i[:kk]
                hits[k] += int(np.any(rel_k))
                denom = min(int((y == y[row_ids[local_i]]).sum()) - 1, kk)
                if denom > 0:
                    precisions = np.cumsum(rel_k) / np.arange(1, kk + 1)
                    ap_sum[k] += float((precisions * rel_k).sum() / denom)

    out: dict[str, float] = {"mrr": reciprocal_sum / n}
    for k in k_values:
        out[f"recall_at_{k}"] = hits[k] / n
        out[f"map_at_{k}"] = ap_sum[k] / n
    return out


def clustering_metrics(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict[str, float]:
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels).astype(int)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    n_clusters = int(len(np.unique(y)))
    pred = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=2048, n_init="auto").fit_predict(x)
    table = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for true, got in zip(y, pred):
        if true < n_clusters and got < n_clusters:
            table[int(true), int(got)] += 1
    rows, cols = linear_sum_assignment(-table)
    acc = table[rows, cols].sum() / len(y)
    sil = silhouette_score(x, pred, metric="cosine", sample_size=min(5000, len(y)), random_state=seed) if len(y) > n_clusters else float("nan")
    return {
        "cluster_accuracy": float(acc),
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "silhouette_cosine": float(sil),
    }
