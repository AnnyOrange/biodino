#!/usr/bin/env python3
"""Run audited retrieval/clustering protocols for external frozen encoders."""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DINOV3_ROOT = Path("/mnt/huawei_deepcad/dinov3")
BENCHMARK_MODEL_ROOT = Path("/mnt/huawei_deepcad/benchmark_model")
sys.path.insert(0, str(DINOV3_ROOT / "scripts"))
sys.path.insert(0, str(DINOV3_ROOT))
sys.path.insert(0, str(BENCHMARK_MODEL_ROOT))

from benchmark_eval.encoders import MODEL_REGISTRY  # noqa: E402
from benchmark_eval.retrieval_clustering import clustering_metrics, retrieval_metrics  # noqa: E402
from dinov3.eval.bio_frozen_eval.encoder import extract_features  # noqa: E402
from dinov3.eval.bio_frozen_eval.registry import build_dataset  # noqa: E402
from run_external_fm_linear_probe import ExternalEncoderAdapter  # noqa: E402


PROTOCOLS = ("nct-cross", "hpa", "rxrx1-cross", "lc25000-diagnostic")
CSV_FIELDS = [
    "model", "dataset", "task", "protocol", "aggregation", "n_gallery",
    "n_query", "n_samples", "n_classes", "recall_at_1", "recall_at_5",
    "recall_at_10", "map_at_1", "map_at_5", "map_at_10", "mrr",
    "cluster_accuracy", "ari", "nmi", "silhouette_cosine", "feature_file",
    "encoder_preprocess", "channel_policy", "error",
]


class ManifestImageDataset(Dataset):
    def __init__(self, root: Path, manifest: Path, role: str | None = None, robust_only: bool = False):
        self.root = root
        with manifest.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if role is not None:
            rows = [row for row in rows if row.get("role") == role]
        if robust_only:
            rows = [row for row in rows if row.get("robust_ge10") == "1"]
        if not rows:
            raise ValueError(f"No rows selected from {manifest}")
        self.rows = rows
        self.labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = self.root / row["image_path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return image, int(row["label"]), str(path)


class RxRx1ZipDataset(Dataset):
    def __init__(self, archive: Path, manifest: Path, role: str):
        self.archive = archive
        with manifest.open(newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["role"] == role]
        if not self.rows:
            raise ValueError(f"No RxRx1 rows for role={role}")
        self.labels = np.asarray([int(row["sirna_id"]) for row in self.rows], dtype=np.int64)
        self.cell_types = np.asarray([row["cell_type"] for row in self.rows])
        self._zip: zipfile.ZipFile | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip"] = None
        return state

    def _reader(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.archive)
        return self._zip

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        channels = []
        reader = self._reader()
        for key in ("c1", "c2", "c3"):
            with reader.open(row[key]) as handle:
                with Image.open(io.BytesIO(handle.read())) as image:
                    channels.append(np.asarray(image.convert("L"), dtype=np.uint8))
        rgb = Image.fromarray(np.stack(channels, axis=-1), mode="RGB")
        return rgb, int(row["sirna_id"]), row["site_id"]


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def extract(dataset, encoder, path: Path, args) -> tuple[np.ndarray, np.ndarray]:
    return extract_features(
        dataset,
        encoder,
        path,
        args.batch_size,
        args.num_workers,
        args.overwrite_features,
        args.model,
        save_features=True,
        save_paths=False,
    )


def feature_root(args, protocol: str) -> Path:
    base = Path(args.feature_root) if args.feature_root else Path(args.output_dir) / "features"
    return base / protocol


def remap_labels(labels: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(labels), return_inverse=True)[1].astype(np.int64)


def query_gallery_metrics(
    gallery_features: np.ndarray,
    gallery_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    chunk_size: int = 256,
    metric_device: str = "auto",
) -> dict[str, float]:
    gallery_labels = np.asarray(gallery_labels).reshape(-1)
    query_labels = np.asarray(query_labels).reshape(-1)
    if not set(np.unique(query_labels)).issubset(set(np.unique(gallery_labels))):
        raise ValueError("Query labels are absent from the gallery")

    use_cuda = metric_device == "cuda" or (metric_device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    gallery = torch.as_tensor(gallery_features, dtype=torch.float32, device=device)
    gallery = torch.nn.functional.normalize(gallery, dim=1)
    gallery_y = torch.as_tensor(gallery_labels, device=device)
    k_values = (1, 5, 10)
    max_k = min(max(k_values), len(gallery_labels))
    hits = {k: 0 for k in k_values}
    ap_sum = {k: 0.0 for k in k_values}
    reciprocal_sum = 0.0

    class_counts = {label: int((gallery_labels == label).sum()) for label in np.unique(gallery_labels)}
    for start in range(0, len(query_labels), chunk_size):
        end = min(start + chunk_size, len(query_labels))
        query = torch.as_tensor(query_features[start:end], dtype=torch.float32, device=device)
        query = torch.nn.functional.normalize(query, dim=1)
        query_y = torch.as_tensor(query_labels[start:end], device=device)
        top_idx = torch.topk(query @ gallery.T, k=max_k, dim=1, largest=True, sorted=True).indices
        relevant = (gallery_y[top_idx] == query_y[:, None]).cpu().numpy()
        for local_idx, rel in enumerate(relevant):
            positives = class_counts[query_labels[start + local_idx]]
            positive_ranks = np.flatnonzero(rel)
            if len(positive_ranks):
                reciprocal_sum += 1.0 / float(positive_ranks[0] + 1)
            for k in k_values:
                kk = min(k, max_k)
                rel_k = rel[:kk]
                hits[k] += int(np.any(rel_k))
                denom = min(positives, kk)
                if denom:
                    precision = np.cumsum(rel_k) / np.arange(1, kk + 1)
                    ap_sum[k] += float((precision * rel_k).sum() / denom)
    n_query = len(query_labels)
    result = {"mrr": reciprocal_sum / n_query}
    for k in k_values:
        result[f"recall_at_{k}"] = hits[k] / n_query
        result[f"map_at_{k}"] = ap_sum[k] / n_query
    del gallery, gallery_y
    if use_cuda:
        torch.cuda.empty_cache()
    return result


def base_row(args, dataset: str, task: str, protocol: str, feature_file: str) -> dict:
    return {
        "model": args.model,
        "dataset": dataset,
        "task": task,
        "protocol": protocol,
        "feature_file": feature_file,
        "encoder_preprocess": "model-native",
        "channel_policy": "first3",
    }


def run_nct(args, encoder) -> list[dict]:
    gallery_ds, _ = build_dataset("nct-crc-he", "train", args.max_samples, None, args.benchmark_root)
    query_ds, _ = build_dataset("nct-crc-he", "test", args.max_samples, None, args.benchmark_root)
    # These filenames intentionally match the classification runner so one
    # extraction serves both the official linear probe and cross-set retrieval.
    root = feature_root(args, "nct-crc-he")
    gallery_file, query_file = root / f"{args.model}_train.npz", root / f"{args.model}_test.npz"
    gallery_x, gallery_y = extract(gallery_ds, encoder, gallery_file, args)
    query_x, query_y = extract(query_ds, encoder, query_file, args)
    retrieval = {
        **base_row(args, "nct-crc-he", "retrieval", "official-nct-gallery-to-crc-query", str(gallery_file)),
        "aggregation": "global",
        "n_gallery": len(gallery_y),
        "n_query": len(query_y),
        "n_classes": len(np.unique(query_y)),
        **query_gallery_metrics(gallery_x, gallery_y, query_x, query_y, args.metric_chunk_size, args.metric_device),
    }
    clustering = {
        **base_row(args, "nct-crc-he", "clustering", "official-crc-query-only", str(query_file)),
        "aggregation": "tissue-class",
        "n_samples": len(query_y),
        "n_classes": len(np.unique(query_y)),
        **clustering_metrics(query_x, remap_labels(query_y), seed=args.seed),
    }
    return [retrieval, clustering]


def run_hpa(args, encoder) -> list[dict]:
    protocol_root = Path(args.protocol_root)
    hpa_root = Path(args.benchmark_root) / "Retrieval_Clustering/HPA_Subcellular"
    retrieval_manifest = protocol_root / "hpa_same_gene_query_gallery.csv"
    clustering_manifest = protocol_root / "hpa_single_location_clustering.csv"
    gallery_ds = ManifestImageDataset(hpa_root, retrieval_manifest, role="gallery")
    query_ds = ManifestImageDataset(hpa_root, retrieval_manifest, role="query")
    cluster_ds = ManifestImageDataset(hpa_root, clustering_manifest)
    root = feature_root(args, "hpa")
    gallery_file, query_file = root / f"{args.model}_gallery.npz", root / f"{args.model}_query.npz"
    cluster_file = root / f"{args.model}_cluster41.npz"
    gallery_x, gallery_y = extract(gallery_ds, encoder, gallery_file, args)
    query_x, query_y = extract(query_ds, encoder, query_file, args)
    cluster_x, cluster_y = extract(cluster_ds, encoder, cluster_file, args)
    robust_mask = np.asarray([row["robust_ge10"] == "1" for row in cluster_ds.rows])
    robust_x, robust_y = cluster_x[robust_mask], cluster_y[robust_mask]
    return [
        {
            **base_row(args, "hpa-subcellular", "retrieval", "custom-v1-same-gene-query-gallery", str(gallery_file)),
            "aggregation": "global",
            "n_gallery": len(gallery_y),
            "n_query": len(query_y),
            "n_classes": len(np.unique(query_y)),
            **query_gallery_metrics(gallery_x, gallery_y, query_x, query_y, args.metric_chunk_size, args.metric_device),
        },
        {
            **base_row(args, "hpa-subcellular", "clustering", "custom-v1-single-location-all41", str(cluster_file)),
            "aggregation": "location",
            "n_samples": len(cluster_y),
            "n_classes": len(np.unique(cluster_y)),
            **clustering_metrics(cluster_x, remap_labels(cluster_y), seed=args.seed),
        },
        {
            **base_row(args, "hpa-subcellular", "clustering", "custom-v1-single-location-ge10-34", str(cluster_file)),
            "aggregation": "location",
            "n_samples": len(robust_y),
            "n_classes": len(np.unique(robust_y)),
            **clustering_metrics(robust_x, remap_labels(robust_y), seed=args.seed),
        },
    ]


def run_rxrx1(args, encoder) -> list[dict]:
    manifest = Path(args.protocol_root) / "rxrx1_official_cross_experiment.csv"
    archive = Path(args.benchmark_root) / "Retrieval_Clustering/RxRx1/archives/rxrx1-images.zip"
    gallery_ds = RxRx1ZipDataset(archive, manifest, "gallery")
    query_ds = RxRx1ZipDataset(archive, manifest, "query")
    root = feature_root(args, "rxrx1-cross")
    gallery_file, query_file = root / f"{args.model}_gallery.npz", root / f"{args.model}_query.npz"
    gallery_x, gallery_y = extract(gallery_ds, encoder, gallery_file, args)
    query_x, query_y = extract(query_ds, encoder, query_file, args)
    rows = [{
        **base_row(args, "rxrx1", "retrieval", "official-train-gallery-to-test-query", str(gallery_file)),
        "aggregation": "global",
        "n_gallery": len(gallery_y),
        "n_query": len(query_y),
        "n_classes": len(np.unique(query_y)),
        **query_gallery_metrics(gallery_x, gallery_y, query_x, query_y, args.metric_chunk_size, args.metric_device),
    }]
    cell_metrics = []
    for cell_type in sorted(set(query_ds.cell_types)):
        gallery_mask = gallery_ds.cell_types == cell_type
        query_mask = query_ds.cell_types == cell_type
        metrics = query_gallery_metrics(
            gallery_x[gallery_mask], gallery_y[gallery_mask], query_x[query_mask], query_y[query_mask],
            args.metric_chunk_size, args.metric_device,
        )
        cell_metrics.append(metrics)
        rows.append({
            **base_row(args, "rxrx1", "retrieval", "official-cross-experiment-same-cell-type", str(gallery_file)),
            "aggregation": cell_type,
            "n_gallery": int(gallery_mask.sum()),
            "n_query": int(query_mask.sum()),
            "n_classes": len(np.unique(query_y[query_mask])),
            **metrics,
        })
    metric_names = list(cell_metrics[0])
    rows.append({
        **base_row(args, "rxrx1", "retrieval", "official-cross-experiment-same-cell-type", str(gallery_file)),
        "aggregation": "macro-cell-type",
        "n_gallery": len(gallery_y),
        "n_query": len(query_y),
        "n_classes": len(np.unique(query_y)),
        **{name: float(np.mean([metrics[name] for metrics in cell_metrics])) for name in metric_names},
    })
    rows.extend([
        {
            **base_row(args, "rxrx1", "clustering", "official-test-query-perturbation", str(query_file)),
            "aggregation": "sirna",
            "n_samples": len(query_y),
            "n_classes": len(np.unique(query_y)),
            **clustering_metrics(query_x, remap_labels(query_y), seed=args.seed),
        },
        {
            **base_row(args, "rxrx1", "clustering", "official-test-query-cell-type", str(query_file)),
            "aggregation": "cell-type",
            "n_samples": len(query_y),
            "n_classes": len(np.unique(query_ds.cell_types)),
            **clustering_metrics(query_x, remap_labels(query_ds.cell_types), seed=args.seed),
        },
    ])
    return rows


def run_lc25000(args, encoder) -> list[dict]:
    dataset, _ = build_dataset("lc25000", "train", args.max_samples, None, args.benchmark_root)
    # Match the C25 diagnostic cache name to avoid extracting LC25000 twice.
    feature_file = feature_root(args, "lc25000") / f"{args.model}.npz"
    features, labels = extract(dataset, encoder, feature_file, args)
    return [
        {
            **base_row(args, "lc25000", "retrieval", "within-set-leave-one-out-diagnostic", str(feature_file)),
            "aggregation": "global",
            "n_samples": len(labels),
            "n_classes": len(np.unique(labels)),
            **retrieval_metrics(features, labels),
        },
        {
            **base_row(args, "lc25000", "clustering", "within-set-diagnostic", str(feature_file)),
            "aggregation": "class",
            "n_samples": len(labels),
            "n_classes": len(np.unique(labels)),
            **clustering_metrics(features, remap_labels(labels), seed=args.seed),
        },
    ]


RUNNERS = {
    "nct-cross": run_nct,
    "hpa": run_hpa,
    "rxrx1-cross": run_rxrx1,
    "lc25000-diagnostic": run_lc25000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument(
        "--protocol-root",
        default="/mnt/huawei_deepcad/benchmark/Retrieval_Clustering/protocols/v1",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-root", help="Optional shared feature-cache root across task runners")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--metric-chunk-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.output_dir)
    summary_path = out_root / "summary.csv"
    pending = [
        protocol for protocol in args.protocols
        if args.overwrite_results or not (out_root / "done" / args.model / f"{protocol}.json").exists()
    ]
    if not pending:
        print(f"[complete] {args.model}: all requested protocols already complete", flush=True)
        return 0
    encoder = ExternalEncoderAdapter(args.model, args.device, args.batch_size)
    failed = []
    for protocol in pending:
        print(f"[protocol] {args.model}/{protocol}", flush=True)
        try:
            rows = RUNNERS[protocol](args, encoder)
            for row in rows:
                append_csv(summary_path, row)
                print(json.dumps(row, indent=2), flush=True)
            marker = out_root / "done" / args.model / f"{protocol}.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"model": args.model, "protocol": protocol, "rows": rows}, indent=2))
        except Exception as exc:
            failed.append(protocol)
            row = {
                **base_row(args, protocol, "unknown", protocol, ""),
                "error": f"{type(exc).__name__}: {exc}",
            }
            append_csv(summary_path, row)
            print(f"[error] {args.model}/{protocol}: {row['error']}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del encoder
    if failed:
        print(f"[failed] {args.model}: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
