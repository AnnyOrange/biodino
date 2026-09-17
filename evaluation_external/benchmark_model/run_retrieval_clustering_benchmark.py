#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_eval.encoders import MODEL_REGISTRY, extract_features
from benchmark_eval.retrieval_clustering import (
    ManifestImageDataset,
    RxRx1ZipDataset,
    build_retrieval_dataset,
    clustering_metrics,
    query_gallery_metrics,
    remap_labels,
    retrieval_metrics,
)
DATASET_CHOICES = [
    "lc25000",
    "nct-crc-he-100",
    "nct-crc-he-1k",
    "crc-val-he-7k",
    "hpa-subcellular",
    "rxrx1-cross",
]


def append_csv(path: Path, row: dict) -> None:
    fields = [
        "model", "dataset", "task", "protocol", "aggregation",
        "n_gallery", "n_query", "n_samples", "n_classes",
        "recall_at_1", "recall_at_5", "recall_at_10",
        "map_at_1", "map_at_5", "map_at_10", "mrr",
        "cluster_accuracy", "ari", "nmi", "silhouette_cosine",
        "feature_file", "encoder_preprocess", "channel_policy", "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def completed(summary_path: Path, dataset: str, model: str, rxrx1_full: bool = False) -> bool:
    if not summary_path.exists():
        return False
    try:
        with summary_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("dataset") != dataset or row.get("model") != model or row.get("error"):
                    continue
                if dataset == "rxrx1-cross":
                    expected_scope = "-full" if rxrx1_full else "-core"
                    if expected_scope not in row.get("protocol", ""):
                        continue
                    return True
                return True
    except Exception:
        return False
    return False


def _extract_arrays(dataset, model: str, feature_file: Path, args) -> tuple[np.ndarray, np.ndarray]:
    extract_features(
        dataset,
        model,
        feature_file,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite_features,
    )
    pack = np.load(feature_file, allow_pickle=True)
    return pack["features"].astype(np.float32), np.asarray(pack["labels"])


def _base_row(model: str, dataset: str, task: str, protocol: str, feature_file: Path) -> dict:
    return {
        "model": model,
        "dataset": dataset,
        "task": task,
        "protocol": protocol,
        "feature_file": str(feature_file),
        "encoder_preprocess": "model-native",
        "channel_policy": "first3",
    }


def _run_hpa(args, model: str, feature_stem: str) -> list[dict]:
    benchmark_root = Path(args.benchmark_root)
    protocol_root = Path(args.protocol_root or benchmark_root / "Retrieval_Clustering/protocols/v1")
    hpa_root = benchmark_root / "Retrieval_Clustering/HPA_Subcellular"
    retrieval_manifest = protocol_root / "hpa_same_gene_query_gallery.csv"
    gallery_ds = ManifestImageDataset(hpa_root, retrieval_manifest, role="gallery", max_samples=args.max_samples)
    query_ds = ManifestImageDataset(hpa_root, retrieval_manifest, role="query", max_samples=args.max_samples)
    cluster_ds = ManifestImageDataset(
        hpa_root,
        protocol_root / "hpa_single_location_clustering.csv",
        max_samples=args.max_samples,
    )
    feature_root = Path(args.output_dir) / "features/hpa-subcellular"
    gallery_file = feature_root / f"{feature_stem}_gallery.npz"
    query_file = feature_root / f"{feature_stem}_query.npz"
    cluster_file = feature_root / f"{feature_stem}_cluster.npz"
    gallery_x, gallery_y = _extract_arrays(gallery_ds, model, gallery_file, args)
    query_x, query_y = _extract_arrays(query_ds, model, query_file, args)
    cluster_x, cluster_y = _extract_arrays(cluster_ds, model, cluster_file, args)
    robust_mask = np.asarray([row["robust_ge10"] == "1" for row in cluster_ds.rows])
    robust_x, robust_y = cluster_x[robust_mask], cluster_y[robust_mask]
    suffix = f"-subset-ms{args.max_samples}" if args.max_samples is not None else ""
    return [
        {
            **_base_row(model, "hpa-subcellular", "retrieval", f"custom-v1-same-gene-query-gallery{suffix}", gallery_file),
            "aggregation": "global",
            "n_gallery": len(gallery_y),
            "n_query": len(query_y),
            "n_classes": len(np.unique(query_y)),
            **query_gallery_metrics(
                gallery_x,
                gallery_y,
                query_x,
                query_y,
                chunk_size=args.metric_chunk_size,
                metric_device=args.metric_device,
            ),
        },
        {
            **_base_row(model, "hpa-subcellular", "clustering", f"custom-v1-single-location-all41{suffix}", cluster_file),
            "aggregation": "location",
            "n_samples": len(cluster_y),
            "n_classes": len(np.unique(cluster_y)),
            **clustering_metrics(cluster_x, remap_labels(cluster_y), seed=args.seed),
        },
        {
            **_base_row(model, "hpa-subcellular", "clustering", f"custom-v1-single-location-ge10-34{suffix}", cluster_file),
            "aggregation": "location",
            "n_samples": len(robust_y),
            "n_classes": len(np.unique(robust_y)),
            **clustering_metrics(robust_x, remap_labels(robust_y), seed=args.seed),
        },
    ]


def _run_rxrx1(args, model: str, feature_stem: str) -> list[dict]:
    benchmark_root = Path(args.benchmark_root)
    protocol_root = Path(args.protocol_root or benchmark_root / "Retrieval_Clustering/protocols/v1")
    scope = "full" if args.rxrx1_full else "core"
    manifest_name = (
        "rxrx1_official_cross_experiment.csv"
        if args.rxrx1_full
        else "rxrx1_official_cross_experiment_core.csv"
    )
    archive = benchmark_root / "Retrieval_Clustering/RxRx1/archives/rxrx1-images.zip"
    gallery_ds = RxRx1ZipDataset(archive, protocol_root / manifest_name, "gallery", args.max_samples)
    query_ds = RxRx1ZipDataset(archive, protocol_root / manifest_name, "query", args.max_samples)
    feature_root = Path(args.output_dir) / "features/rxrx1-cross"
    gallery_file = feature_root / f"{feature_stem}_{scope}_gallery.npz"
    query_file = feature_root / f"{feature_stem}_{scope}_query.npz"
    gallery_x, gallery_y = _extract_arrays(gallery_ds, model, gallery_file, args)
    query_x, query_y = _extract_arrays(query_ds, model, query_file, args)
    suffix = f"-subset-ms{args.max_samples}" if args.max_samples is not None else ""
    protocol = f"official-cross-experiment-{scope}{suffix}"
    rows = [{
        **_base_row(model, "rxrx1-cross", "retrieval", protocol, gallery_file),
        "aggregation": "global",
        "n_gallery": len(gallery_y),
        "n_query": len(query_y),
        "n_classes": len(np.unique(query_y)),
        **query_gallery_metrics(
            gallery_x,
            gallery_y,
            query_x,
            query_y,
            chunk_size=args.metric_chunk_size,
            metric_device=args.metric_device,
        ),
    }]
    cell_metrics = []
    for cell_type in sorted(set(query_ds.cell_types)):
        gallery_mask = gallery_ds.cell_types == cell_type
        query_mask = query_ds.cell_types == cell_type
        metrics = query_gallery_metrics(
            gallery_x[gallery_mask],
            gallery_y[gallery_mask],
            query_x[query_mask],
            query_y[query_mask],
            chunk_size=args.metric_chunk_size,
            metric_device=args.metric_device,
        )
        cell_metrics.append(metrics)
        rows.append({
            **_base_row(model, "rxrx1-cross", "retrieval", protocol, gallery_file),
            "aggregation": cell_type,
            "n_gallery": int(gallery_mask.sum()),
            "n_query": int(query_mask.sum()),
            "n_classes": len(np.unique(query_y[query_mask])),
            **metrics,
        })
    metric_names = list(cell_metrics[0])
    rows.append({
        **_base_row(model, "rxrx1-cross", "retrieval", protocol, gallery_file),
        "aggregation": "macro-cell-type",
        "n_gallery": len(gallery_y),
        "n_query": len(query_y),
        "n_classes": len(np.unique(query_y)),
        **{name: float(np.mean([metrics[name] for metrics in cell_metrics])) for name in metric_names},
    })
    combined_x = np.concatenate([gallery_x, query_x], axis=0)
    combined_y = np.concatenate([gallery_y, query_y], axis=0)
    rows.append({
        **_base_row(model, "rxrx1-cross", "clustering", protocol, gallery_file),
        "aggregation": "global-perturbation",
        "n_samples": len(combined_y),
        "n_classes": len(np.unique(combined_y)),
        **clustering_metrics(combined_x, remap_labels(combined_y), seed=args.seed),
    })
    return rows


def run_model_dataset(args, model: str, dataset_name: str) -> list[dict]:
    out_dir = Path(args.output_dir)
    summary_path = out_dir / "summary.csv"
    if (
        args.max_samples is None
        and completed(summary_path, dataset_name, model, rxrx1_full=args.rxrx1_full)
        and not args.overwrite_results
    ):
        print(f"[skip] {model} {dataset_name} already in {summary_path}", flush=True)
        return []
    feature_stem = model + (f"_ms{args.max_samples}" if args.max_samples is not None else "")
    feature_file = out_dir / "features" / dataset_name / f"{feature_stem}.npz"
    try:
        if dataset_name == "hpa-subcellular":
            rows = _run_hpa(args, model, feature_stem)
        elif dataset_name == "rxrx1-cross":
            rows = _run_rxrx1(args, model, feature_stem)
        else:
            dataset, classes = build_retrieval_dataset(dataset_name, max_samples=args.max_samples)
            print(f"[run] model={model} dataset={dataset_name} n={len(dataset)} classes={len(classes)}", flush=True)
            features, labels = _extract_arrays(dataset, model, feature_file, args)
            labels = labels.astype(int)
            rows = [{
                **_base_row(model, dataset_name, "retrieval_clustering", "within-set-leave-one-out", feature_file),
                "aggregation": "class",
                "n_samples": int(len(labels)),
                "n_classes": int(len(np.unique(labels))),
                **retrieval_metrics(features, labels),
                **clustering_metrics(features, labels, seed=args.seed),
            }]
    except Exception as exc:
        rows = [{
            "model": model,
            "dataset": dataset_name,
            "task": "retrieval_clustering",
            "feature_file": str(feature_file),
            "error": f"{type(exc).__name__}: {exc}",
        }]
        print(f"[error] {model} {dataset_name}: {rows[0]['error']}", flush=True)
    for row in rows:
        append_csv(summary_path, row)
        print(json.dumps(row, indent=2), flush=True)
    payload = rows[0] if len(rows) == 1 else {"model": model, "dataset": dataset_name, "rows": rows}
    result_name = "last_result_full.json" if dataset_name == "rxrx1-cross" and args.rxrx1_full else "last_result.json"
    (out_dir / result_name).write_text(json.dumps(payload, indent=2))
    return rows


def main() -> int:
    from run_fm_rules_suite import main as rules_main
    return rules_main('retrieval')


def legacy_main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval/clustering benchmark on frozen image features")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY), help=f"Model names: {sorted(MODEL_REGISTRY) + ['cytoimagenet']}")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "lc25000",
            "nct-crc-he-100",
            "nct-crc-he-1k",
            "crc-val-he-7k",
            "hpa-subcellular",
            "rxrx1-cross",
        ],
        choices=DATASET_CHOICES,
    )
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument(
        "--protocol-root",
        help="Retrieval manifest directory; defaults to <benchmark-root>/Retrieval_Clustering/protocols/v1.",
    )
    parser.add_argument("--output-dir", default="benchmark_runs/retrieval_clustering")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--metric-device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--metric-chunk-size", type=int, default=256)
    parser.add_argument(
        "--rxrx1-full",
        action="store_true",
        help="Use all 112,824 RxRx1 treatment views instead of the balanced 17,728-view core manifest.",
    )
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()

    rows = []
    for model in args.models:
        for dataset_name in args.datasets:
            rows.extend(run_model_dataset(args, model, dataset_name))
    out_dir = Path(args.output_dir)
    run_name = "last_run_full.json" if args.rxrx1_full else "last_run.json"
    (out_dir / run_name).write_text(json.dumps(rows, indent=2))
    return int(any(row.get("error") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
