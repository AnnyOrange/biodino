#!/usr/bin/env python3
"""Frozen-feature retrieval + clustering benchmark for a DINOv3 checkpoint.

In-repo port of `benchmark_model/run_dinov3_retrieval_clustering_benchmark.py`
(source of `benchmark_results_retrieval_clustering.md`). Features are the same
L2-normalised frozen features as the classification benchmark; metrics are
kNN retrieval (recall@k / mAP@k / MRR) and MiniBatchKMeans clustering aligned
with Hungarian matching (cluster_accuracy / ARI / NMI / silhouette).

Example::

    python -m dinov3.eval.bio_frozen_eval.run_retrieval_clustering \
        --checkpoint /path/to/ckpt/12299/checkpoint.pth \
        --train-config /path/to/ckpt/config.yaml \
        --benchmark-root /mnt/huawei_deepcad/benchmark \
        --datasets lc25000 nct-crc-he-1k crc-val-he-7k \
        --output-dir outputs/bio_eval/retrieval --model-name dinov3-12299
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .encoder import CHANNEL_POLICIES, Dinov3CkptEncoder, completed, extract_features, parse_autocast_dtype
from .retrieval_clustering import (
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
CSV_FIELDS = [
    "model", "dataset", "task", "protocol", "aggregation",
    "n_gallery", "n_query", "n_samples", "n_classes",
    "recall_at_1", "recall_at_5", "recall_at_10",
    "map_at_1", "map_at_5", "map_at_10", "mrr",
    "cluster_accuracy", "ari", "nmi", "silhouette_cosine",
    "feature_file", "channel_policy", "channel_tta_samples", "checkpoint", "train_config", "error",
]


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _base_row(args, dataset: str, task: str, protocol: str, feature_file: Path) -> dict:
    return {
        "model": args.model_name,
        "dataset": dataset,
        "task": task,
        "protocol": protocol,
        "feature_file": str(feature_file),
        "channel_policy": args.channel_policy,
        "channel_tta_samples": args.channel_tta_samples,
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
    }


def _extract(dataset, encoder, feature_file: Path, args) -> tuple[np.ndarray, np.ndarray]:
    features, labels = extract_features(
        dataset,
        encoder,
        feature_file,
        args.batch_size,
        args.num_workers,
        args.overwrite_features,
        args.model_name,
        save_features=True,
        save_paths=True,
    )
    return features.astype(np.float32), np.asarray(labels)


def _run_hpa(args, encoder, out_root: Path, feature_stem: str) -> list[dict]:
    benchmark_root = Path(args.benchmark_root)
    protocol_root = Path(args.protocol_root or benchmark_root / "Retrieval_Clustering/protocols/v1")
    hpa_root = benchmark_root / "Retrieval_Clustering/HPA_Subcellular"
    gallery_ds = ManifestImageDataset(
        hpa_root,
        protocol_root / "hpa_same_gene_query_gallery.csv",
        role="gallery",
        max_samples=args.max_samples,
    )
    query_ds = ManifestImageDataset(
        hpa_root,
        protocol_root / "hpa_same_gene_query_gallery.csv",
        role="query",
        max_samples=args.max_samples,
    )
    cluster_ds = ManifestImageDataset(
        hpa_root,
        protocol_root / "hpa_single_location_clustering.csv",
        max_samples=args.max_samples,
    )
    feature_root = out_root / "features/hpa-subcellular"
    gallery_file = feature_root / f"{feature_stem}_gallery.npz"
    query_file = feature_root / f"{feature_stem}_query.npz"
    cluster_file = feature_root / f"{feature_stem}_cluster.npz"
    gallery_x, gallery_y = _extract(gallery_ds, encoder, gallery_file, args)
    query_x, query_y = _extract(query_ds, encoder, query_file, args)
    cluster_x, cluster_y = _extract(cluster_ds, encoder, cluster_file, args)
    robust_mask = np.asarray([row["robust_ge10"] == "1" for row in cluster_ds.rows])
    robust_x, robust_y = cluster_x[robust_mask], cluster_y[robust_mask]
    suffix = f"-subset-ms{args.max_samples}" if args.max_samples is not None else ""
    return [
        {
            **_base_row(
                args,
                "hpa-subcellular",
                "retrieval",
                f"custom-v1-same-gene-query-gallery{suffix}",
                gallery_file,
            ),
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
            **_base_row(
                args,
                "hpa-subcellular",
                "clustering",
                f"custom-v1-single-location-all41{suffix}",
                cluster_file,
            ),
            "aggregation": "location",
            "n_samples": len(cluster_y),
            "n_classes": len(np.unique(cluster_y)),
            **clustering_metrics(cluster_x, remap_labels(cluster_y), seed=args.seed),
        },
        {
            **_base_row(
                args,
                "hpa-subcellular",
                "clustering",
                f"custom-v1-single-location-ge10-34{suffix}",
                cluster_file,
            ),
            "aggregation": "location",
            "n_samples": len(robust_y),
            "n_classes": len(np.unique(robust_y)),
            **clustering_metrics(robust_x, remap_labels(robust_y), seed=args.seed),
        },
    ]


def _run_rxrx1(args, encoder, out_root: Path, feature_stem: str) -> list[dict]:
    benchmark_root = Path(args.benchmark_root)
    protocol_root = Path(args.protocol_root or benchmark_root / "Retrieval_Clustering/protocols/v1")
    manifest_name = (
        "rxrx1_official_cross_experiment.csv"
        if args.rxrx1_full
        else "rxrx1_official_cross_experiment_core.csv"
    )
    manifest = protocol_root / manifest_name
    archive = benchmark_root / "Retrieval_Clustering/RxRx1/archives/rxrx1-images.zip"
    gallery_ds = RxRx1ZipDataset(archive, manifest, "gallery", max_samples=args.max_samples)
    query_ds = RxRx1ZipDataset(archive, manifest, "query", max_samples=args.max_samples)
    feature_root = out_root / "features/rxrx1-cross"
    scope = "full" if args.rxrx1_full else "core"
    gallery_file = feature_root / f"{feature_stem}_{scope}_gallery.npz"
    query_file = feature_root / f"{feature_stem}_{scope}_query.npz"
    gallery_x, gallery_y = _extract(gallery_ds, encoder, gallery_file, args)
    query_x, query_y = _extract(query_ds, encoder, query_file, args)
    subset_suffix = f"-subset-ms{args.max_samples}" if args.max_samples is not None else ""
    protocol = f"official-cross-experiment-{scope}{subset_suffix}"
    rows = [{
        **_base_row(args, "rxrx1-cross", "retrieval", protocol, gallery_file),
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
            **_base_row(args, "rxrx1-cross", "retrieval", protocol, gallery_file),
            "aggregation": cell_type,
            "n_gallery": int(gallery_mask.sum()),
            "n_query": int(query_mask.sum()),
            "n_classes": len(np.unique(query_y[query_mask])),
            **metrics,
        })
    metric_names = list(cell_metrics[0])
    rows.append({
        **_base_row(args, "rxrx1-cross", "retrieval", protocol, gallery_file),
        "aggregation": "macro-cell-type",
        "n_gallery": len(gallery_y),
        "n_query": len(query_y),
        "n_classes": len(np.unique(query_y)),
        **{name: float(np.mean([metrics[name] for metrics in cell_metrics])) for name in metric_names},
    })
    combined_x = np.concatenate([gallery_x, query_x], axis=0)
    combined_y = np.concatenate([gallery_y, query_y], axis=0)
    rows.append({
        **_base_row(args, "rxrx1-cross", "clustering", protocol, gallery_file),
        "aggregation": "global-perturbation",
        "n_samples": len(combined_y),
        "n_classes": len(np.unique(combined_y)),
        **clustering_metrics(combined_x, remap_labels(combined_y), seed=args.seed),
    })
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="checkpoint.pth or a DCP checkpoint directory.")
    p.add_argument("--train-config", required=True)
    p.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    p.add_argument(
        "--protocol-root",
        help="Retrieval manifest directory; defaults to <benchmark-root>/Retrieval_Clustering/protocols/v1.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["lc25000", "nct-crc-he-1k", "crc-val-he-7k", "hpa-subcellular", "rxrx1-cross"],
        choices=DATASET_CHOICES,
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="dinov3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--metric-device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--metric-chunk-size", type=int, default=256)
    p.add_argument(
        "--rxrx1-full",
        action="store_true",
        help="Use all 112,824 RxRx1 treatment views instead of the balanced 17,728-view core manifest.",
    )
    p.add_argument("--n-last-blocks", type=int, default=1)
    p.add_argument("--no-avgpool", action="store_true")
    p.add_argument("--autocast-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--channel-policy", default="auto", choices=CHANNEL_POLICIES)
    p.add_argument("--channel-tta-samples", type=int, default=8)
    p.add_argument("--channel-policy-seed", type=int, default=0)
    p.add_argument("--overwrite-features", action="store_true")
    p.add_argument("--overwrite-results", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_root = Path(args.output_dir)
    checkpoint = Path(args.checkpoint)
    train_config = Path(args.train_config)
    autocast_dtype = parse_autocast_dtype(args.autocast_dtype)
    model_name = args.model_name
    summary_path = out_root / "summary.csv"
    failed_datasets: list[str] = []
    multi_dataset_run = len(args.datasets) > 1

    print(f"[ckpt] {model_name} checkpoint={checkpoint}", flush=True)
    encoder = Dinov3CkptEncoder(
        checkpoint=checkpoint,
        train_config=train_config,
        device=args.device,
        n_last_blocks=args.n_last_blocks,
        use_avgpool=not args.no_avgpool,
        autocast_dtype=autocast_dtype,
        channel_policy=args.channel_policy,
        channel_tta_samples=args.channel_tta_samples,
        channel_policy_seed=args.channel_policy_seed,
    )

    for dataset_name in args.datasets:
        if (
            args.max_samples is None
            and not (dataset_name == "rxrx1-cross" and args.rxrx1_full)
            and completed(
                summary_path,
                dataset_name,
                model_name,
                channel_policy=args.channel_policy,
                channel_tta_samples=args.channel_tta_samples,
            )
            and not args.overwrite_results
        ):
            print(f"[skip] {model_name} {dataset_name} already in {summary_path}", flush=True)
            continue
        feature_stem = model_name
        if args.channel_policy != "auto":
            feature_stem += f"_cp{args.channel_policy}"
            if args.channel_policy == "sample3_tta":
                feature_stem += f"_tta{args.channel_tta_samples}"
        if args.max_samples is not None:
            feature_stem += f"_ms{args.max_samples}"
        feature_file = out_root / "features" / dataset_name / f"{feature_stem}.npz"
        try:
            if dataset_name == "hpa-subcellular":
                rows = _run_hpa(args, encoder, out_root, feature_stem)
            elif dataset_name == "rxrx1-cross":
                rows = _run_rxrx1(args, encoder, out_root, feature_stem)
            else:
                dataset, classes = build_retrieval_dataset(
                    dataset_name, max_samples=args.max_samples, benchmark_root=args.benchmark_root
                )
                print(
                    f"[run] model={model_name} dataset={dataset_name} "
                    f"n={len(dataset)} classes={len(classes)}",
                    flush=True,
                )
                features, labels = extract_features(
                    dataset, encoder, feature_file, args.batch_size, args.num_workers,
                    args.overwrite_features, model_name, save_features=True, save_paths=True,
                )
                features = features.astype(np.float32)
                labels = labels.astype(int)
                rows = [{
                    "model": model_name,
                    "dataset": dataset_name,
                    "task": "retrieval_clustering",
                    "protocol": "within-set-leave-one-out",
                    "aggregation": "class",
                    "feature_file": str(feature_file),
                    "channel_policy": args.channel_policy,
                    "channel_tta_samples": args.channel_tta_samples,
                    "checkpoint": str(checkpoint),
                    "train_config": str(train_config),
                    "n_samples": int(len(labels)),
                    "n_classes": int(len(np.unique(labels))),
                    **retrieval_metrics(features, labels),
                    **clustering_metrics(features, labels, seed=args.seed),
                }]
        except Exception as exc:
            failed_datasets.append(dataset_name)
            rows = [{
                "model": model_name,
                "dataset": dataset_name,
                "task": "retrieval_clustering",
                "feature_file": str(feature_file),
                "channel_policy": args.channel_policy,
                "channel_tta_samples": args.channel_tta_samples,
                "checkpoint": str(checkpoint),
                "train_config": str(train_config),
                "error": f"{type(exc).__name__}: {exc}",
            }]
            print(f"[error] {model_name} {dataset_name}: {rows[0]['error']}", flush=True)
        for row in rows:
            append_csv(summary_path, row)
            print(json.dumps(row, indent=2), flush=True)
        failed = any(row.get("error") for row in rows)
        result_name = "failed_result.json" if failed else "last_result.json"
        result_dir = out_root / dataset_name if multi_dataset_run else out_root
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / result_name
        payload = rows[0] if len(rows) == 1 else {
            "model": model_name,
            "dataset": dataset_name,
            "rows": rows,
        }
        result_path.write_text(json.dumps(payload, indent=2))
        if failed:
            (result_dir / "last_result.json").unlink(missing_ok=True)
        if not failed:
            (result_dir / "failed_result.json").unlink(missing_ok=True)
    if failed_datasets:
        print(f"[failed] datasets={failed_datasets}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
