#!/usr/bin/env python3
"""Freeze deterministic HPA and RxRx1 retrieval manifests from audited metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_hpa(source: Path, output_dir: Path) -> dict:
    with source.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    by_gene: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        by_gene[row["gene"]].append(row)

    gene_to_label = {gene: idx for idx, gene in enumerate(sorted(by_gene))}
    retrieval_rows: list[dict] = []
    for gene in sorted(by_gene):
        rows = sorted(by_gene[gene], key=lambda row: row["image_path"])
        if len(rows) < 2:
            raise ValueError(f"HPA gene {gene} has fewer than two images")
        gallery_count = len(rows) // 2
        for idx, row in enumerate(rows):
            retrieval_rows.append({
                "image_path": row["image_path"],
                "gene": gene,
                "gene_name": row["gene_name"],
                "label": gene_to_label[gene],
                "role": "gallery" if idx < gallery_count else "query",
                "main_location": row["main_location"],
            })

    single = [row for row in source_rows if row["main_location"] and ";" not in row["main_location"]]
    location_counts = Counter(row["main_location"] for row in single)
    location_to_label = {name: idx for idx, name in enumerate(sorted(location_counts))}
    clustering_rows = [
        {
            "image_path": row["image_path"],
            "gene": row["gene"],
            "gene_name": row["gene_name"],
            "location": row["main_location"],
            "label": location_to_label[row["main_location"]],
            "robust_ge10": int(location_counts[row["main_location"]] >= 10),
        }
        for row in sorted(single, key=lambda row: (row["main_location"], row["image_path"]))
    ]

    retrieval_path = output_dir / "hpa_same_gene_query_gallery.csv"
    clustering_path = output_dir / "hpa_single_location_clustering.csv"
    write_csv(
        retrieval_path,
        ["image_path", "gene", "gene_name", "label", "role", "main_location"],
        retrieval_rows,
    )
    write_csv(
        clustering_path,
        ["image_path", "gene", "gene_name", "location", "label", "robust_ge10"],
        clustering_rows,
    )
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "retrieval_manifest": str(retrieval_path),
        "retrieval_manifest_sha256": sha256(retrieval_path),
        "retrieval_rows": len(retrieval_rows),
        "retrieval_genes": len(by_gene),
        "gallery_rows": sum(row["role"] == "gallery" for row in retrieval_rows),
        "query_rows": sum(row["role"] == "query" for row in retrieval_rows),
        "clustering_manifest": str(clustering_path),
        "clustering_manifest_sha256": sha256(clustering_path),
        "single_location_rows": len(clustering_rows),
        "single_location_classes": len(location_counts),
        "robust_ge10_rows": sum(int(row["robust_ge10"]) for row in clustering_rows),
        "robust_ge10_classes": sum(count >= 10 for count in location_counts.values()),
    }


def build_rxrx1(source: Path, output_dir: Path) -> dict:
    rows: list[dict] = []
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["well_type"] != "treatment":
                continue
            prefix = f"rxrx1/images/{row['experiment']}/Plate{row['plate']}/{row['well']}_s{row['site']}"
            rows.append({
                "site_id": row["site_id"],
                "role": "gallery" if row["dataset"] == "train" else "query",
                "sirna_id": row["sirna_id"],
                "cell_type": row["cell_type"],
                "experiment": row["experiment"],
                "plate": row["plate"],
                **{f"c{channel}": f"{prefix}_w{channel}.png" for channel in range(1, 7)},
            })
    rows.sort(key=lambda row: (row["role"], row["cell_type"], row["experiment"], row["site_id"]))
    output = output_dir / "rxrx1_official_cross_experiment.csv"
    core_output = output_dir / "rxrx1_official_cross_experiment_core.csv"
    fields = ["site_id", "role", "sirna_id", "cell_type", "experiment", "plate"] + [
        f"c{channel}" for channel in range(1, 7)
    ]
    write_csv(output, fields, rows)
    gallery_experiments = {row["experiment"] for row in rows if row["role"] == "gallery"}
    query_experiments = {row["experiment"] for row in rows if row["role"] == "query"}
    if gallery_experiments & query_experiments:
        raise ValueError("RxRx1 gallery/query experiment sets overlap")
    gallery_sirna = {row["sirna_id"] for row in rows if row["role"] == "gallery"}
    query_sirna = {row["sirna_id"] for row in rows if row["role"] == "query"}
    missing = query_sirna - gallery_sirna
    if missing:
        raise ValueError(f"RxRx1 query perturbations absent from gallery: {sorted(missing)[:10]}")

    # Keep two sites per (cell type, perturbation, role). Restricting the core
    # to pairs represented on both sides preserves the cross-experiment
    # query/gallery contract while making repeated checkpoint sweeps tractable.
    pair_role_rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        pair_role_rows[(row["cell_type"], row["sirna_id"], row["role"])].append(row)
    eligible_pairs = sorted({
        (cell_type, sirna_id)
        for cell_type, sirna_id, _ in pair_role_rows
        if (cell_type, sirna_id, "gallery") in pair_role_rows
        and (cell_type, sirna_id, "query") in pair_role_rows
    })
    core_rows: list[dict] = []
    for cell_type, sirna_id in eligible_pairs:
        for role in ("gallery", "query"):
            selected = sorted(
                pair_role_rows[(cell_type, sirna_id, role)],
                key=lambda row: (row["experiment"], row["plate"], row["site_id"]),
            )[:2]
            core_rows.extend(selected)
    core_rows.sort(key=lambda row: (row["role"], row["cell_type"], row["sirna_id"], row["site_id"]))
    write_csv(core_output, fields, core_rows)

    core_gallery_sirna = {row["sirna_id"] for row in core_rows if row["role"] == "gallery"}
    core_query_sirna = {row["sirna_id"] for row in core_rows if row["role"] == "query"}
    if core_query_sirna - core_gallery_sirna:
        raise ValueError("RxRx1 core query perturbations are absent from the core gallery")
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "manifest": str(output),
        "manifest_sha256": sha256(output),
        "rows": len(rows),
        "gallery_rows": sum(row["role"] == "gallery" for row in rows),
        "query_rows": sum(row["role"] == "query" for row in rows),
        "gallery_experiments": len(gallery_experiments),
        "query_experiments": len(query_experiments),
        "sirna_classes": len(gallery_sirna | query_sirna),
        "cell_types": sorted({row["cell_type"] for row in rows}),
        "core_manifest": str(core_output),
        "core_manifest_sha256": sha256(core_output),
        "core_rows": len(core_rows),
        "core_gallery_rows": sum(row["role"] == "gallery" for row in core_rows),
        "core_query_rows": sum(row["role"] == "query" for row in core_rows),
        "core_paired_cell_type_sirna": len(eligible_pairs),
        "core_sirna_classes": len(core_gallery_sirna | core_query_sirna),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        default="/mnt/huawei_deepcad/benchmark/Retrieval_Clustering",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/huawei_deepcad/benchmark/Retrieval_Clustering/protocols/v1",
    )
    args = parser.parse_args()
    root = Path(args.benchmark_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol_version": "v1",
        "hpa": build_hpa(
            root / "HPA_Subcellular/metadata/hpa_subcellular_subset_manifest.csv",
            output_dir,
        ),
        "rxrx1": build_rxrx1(
            root / "RxRx1/metadata/rxrx1/metadata.csv",
            output_dir,
        ),
    }
    metadata_path = output_dir / "manifest_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
