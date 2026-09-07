#!/usr/bin/env python3
"""Build a deterministic human-to-yeast organelle retrieval protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ONTOLOGY = {
    "ACTIN": "Actin filaments",
    "CYTOPLASM": "Cytosol",
    "ENDOSOME": "Endosomes",
    "ER": "Endoplasmic reticulum",
    "GOLGI": "Golgi apparatus",
    "MITOCHONDRIA": "Mitochondria",
    "NUCLEARPERIPHERY": "Nuclear membrane",
    "NUCLEI": "Nucleoplasm",
    "NUCLEOLUS": "Nucleoli",
    "PEROXISOME": "Peroxisomes",
}
FIELDS = (
    "role",
    "species",
    "image_path",
    "label",
    "label_name",
    "source_label",
    "sample_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-human-per-class", type=int, default=64)
    parser.add_argument("--max-yeast-per-class", type=int, default=256)
    parser.add_argument("--min-per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def stable_order(paths: list[str], *, seed: int) -> list[str]:
    def digest(value: str) -> bytes:
        return hashlib.sha256(f"{seed}\0{value}".encode()).digest()

    return sorted(paths, key=lambda value: (digest(value), value))


def build_rows(
    benchmark_root: Path,
    *,
    max_human_per_class: int,
    max_yeast_per_class: int,
    min_per_class: int,
    seed: int,
) -> tuple[list[dict[str, str | int]], dict]:
    if min(max_human_per_class, max_yeast_per_class, min_per_class) <= 0:
        raise ValueError("Protocol sample limits must be positive")
    hpa_manifest = (
        benchmark_root
        / "Retrieval_Clustering/HPA_Subcellular/metadata/hpa_subcellular_subset_manifest.csv"
    )
    with hpa_manifest.open(newline="", encoding="utf-8") as handle:
        hpa_rows = list(csv.DictReader(handle))

    rows: list[dict[str, str | int]] = []
    counts: dict[str, dict[str, int]] = {}
    for label, (yeast_label, human_label) in enumerate(sorted(ONTOLOGY.items())):
        human_paths = [
            str(Path("Retrieval_Clustering/HPA_Subcellular") / row["image_path"])
            for row in hpa_rows
            if row["main_location"].strip() == human_label
        ]
        yeast_root = benchmark_root / "Classification/cyclops-protein-loc" / yeast_label
        yeast_paths = [
            str(path.relative_to(benchmark_root))
            for path in yeast_root.glob("*_gfp.tif")
            if path.is_file()
        ]
        human_paths = stable_order(human_paths, seed=seed)[:max_human_per_class]
        yeast_paths = stable_order(yeast_paths, seed=seed)[:max_yeast_per_class]
        if min(len(human_paths), len(yeast_paths)) < min_per_class:
            raise ValueError(
                f"{yeast_label}/{human_label} has human={len(human_paths)} "
                f"yeast={len(yeast_paths)}, below min-per-class={min_per_class}"
            )
        label_name = human_label.lower().replace(" ", "_")
        for role, species, source_label, paths in (
            ("human", "homo_sapiens", human_label, human_paths),
            ("yeast", "saccharomyces_cerevisiae", yeast_label, yeast_paths),
        ):
            for image_path in paths:
                rows.append(
                    {
                        "role": role,
                        "species": species,
                        "image_path": image_path,
                        "label": label,
                        "label_name": label_name,
                        "source_label": source_label,
                        "sample_id": hashlib.sha256(image_path.encode()).hexdigest()[:20],
                    }
                )
        counts[label_name] = {"human": len(human_paths), "yeast": len(yeast_paths)}
    report = {
        "protocol": "hpa-cyclops-shared-organelle-v1",
        "seed": seed,
        "classes": len(ONTOLOGY),
        "samples": len(rows),
        "counts": counts,
        "selection": {
            "human": "exact single HPA main_location",
            "yeast": "Cyclops GFP channel only",
            "ordering": "sha256(seed, relative_path)",
        },
        "ontology": ONTOLOGY,
    }
    return rows, report


def main() -> None:
    args = parse_args()
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite protocol outputs")
    rows, report = build_rows(
        args.benchmark_root,
        max_human_per_class=args.max_human_per_class,
        max_yeast_per_class=args.max_yeast_per_class,
        min_per_class=args.min_per_class,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report["output"] = str(args.output)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
