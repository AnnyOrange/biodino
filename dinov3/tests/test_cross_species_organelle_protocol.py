import csv
from pathlib import Path

import numpy as np

from scripts.build_hpa_cyclops_cross_species_protocol import ONTOLOGY, build_rows
from scripts.eval_hpa_cyclops_cross_species import macro_query_gallery_metrics


def test_protocol_uses_exact_hpa_labels_and_gfp_only(tmp_path: Path) -> None:
    manifest = (
        tmp_path
        / "Retrieval_Clustering/HPA_Subcellular/metadata/hpa_subcellular_subset_manifest.csv"
    )
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "main_location"))
        writer.writeheader()
        for _, human_label in ONTOLOGY.items():
            writer.writerow({"image_path": f"images/{human_label}.jpg", "main_location": human_label})
            writer.writerow(
                {
                    "image_path": f"images/{human_label}_multi.jpg",
                    "main_location": f"{human_label};Other",
                }
            )
    for yeast_label in ONTOLOGY:
        root = tmp_path / "Classification/cyclops-protein-loc" / yeast_label
        root.mkdir(parents=True)
        (root / f"{yeast_label}_1_gfp.tif").touch()
        (root / f"{yeast_label}_1_rfp.tif").touch()
    rows, report = build_rows(
        tmp_path,
        max_human_per_class=1,
        max_yeast_per_class=1,
        min_per_class=1,
        seed=0,
    )
    assert len(rows) == 2 * len(ONTOLOGY)
    assert report["classes"] == len(ONTOLOGY)
    assert all(";Other" not in row["image_path"] for row in rows)
    assert all(
        row["image_path"].endswith("_gfp.tif")
        for row in rows
        if row["role"] == "yeast"
    )


def test_macro_retrieval_weights_classes_equally() -> None:
    gallery_x = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery_y = np.asarray([0, 1])
    query_x = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query_y = np.asarray([0, 0, 1])
    metrics = macro_query_gallery_metrics(
        gallery_x,
        gallery_y,
        query_x,
        query_y,
        metric_device="cpu",
    )
    assert metrics["recall_at_1"] == 1.0
    assert metrics["map_at_5"] == 1.0
