import csv
from pathlib import Path

import pytest

from scripts.analyze_hpa_cls_seed_replication import build_payload, parse_seed_root


FIELDNAMES = (
    "dataset",
    "task",
    "protocol",
    "aggregation",
    "map_at_5",
    "nmi",
    "error",
)


def write_summary(root: Path, arm: str, retrieval: float, clustering: float) -> None:
    path = root / arm / "nlb2_cls" / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "hpa-subcellular",
                "task": "retrieval",
                "protocol": "custom-v1-same-gene-query-gallery",
                "aggregation": "global",
                "map_at_5": retrieval,
            }
        )
        writer.writerow(
            {
                "dataset": "hpa-subcellular",
                "task": "clustering",
                "protocol": "custom-v1-single-location-ge10-34",
                "aggregation": "location",
                "nmi": clustering,
            }
        )


def make_seed(root: Path, seed: int, offset: float = 0.0) -> Path:
    seed_root = root / f"seed{seed}"
    write_summary(seed_root, "baseline", 0.050 + offset, 0.300 + offset)
    write_summary(seed_root, "true", 0.051 + offset, 0.316 + offset)
    write_summary(seed_root, "shuffled", 0.0505 + offset, 0.302 + offset)
    return seed_root


def test_build_payload_passes_consistent_two_seed_effect(tmp_path: Path) -> None:
    seed0 = make_seed(tmp_path, 0)
    seed1 = make_seed(tmp_path, 1, offset=0.001)
    payload = build_payload([(0, seed0), (1, seed1)], clustering_min_delta=0.01)
    assert payload["gates"]["replication_pass"] is True
    assert payload["aggregate"]["hpa_clustering_ge10"]["true_vs_baseline"]["values"] == pytest.approx([0.016, 0.016])


def test_build_payload_rejects_control_failure(tmp_path: Path) -> None:
    seed0 = make_seed(tmp_path, 0)
    seed1 = make_seed(tmp_path, 1)
    write_summary(seed1, "shuffled", retrieval=0.052, clustering=0.318)
    payload = build_payload([(0, seed0), (1, seed1)], clustering_min_delta=0.01)
    assert payload["gates"]["all_seeds_both_endpoints_beat_shuffled"] is False
    assert payload["gates"]["replication_pass"] is False


def test_seed_root_parser_requires_seed_equals_path() -> None:
    assert parse_seed_root("3=/tmp/run") == (3, Path("/tmp/run"))
    with pytest.raises(Exception):
        parse_seed_root("missing-separator")
