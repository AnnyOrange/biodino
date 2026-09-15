import json
from pathlib import Path

import pytest

from scripts.summarize_local_dino_weight_core_gate import _load_cells


def _write(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "last_result.json").write_text(json.dumps(payload), encoding="utf-8")


def _retrieval_payload(dataset: str) -> dict:
    if dataset == "hpa-subcellular":
        rows = [
            {
                "dataset": dataset,
                "task": "retrieval",
                "aggregation": "global",
                "protocol": "custom-v1-same-gene-query-gallery",
                "recall_at_1": 0.31,
            },
            {
                "dataset": dataset,
                "task": "clustering",
                "aggregation": "location",
                "protocol": "custom-v1-single-location-all41",
                "nmi": 0.41,
            },
            {
                "dataset": dataset,
                "task": "clustering",
                "aggregation": "location",
                "protocol": "custom-v1-single-location-ge10-34",
                "nmi": 0.51,
            },
        ]
    else:
        rows = [
            {
                "dataset": dataset,
                "task": "retrieval",
                "aggregation": "global",
                "protocol": "official-cross-experiment-core",
                "recall_at_1": 0.32,
            },
            {
                "dataset": dataset,
                "task": "retrieval",
                "aggregation": "macro-cell-type",
                "protocol": "official-cross-experiment-core",
                "recall_at_1": 0.42,
            },
            {
                "dataset": dataset,
                "task": "clustering",
                "aggregation": "global-perturbation",
                "protocol": "official-cross-experiment-core",
                "nmi": 0.52,
            },
        ]
    return {"dataset": dataset, "rows": rows}


def test_load_cells_selects_preregistered_rows_from_wrapped_results(tmp_path: Path) -> None:
    for index, dataset in enumerate(("bloodmnist", "bbbc048-cellcycle", "cyclops-protein-loc")):
        _write(
            tmp_path / dataset,
            {"dataset": dataset, "task": "classification", "balanced_accuracy": 0.7 + index / 100},
        )
    _write(
        tmp_path / "bbbc005",
        {"dataset": "bbbc005", "task": "regression", "r2": 0.81},
    )
    _write(tmp_path / "hpa", _retrieval_payload("hpa-subcellular"))
    _write(tmp_path / "rxrx1", _retrieval_payload("rxrx1-cross"))

    cells = _load_cells(tmp_path)

    assert cells["retrieval"] == {"hpa-subcellular": 0.31, "rxrx1-cross": 0.42}
    assert cells["clustering"] == {"hpa-subcellular": 0.51, "rxrx1-cross": 0.52}


def test_load_cells_rejects_ambiguous_primary_row(tmp_path: Path) -> None:
    payload = _retrieval_payload("hpa-subcellular")
    payload["rows"].append(dict(payload["rows"][0]))
    _write(tmp_path / "hpa", payload)

    with pytest.raises(ValueError, match="Expected one retrieval/hpa-subcellular row"):
        _load_cells(tmp_path)
