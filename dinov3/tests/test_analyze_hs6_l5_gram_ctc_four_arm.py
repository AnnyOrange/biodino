import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_hs6_l5_gram_ctc_four_arm import (
    ARMS,
    EXPECTED_ADMISSION,
    EXPECTED_MODELS,
    EXPECTED_PROTOCOL,
    EXPECTED_SSL_INTERVENTIONS,
    METRICS,
    AuditError,
    analyze_campaign,
    write_outputs,
)


FOLD_DOMAINS = {
    0: ["domain0", "domain1"],
    1: ["domain2", "domain3"],
    2: ["domain4"],
    3: ["domain5", "domain6"],
    4: ["domain7", "domain8", "domain9"],
}
ARM_OFFSETS = {"control": 0.0, "anchor7807": 0.02, "anchor17079": -0.01, "dual": 0.05}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _macro(rows: list[dict]) -> dict[str, float]:
    values = {}
    for metric in METRICS:
        container = "ctc_metrics" if metric in {"TRA", "SEG", "DET"} else "extra_segmentation_metrics"
        values[metric] = sum(row[container][metric] for row in rows) / len(rows)
    return values


def _domain_row(arm: str, domain: str, fold: int, manifest_sha: str) -> dict:
    domain_index = int(domain.removeprefix("domain"))
    value = 0.2 + domain_index * 0.025 + ARM_OFFSETS[arm]
    return {
        "status": "VALID_COMPLETE",
        "admission": EXPECTED_ADMISSION,
        "protocol_id": EXPECTED_PROTOCOL,
        "campaign_manifest_sha256": manifest_sha,
        "head_sha256": f"head-{arm}-{fold}",
        "fold": fold,
        "domain": domain,
        "ctc_metrics": {"Valid": 1, "TRA": value, "SEG": value + 0.01, "DET": value + 0.02},
        "extra_segmentation_metrics": {
            "mean_foreground_dice": value + 0.03,
            "AP": value + 0.04,
            "AP50": value + 0.05,
            "AP75": value + 0.06,
        },
    }


def _build_fixture(root: Path, *, drift_dual_head: bool = False) -> None:
    common_folds = [
        {
            "fold": fold,
            "train_domains": [f"train{fold}"],
            "test_domains": domains,
            "train_samples": 10,
            "test_frames": 20,
        }
        for fold, domains in FOLD_DOMAINS.items()
    ]
    for arm in ARMS:
        head = {"epochs": 50, "seed": 0, "selection": "epoch_50_no_validation"}
        if arm == "dual" and drift_dual_head:
            head["epochs"] = 49
        checkpoint_sha = hashlib.sha256(f"checkpoint-{arm}".encode()).hexdigest()
        manifest = {
            "status": "LOCKED_BEFORE_RUN",
            "admission": EXPECTED_ADMISSION,
            "protocol_id": EXPECTED_PROTOCOL,
            "teacher_branch": "teacher",
            "branch_arm": arm,
            "models": [
                {
                    "model": EXPECTED_MODELS[arm],
                    "checkpoint_step": 20495,
                    "checkpoint": {"path": f"/{arm}.pth", "sha256": checkpoint_sha},
                }
            ],
            "ssl_training": {
                "student_start_checkpoint": 20007,
                "endpoint_checkpoint": 20495,
                "matched_updates": 488,
                "effective_global_batch": 64,
                "labels_used": False,
                "intervention": EXPECTED_SSL_INTERVENTIONS[arm],
            },
            "data_manifest": {"path": "/data.json", "sha256": "d" * 64},
            "source_split_manifest_sha256": "e" * 64,
            "folds": common_folds,
            "head": head,
            "inference": {"native_geometry": True, "crop_size": 256},
            "linker": {"cost": "fixed"},
            "scoring": {"aggregation": "unweighted macro over 10 domains"},
            "py_ctcmetrics": {"commit": "pinned"},
            "imagecodecs": {"wheel": {"sha256": "w" * 64}},
            "code": [{"path": "/evaluator.py", "sha256": "c" * 64}],
        }
        arm_root = root / arm
        manifest_path = arm_root / "campaign_manifest.json"
        _write_json(manifest_path, manifest)
        manifest_sha = _sha256(manifest_path)

        all_rows = []
        fold_rows = []
        for fold, domains in FOLD_DOMAINS.items():
            rows = [_domain_row(arm, domain, fold, manifest_sha) for domain in domains]
            all_rows.extend(rows)
            fold_rows.append(
                {
                    "status": "VALID_COMPLETE",
                    "protocol_id": EXPECTED_PROTOCOL,
                    "campaign_manifest_sha256": manifest_sha,
                    "model": EXPECTED_MODELS[arm],
                    "fold": fold,
                    "head_sha256": f"head-{arm}-{fold}",
                    "train_domains": [f"train{fold}"],
                    "test_domains": domains,
                    "domain_rows": rows,
                    "macro": _macro(rows),
                }
            )
        result = {
            "status": "VALID_COMPLETE",
            "admission": EXPECTED_ADMISSION,
            "protocol_id": EXPECTED_PROTOCOL,
            "campaign_manifest_sha256": manifest_sha,
            "model": EXPECTED_MODELS[arm],
            "checkpoint_step": 20495,
            "checkpoint_sha256": checkpoint_sha,
            "folds": fold_rows,
            "domain_rows": all_rows,
            "macro": _macro(all_rows),
        }
        _write_json(arm_root / "models" / EXPECTED_MODELS[arm] / "results.json", result)
        _write_json(
            arm_root / "validation_report.json",
            {
                "status": "VALID_COMPLETE",
                "admission": EXPECTED_ADMISSION,
                "protocol_id": EXPECTED_PROTOCOL,
                "campaign_manifest_sha256": manifest_sha,
                "expected_models": 1,
                "valid_models": 1,
                "expected_folds_per_model": 5,
                "expected_domains_per_model": 10,
                "errors": [],
            },
        )


def test_four_arm_analysis_uses_direct_domain_macro_and_fixed_comparisons(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _build_fixture(campaign)

    payload = analyze_campaign(campaign)

    assert payload["status"] == "VALID_COMPLETE"
    assert payload["composite_score"] is None
    assert len(payload["paired_domains"]) == 10
    assert payload["arms"]["control"]["macro"]["TRA"] == pytest.approx(0.3125)
    assert payload["arms"]["control"]["macro"]["TRA"] != pytest.approx(0.3025)
    assert set(payload["common_contract_sha256"]) >= {
        "protocol",
        "data",
        "split",
        "folds",
        "head",
        "inference",
        "linker",
        "scoring",
    }
    comparison = payload["comparisons"]["dual_minus_anchor7807"]["metrics"]["TRA"]
    assert comparison["macro_delta"] == pytest.approx(0.03)
    assert comparison["paired_domain_median_delta"] == pytest.approx(0.03)
    assert (comparison["win_count"], comparison["tie_count"], comparison["loss_count"]) == (10, 0, 0)

    output = tmp_path / "comparison"
    write_outputs(output, payload)
    assert {
        "comparison.json",
        "validation_report.json",
        "arm_macros.csv",
        "comparison_summary.csv",
        "paired_domain_deltas.csv",
        "README.md",
    } == {path.name for path in output.iterdir()}
    rows = (output / "paired_domain_deltas.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 4 * len(METRICS) * 10


def test_four_arm_analysis_rejects_common_contract_drift(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _build_fixture(campaign, drift_dual_head=True)

    with pytest.raises(AuditError, match="common head contract differs"):
        analyze_campaign(campaign)


def test_four_arm_analysis_rejects_missing_domain(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _build_fixture(campaign)
    result_path = campaign / "dual" / "models" / EXPECTED_MODELS["dual"] / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["domain_rows"].pop()
    _write_json(result_path, result)

    with pytest.raises(AuditError, match="expected 10 domain rows"):
        analyze_campaign(campaign)
