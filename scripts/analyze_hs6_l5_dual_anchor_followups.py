#!/usr/bin/env python3
"""Fail-closed audit of the three HS6-L5 dual-anchor Gram follow-ups.

The decisions in this report use only controlled training logs and frozen,
label-free relation/spatial diagnostics.  Frozen labeled probes are deliberately
outside this analyzer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


EXPECTED_UPDATES = list(range(20008, 20496))
EXPECTED_UPDATE_COUNT = len(EXPECTED_UPDATES)
RELATION_METRIC = "relation_mse_including_diagonal"
FIXED_SPATIAL_DATASET = (
    "packwds_robust:/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/"
    "filtered_mixed_train_w*.tar::pct=1,99"
)
RELATION_DATASET = (
    "mixwds_robust:0.3=/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/"
    "filtered_mixed_train_w*.tar||0.7=/mnt/huawei_blm/deepcad_5t_v1/wds_patched_shuffle/"
    "filtered_mixed_train*.tar::pct=1,99"
)
CORE_LOSSES = (
    "total_loss",
    "dino_global_crops_loss",
    "dino_local_crops_loss",
    "ibot_loss",
    "koleo_loss",
)
MATCHED_CONFIG_PATHS = (
    "train.dataset_path",
    "train.batch_size_per_gpu",
    "train.num_workers",
    "train.seed",
    "train.OFFICIAL_EPOCH_LENGTH",
    "train.max_updates",
    "train.wds_deterministic_resampling",
    "train.wds_shuffle_buffer",
    "optim.gradient_accumulation_steps",
    "crops.global_crops_size",
    "crops.local_crops_size",
    "crops.gram_teacher_crops_size",
    "crops.gram_teacher_no_distortions",
    "crops.augmentation_policy",
    "crops.rgb_mean",
    "crops.rgb_std",
)


class AuditError(RuntimeError):
    """An input is incomplete or violates the locked protocol."""


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    run: Path
    patch_weight: float
    relation_weight: float
    observation_protocol: str
    allow_missing_protocol: bool = False
    required_mask_flag: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--strong-run", type=Path, required=True)
    parser.add_argument("--mask-matched-run", type=Path, required=True)
    parser.add_argument("--relation-only-run", type=Path, required=True)
    parser.add_argument("--relation-json", type=Path, required=True)
    parser.add_argument("--spatial-reference-dir", type=Path, required=True)
    parser.add_argument("--strong-spatial-dir", type=Path, required=True)
    parser.add_argument("--mask-matched-spatial-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-spatial-summary-name",
        default="dual.json",
        help="Summary filename used independently in each candidate spatial directory.",
    )
    parser.add_argument(
        "--candidate-spatial-records-name",
        default="dual_records.json",
        help="Records filename used independently in each candidate spatial directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_json(path: Path) -> Any:
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid JSON {path}: {error}") from error


def read_yaml(path: Path) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty config: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AuditError(f"invalid YAML {path}: {error}") from error
    require(isinstance(value, dict), f"config is not a mapping: {path}")
    return value


def nested(payload: dict[str, Any], dotted_path: str, *, missing: Any = ...) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            if missing is not ...:
                return missing
            raise AuditError(f"missing field {dotted_path}")
        value = value[part]
    return value


def finite_number(value: Any, context: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{context} must be numeric, got {value!r}",
    )
    result = float(value)
    require(math.isfinite(result), f"{context} is non-finite: {value!r}")
    return result


def exact_number(value: Any, expected: float, context: str) -> float:
    result = finite_number(value, context)
    require(
        math.isclose(result, expected, rel_tol=0.0, abs_tol=1e-12),
        f"{context}={result}, expected {expected}",
    )
    return result


@lru_cache(maxsize=None)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file() and resolved.stat().st_size > 0, f"missing or empty file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def run_checkpoint(run: Path, update: int) -> Path:
    return run.resolve() / "eval" / f"training_{update}" / "teacher_checkpoint.pth"


def run_config(run: Path) -> Path:
    return run.resolve() / "config.yaml"


def load_training_rows(run: Path, name: str) -> list[dict[str, Any]]:
    path = run.resolve() / "raw_loss_metrics.jsonl"
    require(path.is_file() and path.stat().st_size > 0, f"missing metrics for {name}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AuditError(f"invalid JSON in {path}:{line_number}: {error}") from error
        require(isinstance(row, dict), f"non-mapping row in {path}:{line_number}")
        rows.append(row)
    require(
        len(rows) == EXPECTED_UPDATE_COUNT,
        f"{name} has {len(rows)} metric rows, expected {EXPECTED_UPDATE_COUNT}",
    )
    updates = []
    for index, row in enumerate(rows):
        raw_update = row.get("optimizer_update")
        require(
            isinstance(raw_update, int) and not isinstance(raw_update, bool),
            f"{name} row {index} has invalid optimizer_update={raw_update!r}",
        )
        updates.append(raw_update)
    require(updates == EXPECTED_UPDATES, f"{name} update sequence is not ck20008--ck20495")
    return rows


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "first": values[0],
        "last": values[-1],
    }


def validate_loss_fields(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    collected: dict[str, list[float]] = {field: [] for field in CORE_LOSSES}
    for index, row in enumerate(rows):
        for field in CORE_LOSSES:
            require(field in row, f"{name} row {index} missing {field}")
        loss_fields = [key for key in row if key == "total_loss" or key.endswith("_loss")]
        require(loss_fields, f"{name} row {index} has no loss fields")
        for field in loss_fields:
            value = finite_number(row[field], f"{name} row {index} {field}")
            if field in collected:
                collected[field].append(value)
        exact_number(
            row.get("effective_global_batch_size"),
            64.0,
            f"{name} row {index} effective_global_batch_size",
        )
    return {field: summarize(values) for field, values in collected.items()}


def validate_candidate_rows(
    rows: list[dict[str, Any]], spec: CandidateSpec
) -> dict[str, Any]:
    relation_losses: list[float] = []
    patch_losses: list[float] = []
    present_mask_flags = 0
    for index, row in enumerate(rows):
        patch_losses.append(finite_number(row.get("gram_loss"), f"{spec.name} row {index} gram_loss"))
        relation_loss = finite_number(
            row.get("gram_global_relation_loss"),
            f"{spec.name} row {index} gram_global_relation_loss",
        )
        require(relation_loss > 0, f"{spec.name} row {index} relation loss is not positive")
        relation_losses.append(relation_loss)
        exact_number(
            row.get("gram_loss_weight"),
            spec.patch_weight,
            f"{spec.name} row {index} gram_loss_weight",
        )
        exact_number(
            row.get("gram_global_relation_loss_weight"),
            spec.relation_weight,
            f"{spec.name} row {index} gram_global_relation_loss_weight",
        )
        exact_number(
            row.get("gram_global_relation_batch"),
            64.0,
            f"{spec.name} row {index} gram_global_relation_batch",
        )
        if "gram_global_relation_mask_matched" in row:
            present_mask_flags += 1
            expected_flag = 1.0 if spec.observation_protocol == "mask_matched_student" else 0.0
            exact_number(
                row["gram_global_relation_mask_matched"],
                expected_flag,
                f"{spec.name} row {index} gram_global_relation_mask_matched",
            )

    if spec.required_mask_flag is not None:
        require(
            present_mask_flags == EXPECTED_UPDATE_COUNT,
            f"{spec.name} has mask flag on {present_mask_flags}/{EXPECTED_UPDATE_COUNT} rows",
        )
    else:
        require(
            present_mask_flags in (0, EXPECTED_UPDATE_COUNT),
            f"{spec.name} has a partially recorded observation flag",
        )
    return {
        "patch_loss": summarize(patch_losses),
        "relation_loss": summarize(relation_losses),
        "global_relation_batch": 64,
        "patch_weight": spec.patch_weight,
        "relation_weight": spec.relation_weight,
        "mask_flag_rows": present_mask_flags,
    }


def validate_candidate_config(config: dict[str, Any], spec: CandidateSpec) -> dict[str, Any]:
    require(nested(config, "gram.use_loss") is True, f"{spec.name} gram.use_loss must be true")
    exact_number(nested(config, "gram.loss_weight"), spec.patch_weight, f"{spec.name} config patch weight")
    exact_number(
        nested(config, "gram.global_relation_loss_weight"),
        spec.relation_weight,
        f"{spec.name} config relation weight",
    )
    missing = object()
    configured = nested(config, "gram.global_relation_observation_protocol", missing=missing)
    if configured is missing:
        require(
            spec.allow_missing_protocol,
            f"{spec.name} config is missing gram.global_relation_observation_protocol",
        )
        effective = "clean_teacher"
        source = "implicit_historical_default_clean_teacher"
    else:
        require(isinstance(configured, str), f"{spec.name} observation protocol must be a string")
        effective = configured
        source = "explicit_config"
    require(
        effective == spec.observation_protocol,
        f"{spec.name} observation protocol={effective!r}, expected {spec.observation_protocol!r}",
    )
    return {"effective": effective, "source": source, "configured_value": None if configured is missing else configured}


def audit_training(
    control_run: Path,
    official_run: Path,
    candidates: list[CandidateSpec],
) -> dict[str, Any]:
    paths = [control_run.resolve(), official_run.resolve(), *(spec.run.resolve() for spec in candidates)]
    require(len(set(paths)) == len(paths), "training run paths must be distinct")

    control_rows = load_training_rows(control_run, "control")
    control_losses = validate_loss_fields(control_rows, "control")
    control_digests = []
    for index, row in enumerate(control_rows):
        digest = row.get("batch_sample_key_digest")
        require(isinstance(digest, str) and digest, f"control row {index} has no sample digest")
        control_digests.append(digest)

    configs = {
        "control": read_yaml(run_config(control_run)),
        "official_a7807": read_yaml(run_config(official_run)),
        **{spec.name: read_yaml(run_config(spec.run)) for spec in candidates},
    }
    matched_config: dict[str, dict[str, Any]] = {}
    for dotted_path in MATCHED_CONFIG_PATHS:
        values = {name: nested(config, dotted_path) for name, config in configs.items()}
        reference = values["control"]
        require(
            all(value == reference for value in values.values()),
            f"matched config differs at {dotted_path}: {values}",
        )
        matched_config[dotted_path] = values
    require(nested(configs["control"], "train.max_updates") == 20496, "train.max_updates must be 20496")

    candidate_results: dict[str, Any] = {}
    for spec in candidates:
        rows = load_training_rows(spec.run, spec.name)
        loss_summary = validate_loss_fields(rows, spec.name)
        digests = []
        for index, row in enumerate(rows):
            digest = row.get("batch_sample_key_digest")
            require(isinstance(digest, str) and digest, f"{spec.name} row {index} has no sample digest")
            digests.append(digest)
        mismatches = [
            EXPECTED_UPDATES[index]
            for index, (control, candidate) in enumerate(zip(control_digests, digests, strict=True))
            if control != candidate
        ]
        require(not mismatches, f"{spec.name} sample digest mismatch at {mismatches[:8]}")
        candidate_results[spec.name] = {
            "status": "VALID_MATCHED_488_STEP_SCREEN",
            "updates": EXPECTED_UPDATE_COUNT,
            "optimizer_update_range": [EXPECTED_UPDATES[0], EXPECTED_UPDATES[-1]],
            "sample_digest_matches_control": EXPECTED_UPDATE_COUNT,
            "losses": loss_summary,
            "gram": validate_candidate_rows(rows, spec),
            "observation_protocol": validate_candidate_config(configs[spec.name], spec),
            "inputs": {
                "metrics": file_record(spec.run.resolve() / "raw_loss_metrics.jsonl"),
                "config": file_record(run_config(spec.run)),
                "endpoint": file_record(run_checkpoint(spec.run, 20495)),
            },
        }

    return {
        "status": "VALID_MATCHED_THREE_ARM_488_STEP_SCREENS",
        "control": {
            "updates": EXPECTED_UPDATE_COUNT,
            "optimizer_update_range": [EXPECTED_UPDATES[0], EXPECTED_UPDATES[-1]],
            "losses": control_losses,
            "inputs": {
                "metrics": file_record(control_run.resolve() / "raw_loss_metrics.jsonl"),
                "config": file_record(run_config(control_run)),
                "endpoint": file_record(run_checkpoint(control_run, 20495)),
            },
        },
        "candidates": candidate_results,
        "critical_matched_config": matched_config,
    }


def validate_diagnostic_input(
    record: dict[str, Any], checkpoint: Path, config: Path, context: str
) -> dict[str, Any]:
    require(isinstance(record, dict), f"{context} relation input is not a mapping")
    checkpoint_record = file_record(checkpoint)
    config_record = file_record(config)
    require(
        Path(str(record.get("checkpoint", ""))).resolve() == checkpoint.resolve(),
        f"{context} relation checkpoint path mismatch",
    )
    require(record.get("checkpoint_sha256") == checkpoint_record["sha256"], f"{context} checkpoint hash mismatch")
    require(
        Path(str(record.get("config", ""))).resolve() == config.resolve(),
        f"{context} relation config path mismatch",
    )
    require(record.get("config_sha256") == config_record["sha256"], f"{context} config hash mismatch")
    return {"checkpoint": checkpoint_record, "config": config_record}


def audit_relation(
    path: Path,
    anchor_run: Path,
    control_run: Path,
    official_run: Path,
    candidates: list[CandidateSpec],
) -> dict[str, Any]:
    payload = read_json(path)
    require(isinstance(payload, dict), "relation JSON must be a mapping")
    require(payload.get("status") == "VALID_COMPLETE_LABEL_FREE_RELATION_DIAGNOSTIC", "invalid relation status")
    require(payload.get("diagnostic") == "frozen_anchor_clean_two_view_cls_relation_drift_v1", "wrong relation diagnostic")
    require(payload.get("dataset") == RELATION_DATASET, "relation diagnostic used the wrong dataset")
    require(payload.get("seed") == 20260912, "relation seed must be 20260912")
    require(payload.get("n_images") == 128 and payload.get("n_unique_keys") == 128, "relation diagnostic must contain 128 unique images")
    require(payload.get("crop_size") == 512, "relation crop size must be 512")
    require(payload.get("crop_area_range") == [0.32, 1.0], "relation crop area range mismatch")
    require(payload.get("views") == 2 and payload.get("anchor") == "anchor", "relation view/anchor protocol mismatch")

    summaries = payload.get("summaries")
    inputs = payload.get("inputs")
    require(isinstance(summaries, dict) and isinstance(inputs, dict), "relation summaries/inputs missing")
    expected = {
        "anchor": (run_checkpoint(anchor_run, 20007), run_config(anchor_run)),
        "control": (run_checkpoint(control_run, 20495), run_config(control_run)),
        "official_a7807": (run_checkpoint(official_run, 20495), run_config(official_run)),
        **{
            spec.name: (run_checkpoint(spec.run, 20495), run_config(spec.run))
            for spec in candidates
        },
    }
    validated_inputs = {}
    values = {}
    for name, (checkpoint, config) in expected.items():
        require(name in summaries, f"relation summary missing {name}")
        require(name in inputs, f"relation input missing {name}")
        summary = summaries[name]
        require(isinstance(summary, dict), f"relation summary {name} is not a mapping")
        value = finite_number(summary.get(RELATION_METRIC), f"relation {name} {RELATION_METRIC}")
        require(value >= 0, f"relation drift for {name} is negative")
        values[name] = value
        validated_inputs[name] = validate_diagnostic_input(inputs[name], checkpoint, config, name)
    require(values["anchor"] == 0.0, "anchor self-drift must be exactly zero")
    require(values["control"] > 0 and values["official_a7807"] > 0, "relation denominators must be positive")

    official = values["official_a7807"]
    control = values["control"]
    gates: dict[str, Any] = {}
    for name in ("strong_dual", "mask_matched_dual"):
        candidate = values[name]
        gates[name] = {
            "criteria": {
                "candidate_over_official_a7807_at_most": 0.8,
                "candidate_over_control_at_most": 1.1,
            },
            "observed": {
                "drift": candidate,
                "candidate_over_official_a7807": candidate / official,
                "candidate_over_control": candidate / control,
            },
            "pass": candidate <= 0.8 * official and candidate <= 1.1 * control,
        }
    relation_only = values["relation_only"]
    gates["relation_only"] = {
        "criteria": {"candidate_drift_strictly_below_control": True},
        "observed": {
            "drift": relation_only,
            "control_drift": control,
            "candidate_over_control": relation_only / control,
        },
        "pass": relation_only < control,
    }
    return {
        "status": "VALID_COMPLETE_FIVE_ARM_LABEL_FREE_RELATION_AUDIT",
        "metric": RELATION_METRIC,
        "values": values,
        "gates": gates,
        "inputs": validated_inputs,
        "source": file_record(path.resolve()),
    }


def validate_spatial_artifact(
    summary_path: Path,
    records_path: Path,
    checkpoint: Path,
    config: Path,
    name: str,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    records = read_json(records_path)
    require(isinstance(summary, dict) and isinstance(records, dict), f"{name} spatial files must be mappings")
    diagnostic = "frozen_nested_local_global_spatial_signal_v1"
    require(summary.get("diagnostic") == diagnostic and records.get("diagnostic") == diagnostic, f"{name} spatial diagnostic mismatch")
    require(summary.get("dataset") == FIXED_SPATIAL_DATASET, f"{name} did not use the fixed 1TB spatial stream")
    require(summary.get("seed") == 20260911 and records.get("seed") == 20260911, f"{name} spatial seed mismatch")
    require(summary.get("n_images") == 128 and summary.get("n_unique_keys") == 128, f"{name} spatial summary must contain 128 unique images")
    require(summary.get("patch_size") == 16 and summary.get("global_grid") == 16 and summary.get("local_grid") == 7, f"{name} spatial grid mismatch")
    require(summary.get("bootstrap_reps") == 2000, f"{name} spatial bootstrap count mismatch")
    crop_spec = summary.get("crop_spec")
    require(isinstance(crop_spec, dict), f"{name} spatial crop spec missing")
    expected_crop = {
        "crop_seed": 20261012,
        "global_area_min": 0.5,
        "global_size": 256,
        "local_area_max": 0.32,
        "local_area_min": 0.05,
        "local_size": 112,
        "photometric_policy": "bio_safe",
    }
    for field, expected_value in expected_crop.items():
        require(crop_spec.get(field) == expected_value, f"{name} spatial crop field {field} mismatch")

    expected_checkpoint = checkpoint.resolve()
    expected_config = config.resolve()
    require(Path(str(summary.get("checkpoint", ""))).resolve() == expected_checkpoint, f"{name} spatial summary checkpoint mismatch")
    require(Path(str(records.get("checkpoint", ""))).resolve() == expected_checkpoint, f"{name} spatial records checkpoint mismatch")
    require(Path(str(summary.get("train_config", ""))).resolve() == expected_config, f"{name} spatial config mismatch")

    keys = records.get("keys")
    areas = records.get("local_area")
    require(isinstance(keys, list) and len(keys) == 128, f"{name} spatial records need 128 keys")
    require(all(isinstance(key, str) and key for key in keys), f"{name} has an invalid spatial key")
    require(len(set(keys)) == 128, f"{name} spatial keys are not unique")
    require(isinstance(areas, list) and len(areas) == 128, f"{name} spatial records need 128 areas")
    areas_float = [finite_number(value, f"{name} local_area") for value in areas]
    layers = records.get("layers")
    require(isinstance(layers, dict), f"{name} spatial layers missing")
    block24 = layers.get("block_24")
    require(isinstance(block24, dict), f"{name} block_24 records missing")
    values = block24.get("true_minus_shifted")
    require(isinstance(values, list) and len(values) == 128, f"{name} needs 128 block_24 margins")
    margins = [finite_number(value, f"{name} block_24 margin") for value in values]
    return {
        "summary": summary,
        "keys": keys,
        "areas": areas_float,
        "margins": margins,
        "inputs": {
            "summary": file_record(summary_path.resolve()),
            "records": file_record(records_path.resolve()),
        },
    }


def audit_spatial(
    reference_dir: Path,
    strong_dir: Path,
    mask_dir: Path,
    summary_name: str,
    records_name: str,
    control_run: Path,
    official_run: Path,
    candidates: list[CandidateSpec],
) -> dict[str, Any]:
    candidate_by_name = {spec.name: spec for spec in candidates}
    artifacts = {
        "control": validate_spatial_artifact(
            reference_dir / "control.json",
            reference_dir / "control_records.json",
            run_checkpoint(control_run, 20495),
            run_config(control_run),
            "control",
        ),
        "official_a7807": validate_spatial_artifact(
            reference_dir / "official.json",
            reference_dir / "official_records.json",
            run_checkpoint(official_run, 20495),
            run_config(official_run),
            "official_a7807",
        ),
        "strong_dual": validate_spatial_artifact(
            strong_dir / summary_name,
            strong_dir / records_name,
            run_checkpoint(candidate_by_name["strong_dual"].run, 20495),
            run_config(candidate_by_name["strong_dual"].run),
            "strong_dual",
        ),
        "mask_matched_dual": validate_spatial_artifact(
            mask_dir / summary_name,
            mask_dir / records_name,
            run_checkpoint(candidate_by_name["mask_matched_dual"].run, 20495),
            run_config(candidate_by_name["mask_matched_dual"].run),
            "mask_matched_dual",
        ),
    }
    reference = artifacts["control"]
    reference_crop_spec = reference["summary"]["crop_spec"]
    for name, artifact in artifacts.items():
        require(artifact["keys"] == reference["keys"], f"{name} spatial keys differ from fixed reference")
        require(artifact["areas"] == reference["areas"], f"{name} spatial areas differ from fixed reference")
        require(artifact["summary"]["crop_spec"] == reference_crop_spec, f"{name} full spatial crop spec differs from reference")

    control_mean = statistics.fmean(artifacts["control"]["margins"])
    official_mean = statistics.fmean(artifacts["official_a7807"]["margins"])
    official_gain = official_mean - control_mean
    require(official_gain > 0, "official_a7807 spatial gain over control must be positive")
    gates = {}
    for name in ("strong_dual", "mask_matched_dual"):
        candidate_mean = statistics.fmean(artifacts[name]["margins"])
        candidate_gain = candidate_mean - control_mean
        retained = candidate_gain / official_gain
        gates[name] = {
            "criteria": {
                "candidate_minus_control_at_least": 0.0,
                "retained_official_gain_at_least": 0.5,
            },
            "observed": {
                "control_mean": control_mean,
                "official_a7807_mean": official_mean,
                "candidate_mean": candidate_mean,
                "official_a7807_minus_control": official_gain,
                "candidate_minus_control": candidate_gain,
                "retained_official_gain": retained,
            },
            "pass": candidate_gain >= 0 and retained >= 0.5,
        }
    return {
        "status": "VALID_EXACTLY_PAIRED_FIXED_1TB_SPATIAL_AUDIT",
        "n_images": 128,
        "n_unique_keys": 128,
        "ordered_keys_exact_match": True,
        "local_areas_exact_match": True,
        "gates": gates,
        "inputs": {name: artifact["inputs"] for name, artifact in artifacts.items()},
    }


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def render_readme(payload: dict[str, Any]) -> str:
    relation = payload["relation"]["gates"]
    spatial = payload["spatial"]["gates"]
    decisions = payload["decisions"]
    training = payload["training"]["candidates"]
    lines = [
        "# HS6-L5 dual-anchor relation follow-ups",
        "",
        f"Status: `{payload['status']}`.",
        "",
        "This report uses no labels and defines no aggregate score. Each locked gate is shown separately.",
        "",
        "## Training integrity",
        "",
        "| Arm | Updates | Digest matches | Patch / relation weight | Observation |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ("strong_dual", "mask_matched_dual", "relation_only"):
        row = training[name]
        gram = row["gram"]
        observation = row["observation_protocol"]
        lines.append(
            f"| `{name}` | {row['updates']} | {row['sample_digest_matches_control']}/488 | "
            f"{gram['patch_weight']:.1f} / {gram['relation_weight']:.1f} | "
            f"`{observation['effective']}` ({observation['source']}) |"
        )
    lines += [
        "",
        "## Label-free gates",
        "",
        "| Arm | Relation result | Spatial result | Decision |",
        "|---|---|---|---|",
    ]
    for name in ("strong_dual", "mask_matched_dual"):
        rel = relation[name]
        spa = spatial[name]
        lines.append(
            f"| `{name}` | {'PASS' if rel['pass'] else 'FAIL'} "
            f"(control x{rel['observed']['candidate_over_control']:.4f}, "
            f"official x{rel['observed']['candidate_over_official_a7807']:.4f}) | "
            f"{'PASS' if spa['pass'] else 'FAIL'} "
            f"(retained {spa['observed']['retained_official_gain']:.4f}) | "
            f"`{'ELIGIBLE' if decisions[name]['eligible_for_longer_self_supervised_run'] else 'STOP'}` |"
        )
    rel_only = relation["relation_only"]
    lines.append(
        f"| `relation_only` | {'PASS' if rel_only['pass'] else 'FAIL'} "
        f"(control x{rel_only['observed']['candidate_over_control']:.4f}) | n/a | "
        f"`{'SUPPORTED' if decisions['relation_only']['relation_mechanism_supported'] else 'NOT_SUPPORTED'}` |"
    )
    lines += [
        "",
        "Strong and mask-matched eligibility requires both their relation and spatial gates. "
        "Relation-only is a mechanism control and is judged only against control relation drift.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    candidates = [
        CandidateSpec(
            "strong_dual",
            args.strong_run,
            patch_weight=2.0,
            relation_weight=2.0,
            observation_protocol="clean_teacher",
            allow_missing_protocol=True,
        ),
        CandidateSpec(
            "mask_matched_dual",
            args.mask_matched_run,
            patch_weight=2.0,
            relation_weight=2.0,
            observation_protocol="mask_matched_student",
            required_mask_flag=1.0,
        ),
        CandidateSpec(
            "relation_only",
            args.relation_only_run,
            patch_weight=0.0,
            relation_weight=2.0,
            observation_protocol="clean_teacher",
            required_mask_flag=0.0,
        ),
    ]
    training = audit_training(args.control_run, args.official_run, candidates)
    relation = audit_relation(
        args.relation_json,
        args.anchor_run,
        args.control_run,
        args.official_run,
        candidates,
    )
    spatial = audit_spatial(
        args.spatial_reference_dir,
        args.strong_spatial_dir,
        args.mask_matched_spatial_dir,
        args.candidate_spatial_summary_name,
        args.candidate_spatial_records_name,
        args.control_run,
        args.official_run,
        candidates,
    )
    decisions = {
        name: {
            "relation_gate_pass": relation["gates"][name]["pass"],
            "spatial_gate_pass": spatial["gates"][name]["pass"],
            "eligible_for_longer_self_supervised_run": bool(
                relation["gates"][name]["pass"] and spatial["gates"][name]["pass"]
            ),
        }
        for name in ("strong_dual", "mask_matched_dual")
    }
    decisions["relation_only"] = {
        "relation_gate_pass": relation["gates"]["relation_only"]["pass"],
        "relation_mechanism_supported": relation["gates"]["relation_only"]["pass"],
    }
    payload = {
        "status": "VALID_COMPLETE_LABEL_FREE_DUAL_ANCHOR_FOLLOWUP_AUDIT",
        "admission": "LABEL_FREE_METHOD_GATES_ONLY",
        "no_composite_score": True,
        "label_policy": "No labels or frozen labeled-probe metrics enter this report or its decisions.",
        "training": training,
        "relation": relation,
        "spatial": spatial,
        "decisions": decisions,
    }
    output_dir = args.output.resolve()
    atomic_write(output_dir / "analysis.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_write(output_dir / "README.md", render_readme(payload))
    print(
        json.dumps(
            {
                "output": str(output_dir / "analysis.json"),
                "decisions": decisions,
                "no_composite_score": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        raise SystemExit(f"AUDIT FAILED: {error}") from error
