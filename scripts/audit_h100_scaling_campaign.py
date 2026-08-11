#!/usr/bin/env python3
"""Audit every completion invariant of the H100 B/L/H+/7B scaling campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CHECKPOINTS_15 = tuple(1024 + 1025 * index for index in range(15))
CHECKPOINTS_7B = CHECKPOINTS_15[-3:]
REQUIRED_METRICS = (
    "family6_equal_mean",
    "c25_macro_f1",
    "bbbc005_r2",
    "retrieval4_map_at_5",
    "clustering4_nmi",
    "segmentation8_mdice",
    "livecell_detection_f1",
)
RESULT_LAYOUT = {
    "bio_classification": ("last_result.json", 25),
    "bio_regression": ("last_result.json", 2),
    "bio_retrieval": ("last_result.json", 4),
    "bio_detection": ("results_bio_detection.json", 1),
    "bio_segmentation": ("results.json", 8),
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


class Auditor:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, ok: bool, detail: str) -> bool:
        self.checks.append(Check(name=name, ok=bool(ok), detail=detail))
        return bool(ok)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("/data_2/suxin/runs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--ignore-campaign-marker",
        action="store_true",
        help="Omit campaign.done from the checks so the audit can gate creation of that marker.",
    )
    return parser.parse_args()


def check_finite(value: Any, source: Path, key_path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            check_finite(nested, source, f"{key_path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            check_finite(nested, source, f"{key_path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{source}: non-finite value at {key_path}: {value}")
    elif isinstance(value, str) and value.strip().lower() in {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        raise ValueError(f"{source}: non-finite string at {key_path}: {value!r}")


def load_json(path: Path) -> Any:
    with path.open() as handle:
        payload = json.load(handle)
    check_finite(payload, path)
    return payload


def checkpoint_ready(run_dir: Path, checkpoint: int) -> bool:
    root = run_dir / "ckpt" / str(checkpoint)
    return (root / "checkpoint.pth").is_file() or (root / ".metadata").is_file()


def audit_summary(auditor: Auditor, name: str, path: Path, alpha: bool) -> None:
    if not auditor.add(f"{name}.exists", path.is_file(), str(path)):
        return
    try:
        payload = load_json(path)
        values = {metric: float(payload[metric]) for metric in REQUIRED_METRICS}
        finite = all(math.isfinite(value) for value in values.values())
        if alpha:
            alpha_value = float(payload["alpha"])
            finite = finite and math.isfinite(alpha_value) and 0.0 <= alpha_value <= 1.0
        auditor.add(f"{name}.valid", finite, json.dumps(values, sort_keys=True))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        auditor.add(f"{name}.valid", False, str(error))


def result_paths(eval_root: Path, family: str, checkpoint: int) -> list[Path]:
    filename, _ = RESULT_LAYOUT[family]
    family_root = eval_root / family
    if not family_root.is_dir():
        return []
    return sorted(
        path for path in family_root.rglob(filename) if path.parent.name == str(checkpoint)
    )


def audit_full_eval(
    auditor: Auditor,
    name: str,
    eval_root: Path,
    expected_checkpoints: tuple[int, ...],
) -> None:
    status_root = eval_root / "_online_status"
    done = sorted(
        int(path.stem.removeprefix("ckpt_"))
        for path in status_root.glob("ckpt_*.done")
        if path.stem.removeprefix("ckpt_").isdigit()
    )
    failed = sorted(path.name for path in status_root.glob("ckpt_*.failed"))
    auditor.add(f"{name}.done_markers", tuple(done) == expected_checkpoints, str(done))
    auditor.add(f"{name}.failed_markers", not failed, str(failed))
    if tuple(done) != expected_checkpoints or failed:
        return

    errors: list[str] = []
    parsed = 0
    for checkpoint in expected_checkpoints:
        for family, (_, expected_count) in RESULT_LAYOUT.items():
            paths = result_paths(eval_root, family, checkpoint)
            if len(paths) != expected_count:
                errors.append(f"ckpt={checkpoint} {family} count={len(paths)} expected={expected_count}")
                continue
            for path in paths:
                try:
                    load_json(path)
                    parsed += 1
                except (ValueError, json.JSONDecodeError) as error:
                    errors.append(str(error))
    expected_results = len(expected_checkpoints) * sum(count for _, count in RESULT_LAYOUT.values())
    auditor.add(
        f"{name}.results",
        not errors and parsed == expected_results,
        f"parsed={parsed}/{expected_results}; errors={errors[:10]}",
    )


def audit_bl_case(
    auditor: Auditor,
    name: str,
    run_dir: Path,
    eval_root: Path,
    alpha_root: Path,
) -> None:
    auditor.add(
        f"{name}.final_checkpoint",
        checkpoint_ready(run_dir, CHECKPOINTS_15[-1]),
        str(run_dir / "ckpt" / str(CHECKPOINTS_15[-1])),
    )
    audit_full_eval(auditor, f"{name}.full_eval", eval_root, CHECKPOINTS_15)
    auditor.add(
        f"{name}.alpha_pipeline",
        (alpha_root / "_status" / "pipeline.done").is_file(),
        str(alpha_root / "_status" / "pipeline.done"),
    )
    audit_summary(auditor, f"{name}.raw_summary", alpha_root / "raw_sweep" / "best.json", False)
    audit_summary(auditor, f"{name}.alpha_summary", alpha_root / "full_summary" / "best.json", True)


def audit_huge_case(
    auditor: Auditor,
    campaign_root: Path,
    model: str,
    objective: str,
) -> None:
    name = f"{model}.{objective}"
    objective_root = campaign_root / f"{model}_{objective}"
    status_marker = campaign_root / "_status" / f"{model}_{objective}.done"
    auditor.add(f"{name}.status", status_marker.is_file(), str(status_marker))
    for ranking in ("lr", "warmup"):
        path = objective_root / "ranking" / ranking / "best.json"
        valid = False
        detail = str(path)
        if path.is_file():
            try:
                payload = load_json(path)
                valid = "label" in payload
                detail = json.dumps(payload, sort_keys=True)
            except (ValueError, json.JSONDecodeError) as error:
                detail = str(error)
        auditor.add(f"{name}.ranking_{ranking}", valid, detail)

    final_dirs = sorted(path for path in objective_root.glob("final_lr*_wu*") if path.is_dir())
    if not auditor.add(f"{name}.final_run_unique", len(final_dirs) == 1, str(final_dirs)):
        return
    final_dir = final_dirs[0]
    expected_checkpoints = CHECKPOINTS_15 if model == "hplus" else CHECKPOINTS_7B
    auditor.add(
        f"{name}.final_checkpoint",
        checkpoint_ready(final_dir, CHECKPOINTS_15[-1]),
        str(final_dir / "ckpt" / str(CHECKPOINTS_15[-1])),
    )
    audit_full_eval(auditor, f"{name}.full_eval", final_dir / "eval_full", expected_checkpoints)
    alpha_root = objective_root / "alpha_tune"
    auditor.add(
        f"{name}.alpha_pipeline",
        (alpha_root / "_status" / "pipeline.done").is_file(),
        str(alpha_root / "_status" / "pipeline.done"),
    )
    audit_summary(auditor, f"{name}.raw_summary", alpha_root / "raw_sweep" / "best.json", False)
    audit_summary(auditor, f"{name}.alpha_summary", alpha_root / "full_summary" / "best.json", True)


def audit_scaling(auditor: Auditor, campaign_root: Path, ignore_campaign_marker: bool) -> None:
    status_root = campaign_root / "_status"
    markers = ("scaling_law.done",) if ignore_campaign_marker else ("scaling_law.done", "campaign.done")
    for marker in markers:
        path = status_root / marker
        auditor.add(f"scaling.{marker}", path.is_file(), str(path))
    output = campaign_root / "scaling_law_final"
    manifest_path = output / "manifest.json"
    manifest_valid = False
    detail = str(manifest_path)
    if manifest_path.is_file():
        try:
            manifest = load_json(manifest_path)
            manifest_valid = (
                manifest.get("complete") is True
                and manifest.get("expected_rows") == 16
                and manifest.get("available_rows") == 16
                and not manifest.get("missing")
            )
            detail = json.dumps(manifest, sort_keys=True)
        except (ValueError, json.JSONDecodeError) as error:
            detail = str(error)
    auditor.add("scaling.manifest", manifest_valid, detail)

    csv_path = output / "h100_sigreg_scaling_values.csv"
    row_count = -1
    if csv_path.is_file():
        with csv_path.open(newline="") as handle:
            row_count = len(list(csv.DictReader(handle)))
    auditor.add("scaling.csv", row_count == 16, f"rows={row_count}; path={csv_path}")
    for stem in ("h100_sigreg_scaling_overall", "h100_sigreg_scaling_tasks"):
        for suffix in ("png", "svg", "pdf"):
            path = output / f"{stem}.{suffix}"
            auditor.add(f"scaling.{stem}.{suffix}", path.is_file() and path.stat().st_size > 0, str(path))


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    auditor = Auditor()
    cases = {
        "B.sigreg": (
            root / "B_rgb_robust_biosafe_sigreg005_gb1024_lr1e4_wu2_e15_seed0_20260722_r1",
            "full_taskwise_b64_bf16_auto_tta8_20260725",
            root / "Binterp_official_sigreg_auto_best_alpha_tune_20260725",
        ),
        "B.nosigreg": (
            root / "B_rgb_robust_biosafe_nosigreg_gb1024_lr1e4_wu2_e15_seed0_20260724_r1",
            "full_taskwise_online_b64_bf16_auto_tta8_20260724",
            root / "Binterp_official_nosigreg_bestck11274_alpha_sweep_20260725",
        ),
        "L.sigreg": (
            root / "L_rgb_robust_biosafe_sigreg005_gb1024_lr5e5_wu2_e15_seed0_cu124_20260723_r1",
            "full_taskwise_b32_bf16_auto_tta8_20260725",
            root / "Linterp_official_sigreg_auto_best_alpha_tune_20260725",
        ),
        "L.nosigreg": (
            root / "L_rgb_robust_biosafe_nosigreg_gb1024_lr5e5_wu2_e15_seed0_20260725_r1",
            "full_taskwise_online_b8_bf16_auto_tta8_20260725",
            root / "Linterp_official_nosigreg_auto_best_alpha_tune_20260725",
        ),
    }
    for name, (run_dir, eval_name, alpha_root) in cases.items():
        audit_bl_case(auditor, name, run_dir, run_dir / "eval" / eval_name, alpha_root)

    campaign_root = root / "h100_hplus_7b_sigreg_ab_tuning_20260725"
    for model in ("hplus", "7b"):
        for objective in ("sigreg", "nosigreg"):
            audit_huge_case(auditor, campaign_root, model, objective)
    audit_scaling(auditor, campaign_root, args.ignore_campaign_marker)

    failed = [check for check in auditor.checks if not check.ok]
    payload = {
        "complete": not failed,
        "checks_total": len(auditor.checks),
        "checks_passed": len(auditor.checks) - len(failed),
        "checks_failed": len(failed),
        "checks": [asdict(check) for check in auditor.checks],
    }
    if args.output:
        write_atomic(args.output, payload)
    print(json.dumps({key: payload[key] for key in payload if key != "checks"}, sort_keys=True))
    if failed and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
