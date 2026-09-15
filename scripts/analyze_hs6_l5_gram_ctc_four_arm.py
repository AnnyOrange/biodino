#!/usr/bin/env python3
"""Fail-closed comparison of the four matched HS6-L5 Gram CTC observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN_ROOT = ROOT / "outputs/02_eval_runs/ctc_native_2d_hs6_l5_gram_branches_observation_20260912"
DEFAULT_OUTPUT = ROOT / "outputs/03_comparisons/hs6_l5_gram_ctc_four_arm_observation_20260912"

ARMS = ("control", "anchor7807", "anchor17079", "dual")
EXPECTED_MODELS = {
    "control": "hs6_l5_control_ck20495",
    "anchor7807": "hs6_l5_anchor7807_ck20495",
    "anchor17079": "hs6_l5_anchor17079_ck20495",
    "dual": "hs6_l5_dual_anchor_ck20495",
}
EXPECTED_SSL_INTERVENTIONS = {
    "control": {"patch_gram": False, "global_relation_gram": False},
    "anchor7807": {
        "patch_gram": True,
        "patch_anchor_checkpoint": 7807,
        "patch_weight": 2.0,
        "global_relation_gram": False,
    },
    "anchor17079": {
        "patch_gram": True,
        "patch_anchor_checkpoint": 17079,
        "patch_weight": 2.0,
        "global_relation_gram": False,
    },
    "dual": {
        "patch_gram": True,
        "patch_anchor_checkpoint": 7807,
        "patch_weight": 2.0,
        "global_relation_gram": True,
        "global_relation_anchor_checkpoint": 20007,
        "global_relation_weight": 0.5,
    },
}
COMPARISONS = (
    ("anchor7807_minus_control", "control", "anchor7807"),
    ("anchor17079_minus_control", "control", "anchor17079"),
    ("dual_minus_control", "control", "dual"),
    ("dual_minus_anchor7807", "anchor7807", "dual"),
)
PRIMARY_METRICS = ("TRA", "SEG")
SECONDARY_METRICS = ("DET", "mean_foreground_dice", "AP", "AP50", "AP75")
METRICS = PRIMARY_METRICS + SECONDARY_METRICS
EXPECTED_PROTOCOL = "ctc-native-2d-domain-heldout-observational-v1"
EXPECTED_ADMISSION = "OBSERVATIONAL_NATIVE_2D"
TIE_TOLERANCE = 1.0e-12


class AuditError(RuntimeError):
    """Raised when an input cannot support the locked four-arm comparison."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read valid JSON: {path}: {error}") from error
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise AuditError(f"cannot hash input: {path}: {error}") from error
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _finite_float(value: Any, context: str) -> float:
    _require(not isinstance(value, bool), f"{context} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AuditError(f"{context} must be a finite number") from error
    _require(math.isfinite(number), f"{context} must be finite")
    return number


def _mean(values: list[float]) -> float:
    _require(bool(values), "cannot compute a mean over an empty sequence")
    return math.fsum(values) / len(values)


def _assert_metric_map(stored: Any, computed: dict[str, float], context: str) -> None:
    _require(isinstance(stored, dict), f"{context} must be an object")
    for metric, expected in computed.items():
        observed = _finite_float(stored.get(metric), f"{context}.{metric}")
        _require(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12),
            f"{context}.{metric} is not the direct domain macro: {observed} != {expected}",
        )


def _domain_metric(row: dict[str, Any], metric: str, context: str) -> float:
    container = "ctc_metrics" if metric in {"TRA", "SEG", "DET"} else "extra_segmentation_metrics"
    values = row.get(container)
    _require(isinstance(values, dict), f"{context}.{container} must be an object")
    return _finite_float(values.get(metric), f"{context}.{container}.{metric}")


def _expected_fold_specs(manifest: dict[str, Any], arm: str) -> dict[int, dict[str, Any]]:
    folds = manifest.get("folds")
    _require(isinstance(folds, list) and len(folds) == 5, f"{arm}: manifest must contain 5 folds")
    by_fold: dict[int, dict[str, Any]] = {}
    held_out: list[str] = []
    for spec in folds:
        _require(isinstance(spec, dict), f"{arm}: invalid manifest fold record")
        try:
            fold = int(spec["fold"])
        except (KeyError, TypeError, ValueError) as error:
            raise AuditError(f"{arm}: invalid manifest fold id") from error
        _require(fold not in by_fold, f"{arm}: duplicate manifest fold {fold}")
        test_domains = spec.get("test_domains")
        _require(
            isinstance(test_domains, list)
            and bool(test_domains)
            and all(isinstance(domain, str) and domain for domain in test_domains),
            f"{arm}: fold {fold} has invalid test domains",
        )
        _require(len(test_domains) == len(set(test_domains)), f"{arm}: duplicate domain in fold {fold}")
        held_out.extend(test_domains)
        by_fold[fold] = spec
    _require(set(by_fold) == set(range(5)), f"{arm}: fold ids must be exactly 0..4")
    _require(len(held_out) == 10, f"{arm}: manifest must contain exactly 10 held-out domains")
    _require(len(set(held_out)) == 10, f"{arm}: each domain must be held out exactly once")
    return by_fold


def _validate_domain_row(
    row: Any,
    *,
    arm: str,
    fold: int,
    manifest_sha: str,
    protocol_id: str,
) -> dict[str, Any]:
    context = f"{arm}: fold {fold} domain row"
    _require(isinstance(row, dict), f"{context} must be an object")
    _require(row.get("status") == "VALID_COMPLETE", f"{context} is not VALID_COMPLETE")
    _require(row.get("admission") == EXPECTED_ADMISSION, f"{context} has wrong admission")
    _require(row.get("protocol_id") == protocol_id, f"{context} has wrong protocol")
    _require(
        row.get("campaign_manifest_sha256") == manifest_sha,
        f"{context} is not bound to its manifest",
    )
    _require(row.get("fold") == fold, f"{context} has the wrong fold id")
    domain = row.get("domain")
    _require(isinstance(domain, str) and domain, f"{context} has no domain")
    ctc = row.get("ctc_metrics")
    _require(isinstance(ctc, dict), f"{context}.ctc_metrics must be an object")
    _require(ctc.get("Valid") == 1, f"{context} has CTC Valid != 1")
    for metric in METRICS:
        _domain_metric(row, metric, context)
    return row


def _validate_arm(campaign_root: Path, arm: str) -> dict[str, Any]:
    arm_root = campaign_root / arm
    manifest_path = arm_root / "campaign_manifest.json"
    validation_path = arm_root / "validation_report.json"
    manifest = _read_json(manifest_path)
    validation = _read_json(validation_path)
    manifest_sha = _file_sha256(manifest_path)

    _require(manifest.get("status") == "LOCKED_BEFORE_RUN", f"{arm}: manifest is not locked")
    _require(manifest.get("branch_arm") == arm, f"{arm}: branch_arm mismatch")
    _require(manifest.get("protocol_id") == EXPECTED_PROTOCOL, f"{arm}: unexpected protocol")
    _require(manifest.get("admission") == EXPECTED_ADMISSION, f"{arm}: unexpected admission")

    models = manifest.get("models")
    _require(isinstance(models, list) and len(models) == 1, f"{arm}: expected exactly one model")
    candidate = models[0]
    _require(isinstance(candidate, dict), f"{arm}: invalid model record")
    _require(candidate.get("model") == EXPECTED_MODELS[arm], f"{arm}: unexpected model name")
    _require(candidate.get("checkpoint_step") == 20495, f"{arm}: endpoint must be ck20495")
    checkpoint = candidate.get("checkpoint")
    _require(isinstance(checkpoint, dict), f"{arm}: missing checkpoint record")
    checkpoint_sha = checkpoint.get("sha256")
    _require(_is_sha256(checkpoint_sha), f"{arm}: invalid checkpoint SHA256")

    data_manifest = manifest.get("data_manifest")
    _require(isinstance(data_manifest, dict), f"{arm}: missing data manifest record")
    _require(
        _is_sha256(data_manifest.get("sha256")),
        f"{arm}: invalid data manifest SHA256",
    )
    _require(
        _is_sha256(manifest.get("source_split_manifest_sha256")),
        f"{arm}: invalid source split SHA256",
    )

    ssl = manifest.get("ssl_training")
    _require(isinstance(ssl, dict), f"{arm}: missing SSL training declaration")
    expected_ssl = {
        "student_start_checkpoint": 20007,
        "endpoint_checkpoint": 20495,
        "matched_updates": 488,
        "effective_global_batch": 64,
        "labels_used": False,
        "intervention": EXPECTED_SSL_INTERVENTIONS[arm],
    }
    _require(ssl == expected_ssl, f"{arm}: unexpected SSL training declaration")

    _require(validation.get("status") == "VALID_COMPLETE", f"{arm}: validator is not VALID_COMPLETE")
    _require(validation.get("admission") == EXPECTED_ADMISSION, f"{arm}: validator admission mismatch")
    _require(validation.get("protocol_id") == EXPECTED_PROTOCOL, f"{arm}: validator protocol mismatch")
    _require(
        validation.get("campaign_manifest_sha256") == manifest_sha,
        f"{arm}: validator is not bound to its manifest",
    )
    _require(validation.get("expected_models") == 1, f"{arm}: validator expected_models != 1")
    _require(validation.get("valid_models") == 1, f"{arm}: validator valid_models != 1")
    _require(validation.get("expected_folds_per_model") == 5, f"{arm}: validator fold count != 5")
    _require(validation.get("expected_domains_per_model") == 10, f"{arm}: validator domain count != 10")
    _require(validation.get("errors") == [], f"{arm}: validator contains errors")

    fold_specs = _expected_fold_specs(manifest, arm)
    result_path = arm_root / "models" / str(candidate["model"]) / "results.json"
    result = _read_json(result_path)
    _require(result.get("status") == "VALID_COMPLETE", f"{arm}: result is not VALID_COMPLETE")
    _require(result.get("admission") == EXPECTED_ADMISSION, f"{arm}: result admission mismatch")
    _require(result.get("protocol_id") == EXPECTED_PROTOCOL, f"{arm}: result protocol mismatch")
    _require(
        result.get("campaign_manifest_sha256") == manifest_sha,
        f"{arm}: result is not bound to its manifest",
    )
    _require(result.get("model") == candidate["model"], f"{arm}: result model mismatch")
    _require(result.get("checkpoint_step") == 20495, f"{arm}: result checkpoint mismatch")
    _require(result.get("checkpoint_sha256") == checkpoint_sha, f"{arm}: result checkpoint hash mismatch")

    top_rows = result.get("domain_rows")
    _require(isinstance(top_rows, list) and len(top_rows) == 10, f"{arm}: expected 10 domain rows")
    top_by_domain: dict[str, dict[str, Any]] = {}
    for raw_row in top_rows:
        _require(isinstance(raw_row, dict), f"{arm}: invalid top-level domain row")
        try:
            fold = int(raw_row["fold"])
        except (KeyError, TypeError, ValueError) as error:
            raise AuditError(f"{arm}: domain row has invalid fold") from error
        row = _validate_domain_row(
            raw_row,
            arm=arm,
            fold=fold,
            manifest_sha=manifest_sha,
            protocol_id=EXPECTED_PROTOCOL,
        )
        domain = str(row["domain"])
        _require(domain not in top_by_domain, f"{arm}: duplicate top-level domain {domain}")
        _require(
            fold in fold_specs and domain in fold_specs[fold]["test_domains"],
            f"{arm}: {domain} is assigned to the wrong fold",
        )
        top_by_domain[domain] = row

    expected_domains = {domain for spec in fold_specs.values() for domain in spec["test_domains"]}
    _require(set(top_by_domain) == expected_domains, f"{arm}: held-out domain set mismatch")

    result_folds = result.get("folds")
    _require(isinstance(result_folds, list) and len(result_folds) == 5, f"{arm}: expected 5 result folds")
    observed_folds: set[int] = set()
    for fold_row in result_folds:
        _require(isinstance(fold_row, dict), f"{arm}: invalid result fold")
        try:
            fold = int(fold_row["fold"])
        except (KeyError, TypeError, ValueError) as error:
            raise AuditError(f"{arm}: result fold has invalid id") from error
        _require(fold in fold_specs and fold not in observed_folds, f"{arm}: duplicate or unknown fold {fold}")
        observed_folds.add(fold)
        context = f"{arm}: fold {fold}"
        _require(fold_row.get("status") == "VALID_COMPLETE", f"{context} is incomplete")
        _require(fold_row.get("protocol_id") == EXPECTED_PROTOCOL, f"{context} protocol mismatch")
        _require(fold_row.get("campaign_manifest_sha256") == manifest_sha, f"{context} manifest mismatch")
        _require(fold_row.get("model") == candidate["model"], f"{context} model mismatch")
        _require(
            fold_row.get("train_domains") == fold_specs[fold].get("train_domains"), f"{context} train domains mismatch"
        )
        _require(
            fold_row.get("test_domains") == fold_specs[fold].get("test_domains"), f"{context} test domains mismatch"
        )
        fold_domain_rows = fold_row.get("domain_rows")
        _require(isinstance(fold_domain_rows, list), f"{context} domain_rows must be a list")
        fold_by_domain: dict[str, dict[str, Any]] = {}
        for raw_row in fold_domain_rows:
            row = _validate_domain_row(
                raw_row,
                arm=arm,
                fold=fold,
                manifest_sha=manifest_sha,
                protocol_id=EXPECTED_PROTOCOL,
            )
            domain = str(row["domain"])
            _require(domain not in fold_by_domain, f"{context} has duplicate domain {domain}")
            fold_by_domain[domain] = row
        _require(
            set(fold_by_domain) == set(fold_specs[fold]["test_domains"]),
            f"{context} domain set mismatch",
        )
        for domain, row in fold_by_domain.items():
            _require(row == top_by_domain[domain], f"{context} top-level row differs for {domain}")
        fold_macro = {
            metric: _mean(
                [_domain_metric(row, metric, f"{context} {domain}") for domain, row in fold_by_domain.items()]
            )
            for metric in METRICS
        }
        _assert_metric_map(fold_row.get("macro"), fold_macro, f"{context}.macro")
    _require(observed_folds == set(range(5)), f"{arm}: result fold ids must be exactly 0..4")

    domain_values = {
        domain: {metric: _domain_metric(row, metric, f"{arm}: {domain}") for metric in METRICS}
        for domain, row in sorted(top_by_domain.items())
    }
    macro = {metric: _mean([values[metric] for values in domain_values.values()]) for metric in METRICS}
    _assert_metric_map(result.get("macro"), macro, f"{arm}: result.macro")

    failure_path = result_path.parent / "failure.json"
    superseded_failure: dict[str, Any] | None = None
    if failure_path.is_file():
        failure = _read_json(failure_path)
        _require(failure.get("model") == candidate["model"], f"{arm}: stale failure model mismatch")
        _require(
            failure.get("campaign_manifest_sha256") == manifest_sha,
            f"{arm}: stale failure is bound to another manifest",
        )
        superseded_failure = {
            "path": str(failure_path.resolve()),
            "sha256": _file_sha256(failure_path),
            "status": "SUPERSEDED_BY_VALID_RESULT",
            "error_type": failure.get("error_type"),
        }

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "validation_path": validation_path,
        "validation_sha256": _file_sha256(validation_path),
        "result_path": result_path,
        "result_sha256": _file_sha256(result_path),
        "model": candidate["model"],
        "checkpoint_sha256": checkpoint_sha,
        "macro": macro,
        "domains": domain_values,
        "superseded_failure": superseded_failure,
    }


def _contract_extractors() -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {
        "protocol": lambda manifest: {
            "protocol_id": manifest.get("protocol_id"),
            "admission": manifest.get("admission"),
            "teacher_branch": manifest.get("teacher_branch"),
            "py_ctcmetrics": manifest.get("py_ctcmetrics"),
            "imagecodecs": manifest.get("imagecodecs"),
        },
        "data": lambda manifest: manifest.get("data_manifest"),
        "split": lambda manifest: manifest.get("source_split_manifest_sha256"),
        "folds": lambda manifest: manifest.get("folds"),
        "head": lambda manifest: manifest.get("head"),
        "inference": lambda manifest: manifest.get("inference"),
        "linker": lambda manifest: manifest.get("linker"),
        "scoring": lambda manifest: manifest.get("scoring"),
        "implementation": lambda manifest: manifest.get("code"),
    }


def analyze_campaign(campaign_root: Path) -> dict[str, Any]:
    campaign_root = campaign_root.resolve()
    arms = {arm: _validate_arm(campaign_root, arm) for arm in ARMS}

    common_contract_sha256: dict[str, str] = {}
    for name, extract in _contract_extractors().items():
        values = {arm: extract(record["manifest"]) for arm, record in arms.items()}
        reference = values["control"]
        for arm, value in values.items():
            _require(value is not None, f"{arm}: missing common {name} contract")
            _require(value == reference, f"{arm}: common {name} contract differs from control")
        common_contract_sha256[name] = _canonical_sha256(reference)

    domain_sets = {arm: set(record["domains"]) for arm, record in arms.items()}
    for arm, domains in domain_sets.items():
        _require(domains == domain_sets["control"], f"{arm}: paired domain set differs from control")
    domains = sorted(domain_sets["control"])

    comparisons: dict[str, Any] = {}
    for name, baseline, candidate in COMPARISONS:
        metric_rows: dict[str, Any] = {}
        for metric in METRICS:
            domain_deltas = {
                domain: arms[candidate]["domains"][domain][metric] - arms[baseline]["domains"][domain][metric]
                for domain in domains
            }
            deltas = list(domain_deltas.values())
            paired_mean = _mean(deltas)
            macro_delta = arms[candidate]["macro"][metric] - arms[baseline]["macro"][metric]
            _require(
                math.isclose(paired_mean, macro_delta, rel_tol=0.0, abs_tol=1.0e-12),
                f"{name} {metric}: paired mean differs from direct 10-domain macro delta",
            )
            wins = sum(delta > TIE_TOLERANCE for delta in deltas)
            losses = sum(delta < -TIE_TOLERANCE for delta in deltas)
            ties = len(deltas) - wins - losses
            metric_rows[metric] = {
                "baseline_macro": arms[baseline]["macro"][metric],
                "candidate_macro": arms[candidate]["macro"][metric],
                "macro_delta": macro_delta,
                "paired_domain_mean_delta": paired_mean,
                "paired_domain_median_delta": float(statistics.median(deltas)),
                "win_count": wins,
                "tie_count": ties,
                "loss_count": losses,
                "domain_deltas": domain_deltas,
            }
        comparisons[name] = {
            "baseline": baseline,
            "candidate": candidate,
            "paired_domains": len(domains),
            "metrics": metric_rows,
        }

    public_arms = {
        arm: {
            "model": record["model"],
            "checkpoint_step": 20495,
            "checkpoint_sha256": record["checkpoint_sha256"],
            "macro": record["macro"],
            "domain_metrics": record["domains"],
            "inputs": {
                "manifest": {
                    "path": str(record["manifest_path"].resolve()),
                    "sha256": record["manifest_sha256"],
                },
                "validation": {
                    "path": str(record["validation_path"].resolve()),
                    "sha256": record["validation_sha256"],
                },
                "result": {
                    "path": str(record["result_path"].resolve()),
                    "sha256": record["result_sha256"],
                },
            },
            "superseded_failure": record["superseded_failure"],
        }
        for arm, record in arms.items()
    }
    return {
        "status": "VALID_COMPLETE",
        "admission": EXPECTED_ADMISSION,
        "protocol_id": EXPECTED_PROTOCOL,
        "campaign_root": str(campaign_root),
        "aggregation": "direct unweighted macro over exactly 10 held-out domains",
        "primary_metrics": list(PRIMARY_METRICS),
        "secondary_metrics": list(SECONDARY_METRICS),
        "composite_score": None,
        "tie_tolerance": TIE_TOLERANCE,
        "common_contract_sha256": common_contract_sha256,
        "paired_domains": domains,
        "arms": public_arms,
        "comparisons": comparisons,
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _readme(payload: dict[str, Any]) -> str:
    lines = [
        "# HS6-L5 Gram CTC four-arm observation",
        "",
        "Status: `VALID_COMPLETE`.",
        "",
        "Admission remains `OBSERVATIONAL_NATIVE_2D`; this is not the formal 20-domain CTC row.",
        "Macros are direct unweighted means over the same ten held-out domains. No composite score is computed.",
        "",
        "## Arm macros",
        "",
        "| arm | " + " | ".join(METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for arm in ARMS:
        macro = payload["arms"][arm]["macro"]
        lines.append("| " + arm + " | " + " | ".join(f"{macro[metric]:.8f}" for metric in METRICS) + " |")
    lines.extend(
        [
            "",
            "## Fixed comparisons",
            "",
            "| comparison | metric | macro delta | paired median delta | wins | ties | losses |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, _, _ in COMPARISONS:
        for metric in METRICS:
            row = payload["comparisons"][name]["metrics"][metric]
            lines.append(
                f"| {name} | {metric} | {row['macro_delta']:.8f} | "
                f"{row['paired_domain_median_delta']:.8f} | {row['win_count']} | "
                f"{row['tie_count']} | {row['loss_count']} |"
            )
    return "\n".join(lines) + "\n"


def write_outputs(output: Path, payload: dict[str, Any]) -> None:
    arm_rows = [
        {
            "arm": arm,
            "model": payload["arms"][arm]["model"],
            "checkpoint": payload["arms"][arm]["checkpoint_step"],
            **payload["arms"][arm]["macro"],
        }
        for arm in ARMS
    ]
    summary_rows = []
    paired_rows = []
    for name, baseline, candidate in COMPARISONS:
        for metric in METRICS:
            row = payload["comparisons"][name]["metrics"][metric]
            summary_rows.append(
                {
                    "comparison": name,
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    **{key: value for key, value in row.items() if key != "domain_deltas"},
                }
            )
            for domain, delta in row["domain_deltas"].items():
                baseline_value = payload["arms"][baseline]["domain_metrics"][domain][metric]
                candidate_value = payload["arms"][candidate]["domain_metrics"][domain][metric]
                paired_rows.append(
                    {
                        "comparison": name,
                        "baseline": baseline,
                        "candidate": candidate,
                        "domain": domain,
                        "metric": metric,
                        "baseline_value": baseline_value,
                        "candidate_value": candidate_value,
                        "delta": delta,
                    }
                )
    comparison_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    validation = {
        "status": "VALID_COMPLETE",
        "admission": payload["admission"],
        "protocol_id": payload["protocol_id"],
        "arms": len(ARMS),
        "folds_per_arm": 5,
        "domains_per_arm": 10,
        "comparisons": [name for name, _, _ in COMPARISONS],
        "composite_score_created": False,
        "comparison_sha256": hashlib.sha256(comparison_text.encode("utf-8")).hexdigest(),
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(output / "comparison.json", comparison_text)
    _atomic_text(output / "validation_report.json", json.dumps(validation, indent=2, sort_keys=True) + "\n")
    _atomic_text(
        output / "arm_macros.csv",
        _csv_text(arm_rows, ["arm", "model", "checkpoint", *METRICS]),
    )
    _atomic_text(
        output / "comparison_summary.csv",
        _csv_text(
            summary_rows,
            [
                "comparison",
                "baseline",
                "candidate",
                "metric",
                "baseline_macro",
                "candidate_macro",
                "macro_delta",
                "paired_domain_mean_delta",
                "paired_domain_median_delta",
                "win_count",
                "tie_count",
                "loss_count",
            ],
        ),
    )
    _atomic_text(
        output / "paired_domain_deltas.csv",
        _csv_text(
            paired_rows,
            [
                "comparison",
                "baseline",
                "candidate",
                "domain",
                "metric",
                "baseline_value",
                "candidate_value",
                "delta",
            ],
        ),
    )
    _atomic_text(output / "README.md", _readme(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        payload = analyze_campaign(args.campaign_root)
        if not args.check_only:
            write_outputs(args.output, payload)
    except AuditError as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "campaign_root": payload["campaign_root"],
                "output": None if args.check_only else str(args.output.resolve()),
                "arms": len(ARMS),
                "domains_per_arm": len(payload["paired_domains"]),
                "comparisons": len(COMPARISONS),
                "composite_score_created": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
