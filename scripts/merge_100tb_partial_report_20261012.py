#!/usr/bin/env python3
"""Merge currently validated 100TB v4 components into the comparison report."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
CAMPAIGN = ROOT / "plot/fig2/data_quality/long_100tb_full_pool_partial_v4_id_20261012_retry1/100tb_ck7999"
REPORT = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/100tb_fullpool_every500_v4_20261019/ck7999"
CHECKPOINT_SHA = "c4bf90f1943c4399c3dcdf55ba32054afc0a9ee9b65f7f429c1643dd703c0d10"


def read(path: Path):
    return json.loads(path.read_text())


def scalar(row: dict, family: str) -> tuple[str, float]:
    metric = {
        "classification": "macro_f1",
        "regression": "spearman",
        "retrieval": "map_at_5",
        "clustering": "nmi",
        "detection_proxy": "test_patch_f1",
    }[family]
    value = float(row[metric])
    if family == "detection_proxy":
        value /= 100.0
    return metric, value


def add_component(rows: list[dict], family: str, dataset: str, split: str,
                  metric: str, value: float, source: Path, gpu: int | str):
    rows.append({
        "family": family, "dataset": dataset, "split": split, "metric": metric,
        "value": value, "status": "VALID_COMPLETE", "gpu": gpu,
        "checkpoint_step": 7999, "checkpoint_sha256": CHECKPOINT_SHA,
        "source": str(source),
    })


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    manifest = read(CAMPAIGN / "shared/campaign_manifest.json")
    done_dir = CAMPAIGN / "shared/_state/done"
    done = {path.stem for path in done_dir.glob("*.json")
            if read(path).get("status") == "VALID_COMPLETE"}
    running = {path.stem for path in (CAMPAIGN / "shared/_state/running").glob("*.json")}
    failed = {path.stem for path in (CAMPAIGN / "shared/_state/failed_resource").glob("*.json")}
    task_gpu = {}
    for path in (CAMPAIGN / "shared/_state/claims").glob("*/owner.json"):
        task_gpu[path.parent.name] = read(path).get("gpu", "")

    rows: list[dict] = []
    family_counts: Counter = Counter()
    gpu_counts: Counter = Counter()
    task_rows = []
    for task in manifest["tasks"]:
        key = task["key"]
        valid = key in done
        family = task["dataset"]["task"]
        dataset = task["dataset"]["dataset"]
        gpu = task_gpu.get(key, "")
        task_status = ("VALID_COMPLETE" if valid else "RUNNING" if key in running else
                       "FAILED_RESOURCE" if key in failed else "PENDING_OR_STALE_CLAIM")
        task_rows.append({"task_key": key, "family": family, "dataset": dataset,
                          "status": task_status, "gpu": gpu})
        if not valid:
            continue
        family_counts[family] += 1
        gpu_counts[str(gpu)] += 1
        cell = CAMPAIGN / "shared/cells" / key
        if family == "segmentation":
            paths = sorted(path for path in (cell / "results").rglob("results.json")
                           if "budget50" in path.parts)
            vals = [float(read(path)["test"]["mDice"]) for path in paths]
            if len(vals) != 3:
                raise ValueError(f"Expected three validated budget50 seeds: {key}: {len(vals)}")
            add_component(rows, family, dataset, task["dataset"].get("split", ""),
                          "mDice_seed_mean", statistics.mean(vals), cell / "validation_report.json", gpu)
            continue
        result = read(cell / "component_result.json")
        if family == "retrieval":
            if "rows" not in result:
                for derived in ("retrieval", "clustering"):
                    metric, value = scalar(result, derived)
                    add_component(rows, derived, dataset, task["dataset"].get("split", ""),
                                  metric, value, cell / "component_result.json", gpu)
                continue
            for derived in ("retrieval", "clustering"):
                candidates = [item for item in result.get("rows", [])
                              if item.get("task") == derived and
                              (derived != "retrieval" or item.get("aggregation") == "global") and
                              (derived != "clustering" or "ge10" not in item.get("protocol", ""))]
                if len(candidates) == 1:
                    metric, value = scalar(candidates[0], derived)
                    add_component(rows, derived, dataset, task["dataset"].get("split", ""),
                                  metric, value, cell / "component_result.json", gpu)
        else:
            metric, value = scalar(result, family)
            add_component(rows, family, dataset, task["dataset"].get("split", ""),
                          metric, value, cell / "component_result.json", gpu)

    ext_root = CAMPAIGN / "extension"
    ext_manifest = read(ext_root / "campaign_manifest.json")
    ext_gpu = {}
    for log in CAMPAIGN.glob("extension_worker_gpu*.log"):
        gpu = int(log.stem.rsplit("gpu", 1)[1])
        for line in log.read_text().splitlines():
            if line.startswith("DONE gpu=") and "VALID_COMPLETE" in line:
                marker = "task="
                ext_gpu[line.split(marker, 1)[1].split()[0]] = gpu
    extension_done = 0
    for task_path in sorted((ext_root / "tasks").glob("*.json")):
        task = read(task_path)
        key = task["id"]
        validation = Path(task["output"]) / "validation_report.json"
        valid_extension = validation.is_file() and read(validation).get("status") == "VALID_COMPLETE"
        gpu = ext_gpu.get(key, "")
        task_rows.append({"task_key": key, "family": task["family"], "dataset": task["dataset"],
                          "status": "VALID_COMPLETE" if valid_extension else "PENDING_OR_FAILED",
                          "gpu": gpu})
        if not valid_extension:
            continue
        extension_done += 1
        family = task["family"]
        dataset = task["dataset"]
        out = Path(task["output"])
        kind = task["done_kind"]
        if kind in {"frozen_csv", "retrieval_csv"}:
            with (out / "summary.csv").open(newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            selected_families = ("retrieval", "clustering") if kind == "retrieval_csv" else (family,)
            for derived in selected_families:
                selected = [row for row in csv_rows if
                            (kind == "retrieval_csv" and row.get("task") == "retrieval_clustering") or
                            (kind == "frozen_csv" and row.get("task") == derived)]
                if len(selected) != 1:
                    raise ValueError(f"Ambiguous extension CSV row for {key}/{derived}: {len(selected)}")
                metric, value = scalar(selected[0], derived)
                add_component(rows, derived, dataset, selected[0].get("split", ""), metric,
                              value, out / "summary.csv", gpu)
        elif kind == "detection_json":
            path = out / "results-detection.csv"
            with path.open(newline="") as stream:
                item_rows = list(csv.DictReader(stream))
            if len(item_rows) != 1:
                raise ValueError(f"Expected one detection row: {path}")
            metric, value = scalar(item_rows[0], "detection_proxy")
            add_component(rows, "detection_proxy", dataset, item_rows[0].get("conic_split_protocol", ""),
                          metric, value, path, gpu)
        elif kind == "rxrx3_json":
            paths = list((out / "models").rglob("results.json"))
            if len(paths) != 1:
                raise ValueError(f"Expected one RxRx3 result under {out}")
            result = read(paths[0])["tests"]["rxrx3"]
            add_component(rows, "retrieval", dataset, "plate-disjoint", "map", float(result["map"]), paths[0], gpu)
            add_component(rows, "clustering", dataset, "plate-disjoint", "nmi", float(result["nmi"]), paths[0], gpu)

    monuseg_root = CAMPAIGN / "monuseg"
    monuseg_manifest_path = monuseg_root / "campaign_manifest.json"
    monuseg_manifest = read(monuseg_manifest_path) if monuseg_manifest_path.is_file() else None
    monuseg_status = "NOT_TESTED"
    monuseg_gpu = ""
    if monuseg_manifest and len(monuseg_manifest.get("tasks", [])) == 1:
        task = monuseg_manifest["tasks"][0]
        cell = monuseg_root / "cells" / task["key"]
        validation_path = cell / "validation_report.json"
        if validation_path.is_file():
            validation = read(validation_path)
            monuseg_status = validation.get("status", "INVALID")
            invocation_path = cell / "invocation_manifest.json"
            if invocation_path.is_file():
                monuseg_gpu = read(invocation_path).get("gpu", "")
            if monuseg_status == "VALID_COMPLETE":
                if task["dataset"].get("counts") != {"train": 30, "val": 7, "test": 14}:
                    raise ValueError("MoNuSeg task does not use the registered 30/7/14 split")
                paths = sorted(path for path in (cell / "results").rglob("results.json")
                               if "budget50" in path.parts)
                vals = [float(read(path)["test"]["mDice"]) for path in paths]
                if len(vals) != 3:
                    raise ValueError(f"Expected three validated MoNuSeg budget50 seeds, got {len(vals)}")
                add_component(rows, "segmentation", "monuseg", "train30-val7-test14",
                              "mDice_seed_mean", statistics.mean(vals), validation_path, monuseg_gpu)
                task_rows.append({"task_key": task["key"], "family": "segmentation",
                                  "dataset": "monuseg", "status": "VALID_COMPLETE", "gpu": monuseg_gpu})
                gpu_counts[str(monuseg_gpu)] += 1
            else:
                task_rows.append({"task_key": task["key"], "family": "segmentation",
                                  "dataset": "monuseg", "status": monuseg_status, "gpu": monuseg_gpu})
        else:
            task_rows.append({"task_key": task["key"], "family": "segmentation",
                              "dataset": "monuseg", "status": monuseg_status, "gpu": ""})

    extension_ids = {read(path)["id"] for path in (ext_root / "tasks").glob("*.json")}
    if extension_ids != set(ext_manifest["task_ids"]):
        raise ValueError("Extension manifest/task inventory mismatch")

    score_path = REPORT / "v4_id_validated_partial_cell_scores.csv"
    with score_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    task_path = REPORT / "v4_id_task_status.csv"
    with task_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("task_key", "family", "dataset", "status", "gpu"))
        writer.writeheader()
        writer.writerows(task_rows)

    monitor = [json.loads(line) for line in (CAMPAIGN / "gpu_monitor.jsonl").read_text().splitlines() if line.strip()]
    peak_total = {gpu: max((row["gpu_status"][gpu]["used_mib"] for row in monitor), default=0)
                  for gpu in range(8)}
    own_peak = {gpu: max((row["gpu_status"][gpu].get("own_used_mib", 0) for row in monitor), default=0)
                for gpu in range(8)}
    metric_counts: Counter = Counter(str(row["gpu"]) for row in rows)
    updated_monuseg_path = CAMPAIGN / "monuseg_updated_manifest_20261012"
    updated_monuseg_summary_path = updated_monuseg_path / "validation_summary.json"
    updated_monuseg = None
    if updated_monuseg_summary_path.is_file():
        updated_monuseg = read(updated_monuseg_summary_path)
        if updated_monuseg.get("status") != "VALID_COMPLETE":
            raise ValueError("Updated-manifest MoNuSeg retest is not validated")
        if updated_monuseg.get("checkpoint_sha256") != CHECKPOINT_SHA:
            raise ValueError("Updated-manifest MoNuSeg checkpoint differs from the report checkpoint")
        updated_rows = []
        for epoch, mean in sorted(updated_monuseg["mean_test_mDice_by_epochs"].items(), key=lambda item: int(item[0])):
            updated_rows.append({
                "protocol": updated_monuseg["split"], "manifest_sha256": updated_monuseg["manifest_sha256"],
                "checkpoint_step": 7999, "epochs": int(epoch), "seed_count": 3,
                "mean_test_mDice": mean,
                "seed_values": ";".join(str(item["test_mDice"]) for item in updated_monuseg["results"]
                                          if item["epochs"] == int(epoch)),
            })
        with (REPORT / "monuseg_updated_manifest_retest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(updated_rows[0]))
            writer.writeheader()
            writer.writerows(updated_rows)
    status = {
        "checkpoint_step": 7999, "optimizer_updates_observed": 8000,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "shared_valid_complete": len(done), "shared_expected": len(manifest["tasks"]),
        "extension_valid_complete": extension_done, "extension_expected": len(ext_manifest["task_ids"]),
        "monuseg": monuseg_status,
        "monuseg_updated_manifest_retest": ({
            "status": "VALID_COMPLETE_SEPARATE_PROTOCOL",
            "protocol": updated_monuseg["split"],
            "source_manifest_sha256": updated_monuseg["manifest_sha256"],
            "mean_test_mDice_by_epochs": updated_monuseg["mean_test_mDice_by_epochs"],
            "source_summary": str(updated_monuseg_summary_path),
        } if updated_monuseg else "NOT_TESTED"),
        "monuseg_split_required": {"train": 30, "val": 7, "test": 14},
        "monuseg_split_identity_sha256": (monuseg_manifest.get("datasets", [{}])[0].get("split_identity_sha256")
                                         if monuseg_manifest else None),
        "validated_task_counts_per_gpu": {str(gpu): int(gpu_counts[str(gpu)] +
            sum(1 for assigned in ext_gpu.values() if assigned == gpu)) for gpu in range(8)},
        "validated_metric_rows_per_gpu": {str(gpu): int(metric_counts[str(gpu)]) for gpu in range(8)},
        "observed_peak_total_gpu_memory_mib": {str(k): v for k, v in peak_total.items()},
        "observed_peak_campaign_attributed_gpu_memory_mib": {str(k): v for k, v in own_peak.items()},
        "gpu_memory_target_60pct_met": False,
        "report_status": ("V4_ID_AND_EXTENSIONS_COMPLETE; MONUSEG_VALID" if
                          len(done) == len(manifest["tasks"]) and
                          extension_done == len(ext_manifest["task_ids"]) and
                          monuseg_status == "VALID_COMPLETE" else
                          "V4_ID_AND_EXTENSIONS_COMPLETE; MONUSEG_PENDING" if
                          len(done) == len(manifest["tasks"]) and
                          extension_done == len(ext_manifest["task_ids"]) else
                          "PARTIAL_V4_ID"),
        "source_campaign": str(CAMPAIGN),
    }
    (REPORT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    lines = [
        "# 100TB Full-Pool v4 ID Results",
        "",
        f"Status: `{status['report_status']}`. Only validated components are included in the score table.",
        "",
        f"- Checkpoint: step 7999 (8,000 optimizer updates), SHA-256 `{CHECKPOINT_SHA}`. This is an intermediate checkpoint from the 40,000-update training run.",
        f"- Shared v4 ID: {len(done)}/{len(manifest['tasks'])} validated components.",
        f"- v4 extensions: {extension_done}/{len(ext_manifest['task_ids'])} validated tasks.",
        "- Per-GPU counts are in `status.json`; unique tasks and metric rows are reported separately.",
        f"- MoNuSeg: `{monuseg_status}` on the registered train30/val7/test14 split; split identity SHA-256 `{status['monuseg_split_identity_sha256'] or 'not recorded'}`.",
        (f"- Additional MoNuSeg run: separate updated-manifest protocol `{updated_monuseg['split']}`, "
         f"manifest SHA-256 `{updated_monuseg['manifest_sha256']}`; mean test mDice E20="
         f"`{updated_monuseg['mean_test_mDice_by_epochs']['20']:.4f}`, E50="
         f"`{updated_monuseg['mean_test_mDice_by_epochs']['50']:.4f}` (3 seeds each). "
         "The source manifest is explicitly named superseded, so this result is not substituted for the registered formal split." if updated_monuseg else ""),
        "- GPU requirement: 8 GPUs were included in the evaluation worker configuration. The 60% VRAM target was not met by campaign-attributed memory; total-card readings include unrelated processes and are not counted as campaign usage.",
        "",
        "Files:",
        "- `v4_id_validated_partial_cell_scores.csv`: validated score rows and their source artifacts.",
        "- `v4_id_task_status.csv`: all shared tasks with current validation status and assigned GPU.",
        "- `status.json`: checkpoint provenance, completion counts, per-GPU counts, and memory audit.",
        ("- `monuseg_updated_manifest_retest.csv`: E20/E50 means and seed scores for the separate updated-manifest run." if updated_monuseg else ""),
        "",
        f"Raw evaluation artifacts: `{CAMPAIGN}`.",
        "",
        "This is the requested v4 ID scope, not an OOD or cell-tracking aggregate. The checkpoint is an intermediate step from the 40k-update training run, not its final result.",
    ]
    (REPORT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
