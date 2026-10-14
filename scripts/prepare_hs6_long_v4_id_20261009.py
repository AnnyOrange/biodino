#!/usr/bin/env python3
"""Prepare locked v4 ID tasks for one audited finite-stream checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import yaml

import prepare_data_quality_v4_id_20261002 as old


REPO = Path("/mnt/huawei_deepcad/dinov3")
EVAL_BASE = REPO / "plot/fig2/data_quality/long_norepeat_eval_20261009"
STEPS = tuple(range(499, 40_000, 500))


def checkpoint_audit(train: Path, arm: str, step: int) -> dict:
    launch = json.loads((train / "launch_manifest.json").read_text())
    cfg = yaml.safe_load((train / "config.yaml").read_text())
    index = Path(launch["stream_index"])
    assignment = json.loads((index / "assignment.json").read_text())
    if arm != "100tb" or not launch.get("source_population", "").startswith("all 100TB") or \
            assignment.get("format") != "100tb_full_pool_stream_v1" or \
            assignment.get("total_items") != 26_614_979:
        raise ValueError("Partial checkpoint is not from the audited 100TB full-pool stream")
    if launch.get("effective_batch") != 1024 or launch.get("world_size") != 4 or \
            cfg["compute_precision"]["distributed_mode"] != "ddp" or \
            cfg["train"]["max_updates"] != 40_000 or cfg["train"]["batch_size_per_gpu"] != 128 or \
            cfg["optim"]["gradient_accumulation_steps"] != 2 or \
            cfg["train"]["dataset_path"] != "raw100tb_full_pool_robust:" + str(index):
        raise ValueError("Partial checkpoint training configuration differs from the 100TB full-pool run")
    teacher = train / f"eval/training_{step}/teacher_checkpoint.pth"
    optimizer = train / f"ckpt/{step}/checkpoint.pth"
    if not teacher.is_file() or teacher.stat().st_size < 600_000_000 or \
            not optimizer.is_file() or optimizer.stat().st_size < 3_000_000_000:
        raise FileNotFoundError(f"Complete teacher/optimizer checkpoints are required at step {step}")
    metrics_path = train / "raw_loss_metrics.jsonl"
    target = None
    count = 0
    with metrics_path.open() as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("optimizer_update") != count - 1 or row.get("samples_seen") != count * 1024 or \
                    row.get("effective_global_batch_size") != 1024 or \
                    not math.isfinite(float(row.get("total_loss", float("nan")))):
                raise ValueError(f"Invalid optimizer/sample/loss record at update {count - 1}")
            if count == step + 1:
                target = row
                break
    if target is None or target["optimizer_update"] != step:
        raise ValueError(f"Metrics do not reach complete optimizer update {step}")

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    return dict(status="PASS_CHECKPOINT_ONLY", arm=arm, protocol="100tb_40k_full_pool_v1",
                optimizer_update=step, optimizer_updates_observed=step + 1,
                image_visits=target["samples_seen"], effective_batch=1024,
                source_subset_size=assignment["total_items"],
                stream_index=str(index), stream_assignment_sha256=sha256(index / "assignment.json"),
                optimizer_checkpoint=str(optimizer), optimizer_checkpoint_sha256=sha256(optimizer),
                teacher_checkpoint=str(teacher), teacher_checkpoint_sha256=sha256(teacher),
                config_sha256=sha256(train / "config.yaml"), latest_loss=target["total_loss"])


def remap_shared_sources(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    aliases = dict(manifest.get("source_path_aliases", {}))
    snapshots = (
        ("/mnt/huawei_deepcad/dinov3_retest_snapshot_20260918/",
         "/mnt/huawei_deepcad/dinov3_retest_snapshot_20260918_fm_bound/"),
    )
    rows = list(manifest.get("datasets", [])) + [task["dataset"] for task in manifest.get("tasks", [])]
    for row in rows:
        sources = row.get("source_hashes", {})
        remapped = {}
        for source, expected in sources.items():
            candidate = Path(source)
            if not candidate.is_file():
                for old_prefix, new_prefix in snapshots:
                    if source.startswith(old_prefix):
                        candidate = Path(source.replace(old_prefix, new_prefix, 1))
                        break
            if not candidate.is_file() or old.sha256(candidate) != expected:
                raise ValueError(f"Missing or changed locked source: {source}")
            resolved = str(candidate)
            if resolved != source:
                aliases[source] = resolved
            remapped[resolved] = expected
        if sources:
            row["source_hashes"] = remapped
    manifest["source_path_aliases"] = aliases
    temp = manifest_path.with_name(f".{manifest_path.name}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    temp.replace(manifest_path)
    return aliases


def prepare(arm: str, step: int, train_root: Path | None = None,
            eval_base: Path | None = None, label_tag: str | None = None,
            allow_partial: bool = False) -> dict:
    if step not in STEPS:
        raise ValueError(f"Unexpected checkpoint: {step}")
    train = train_root or REPO / f"outputs/01_training_runs/HS6_L_quality_long_noreplace_{arm}_ddp_gb1024_16k_20261009"
    if allow_partial:
        audit = checkpoint_audit(train, arm, step)
    else:
        audit = json.loads((train / "audit.json").read_text())
        if audit["status"] != "PASS" or audit["arm"] != arm:
            raise ValueError("Long finite-stream training audit has not passed")
        if audit.get("protocol") in ("100tb_40k_repeated_v1", "100tb_40k_full_pool_v1"):
            if arm != "100tb" or audit.get("optimizer_updates") != 40_000 or \
                    audit.get("image_visits") != 40_960_000:
                raise ValueError("40k 100TB training audit differs from the evaluation protocol")
        elif audit.get("unique_image_visits") != 16_384_000:
            raise ValueError("16k finite-stream training audit differs from the evaluation protocol")
    checkpoint = train / f"eval/training_{step}/teacher_checkpoint.pth"
    config = train / "config.yaml"
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError("Checkpoint or training config is missing")
    checkpoint_hash = old.sha256(checkpoint)
    config_hash = old.sha256(config)
    root = (eval_base or EVAL_BASE) / f"{arm}_ck{step}"
    if root.exists():
        prepared = root / "prepared.json"
        if prepared.is_file():
            record = json.loads(prepared.read_text())
            if record["checkpoint_sha256"] == checkpoint_hash and record["config_sha256"] == config_hash:
                return record
        raise FileExistsError(f"Conflicting evaluation campaign: {root}")

    v4 = old.module_from(old.V4_MODULE)
    v4.SOURCE = old.REPO
    import run_shared_frozen_stage1_20260917 as source_queue
    source_snapshot = source_queue.current_source_snapshot()
    source_snapshot['git_commit'] = os.environ.get('DINOV3_GIT_COMMIT', 'UNKNOWN')
    for path, expected in v4.LOCKED_INPUTS.items():
        if old.sha256(Path(path)) != expected:
            raise ValueError(f"Locked v4 input changed: {path}")
    monuseg = copy.deepcopy(json.loads(old.MONUSEG_REFERENCE.read_text()))
    spec = monuseg["datasets"][0]
    if spec["counts"] != {"train": 30, "val": 7, "test": 14} or \
            spec["split_protocol_id"] != "monuseg2018-train30-extra7val-test14-v1":
        raise ValueError("MoNuSeg split is not the locked 30/7/14 protocol")
    monuseg["source_snapshot"] = source_snapshot
    monuseg["git_commit"] = source_snapshot["git_commit"]
    monuseg["external_source_hashes"] = {
        str(REPO / "scripts/prepare_hs6_long_v4_id_20261009.py"): old.sha256(Path(__file__).resolve()),
        str(REPO / "scripts/run_retest_fleet_20260918.py"): old.sha256(REPO / "scripts/run_retest_fleet_20260918.py"),
    }

    label = f"dq_long_{label_tag + '_' if label_tag else ''}{arm}_ck{step}"
    shared_asset = dict(arm=label, checkpoint_id=str(step), path=str(checkpoint),
                        config=str(config), kind="dinov3", model_id="", reserve_mib=4096)
    shared = copy.deepcopy(json.loads(old.SHARED_REFERENCE.read_text()))
    shared.pop("execution_hosts", None)
    specs = [row for row in shared["datasets"]
             if row["status"] == "PASS" and row["task"] not in ("ood", "cell_tracking")]
    source_aliases = {}
    for row in specs:
        for source, expected in list(row.get("source_hashes", {}).items()):
            candidate = Path(source)
            if not candidate.is_file() and source.startswith(
                    "/mnt/huawei_deepcad/dinov3_retest_snapshot_20260918/"):
                candidate = Path(source.replace(
                    "/mnt/huawei_deepcad/dinov3_retest_snapshot_20260918/",
                    "/mnt/huawei_deepcad/dinov3/", 1))
            if not candidate.is_file() or old.sha256(candidate) != expected:
                raise ValueError(f"Locked v4 ID source is missing or changed: {source}")
            if str(candidate) != source:
                del row["source_hashes"][source]
                row["source_hashes"][str(candidate)] = expected
                source_aliases[source] = str(candidate)
    if len(specs) != 38 or any(row["dataset"] == "monuseg" for row in specs):
        raise ValueError("Unexpected v4 ID task inventory")
    shared.update(
        campaign_scope="V4_ID_HS6_LONG_FINITE_20261009", created_unix=time.time(),
        checkpoint_assets=[shared_asset], datasets=specs,
        checkpoint_teacher_sha256={label: checkpoint_hash},
        checkpoint_config_sha256={label: config_hash},
        old_results_relabelled=False, legacy_reuse=False, online_checkpoints=False,
        v4_aggregate_allowed=False, full_v3_aggregate_allowed=False,
        source_path_aliases=source_aliases,
    )
    shared["tasks"] = [dict(
        key=f"{label}_ck{step}__{row['task']}__{row['dataset']}" +
            (f"__primary-last__{row['split']}" if row["task"] == "segmentation" else ""),
        asset=shared_asset, dataset=row) for row in specs]
    shared["inventory"] = [row for row in shared["inventory"]
                           if row["task"] not in ("ood", "cell_tracking")]
    shared["external_source_hashes"] = {
        str(REPO / "Evaluation Rules/protocol_v4.json"):
            old.sha256(REPO / "Evaluation Rules/protocol_v4.json"),
        str(Path(__file__).resolve()): old.sha256(Path(__file__).resolve()),
        str(REPO / "scripts/run_data_quality_v4_shared_worker_20261012.py"):
            old.sha256(REPO / "scripts/run_data_quality_v4_shared_worker_20261012.py"),
        str(REPO / "scripts/run_retest_fleet_20260918.py"):
            old.sha256(REPO / "scripts/run_retest_fleet_20260918.py"),
    }
    shared["source_snapshot"] = source_snapshot
    shared["git_commit"] = source_snapshot["git_commit"]
    old.save(root / "shared/campaign_manifest.json", shared)

    v4.ROOT = root / "extension"
    asset = dict(arm=label, checkpoint_id=step, checkpoint=checkpoint, config=config,
                 checkpoint_sha256=checkpoint_hash, shared_evidence=root / "shared")
    v4.ASSETS = (asset,)
    builders = [
        lambda order: v4.retrieval_task(asset, "nct-crc-he-100", order),
        lambda order: v4.frozen_task(asset, "conic-cell-count", order),
        lambda order: v4.rxrx3_task(asset, order),
        lambda order: v4.detection_task(asset, "bbbc038", order),
        lambda order: v4.frozen_task(asset, "lc25000", order),
        lambda order: v4.retrieval_task(asset, "lc25000", order),
        lambda order: v4.frozen_task(asset, "livecell-cell-count", order),
        lambda order: v4.detection_task(asset, "conic", order),
        lambda order: v4.detection_task(asset, "livecell", order),
    ]
    tasks = []
    for order, builder in enumerate(builders, 1):
        task = builder(order)
        task.update(protocol_id=v4.PROTOCOL_ID, checkpoint=str(checkpoint),
                    checkpoint_sha256=checkpoint_hash, config=str(config),
                    config_sha256=config_hash,
                    source_entry_sha256=v4.source_digest_for(task))
        tasks.append(task)
        old.save(root / "extension/tasks" / f"{task['id']}.json", task)
        if task["done_kind"] == "rxrx3_json":
            old.save(Path(task["output"]) / "campaign_manifest.json", dict(
                protocol_id=v4.PROTOCOL_ID, protocol_sha256=old.sha256(v4.PROTOCOL),
                fixed_split_protocol_id=v4.RXRX3_PROTOCOL,
                fixed_split_sha256=v4.LOCKED_INPUTS[str(v4.RXRX3_CACHE / "split_manifest.jsonl")],
                checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                config=str(config), config_sha256=config_hash,
                teacher_branch=True, batch_size=64, query=734, gallery=734,
                created_utc=v4.now()))
    old.save(root / "extension/campaign_manifest.json", dict(
        protocol_id=v4.PROTOCOL_ID, protocol_sha256=old.sha256(v4.PROTOCOL),
        scope="v4 ID extensions, with MoNuSeg evaluated by locked 30/7/14 split",
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        config=str(config), config_sha256=config_hash,
        task_ids=[task["id"] for task in tasks], source_snapshot=str(v4.SOURCE)))

    monuseg_asset = dict(shared_asset, reserve_mib=16000)
    for key in ("execution_hosts", "asset_additions", "supersedes", "superseded_split",
                "external_sources_changed_since_legacy", "checkpoint_selection_rule_20tb"):
        monuseg.pop(key, None)
    monuseg.update(
        protocol_id="data-quality-monuseg-train30val7-test14-v1",
        checkpoint_assets=[monuseg_asset], training_roots={label: str(train)},
        online_checkpoints=False, legacy_reuse=False, created_unix=time.time(),
        tasks=[dict(key=f"{label}_ck{step}__segmentation__monuseg__primary-last__{spec['split']}",
                    asset=monuseg_asset, dataset=spec)],
    )
    monuseg["external_source_hashes"][str(Path(__file__).resolve())] = old.sha256(Path(__file__).resolve())
    old.save(root / "monuseg/campaign_manifest.json", monuseg)
    if allow_partial:
        old.save(root / "checkpoint_audit.json", audit)
    result = dict(arm=f"long_{label_tag + '_' if label_tag else ''}{arm}_ck{step}", checkpoint=str(checkpoint),
                  checkpoint_sha256=checkpoint_hash, config_sha256=config_hash,
                  training_audit=str(root / "checkpoint_audit.json") if allow_partial else str(train / "audit.json"),
                  training_state="IN_PROGRESS" if allow_partial else "COMPLETE",
                  checkpoint_optimizer_update=step,
                  shared_tasks=38, extension_tasks=9, monuseg_tasks=1,
                  monuseg_split=spec["split_protocol_id"], monuseg_counts=spec["counts"],
                  prepared_unix=time.time())
    old.save(root / "prepared.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("20tb", "100tb"), required=True)
    parser.add_argument("--step", type=int, choices=STEPS, required=True)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--eval-base", type=Path)
    parser.add_argument("--label-tag")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Prepare a checkpoint-level audited campaign while training is still running")
    parser.add_argument("--repair-shared-manifest-sources", type=Path,
                        help="Repair missing shared split paths only when fallback SHA-256 matches")
    args = parser.parse_args()
    if args.repair_shared_manifest_sources:
        print(json.dumps(remap_shared_sources(args.repair_shared_manifest_sources), indent=2))
        return
    print(json.dumps(prepare(args.arm, args.step, args.train_root, args.eval_base,
                             args.label_tag, args.allow_partial), indent=2))


if __name__ == "__main__":
    main()
