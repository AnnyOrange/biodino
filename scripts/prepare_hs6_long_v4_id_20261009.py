#!/usr/bin/env python3
"""Prepare locked v4 ID tasks for one audited finite-stream checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import prepare_data_quality_v4_id_20261002 as old


REPO = Path("/mnt/huawei_deepcad/dinov3")
EVAL_BASE = REPO / "plot/fig2/data_quality/long_norepeat_eval_20261009"
STEPS = (3999, 7999, 11999, 15999)


def prepare(arm: str, step: int) -> dict:
    if step not in STEPS:
        raise ValueError(f"Unexpected checkpoint: {step}")
    train = REPO / f"outputs/01_training_runs/HS6_L_quality_long_noreplace_{arm}_ddp_gb1024_16k_20261009"
    audit = json.loads((train / "audit.json").read_text())
    if audit["status"] != "PASS" or audit["arm"] != arm or audit["unique_image_visits"] != 16_384_000:
        raise ValueError("Long finite-stream training audit has not passed")
    checkpoint = train / f"eval/training_{step}/teacher_checkpoint.pth"
    config = train / "config.yaml"
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError("Checkpoint or training config is missing")
    checkpoint_hash = old.sha256(checkpoint)
    config_hash = old.sha256(config)
    root = EVAL_BASE / f"{arm}_ck{step}"
    if root.exists():
        prepared = root / "prepared.json"
        if prepared.is_file():
            record = json.loads(prepared.read_text())
            if record["checkpoint_sha256"] == checkpoint_hash and record["config_sha256"] == config_hash:
                return record
        raise FileExistsError(f"Conflicting evaluation campaign: {root}")

    v4 = old.module_from(old.V4_MODULE)
    for path, expected in v4.LOCKED_INPUTS.items():
        if old.sha256(Path(path)) != expected:
            raise ValueError(f"Locked v4 input changed: {path}")
    monuseg = copy.deepcopy(json.loads(old.MONUSEG_REFERENCE.read_text()))
    spec = monuseg["datasets"][0]
    if spec["counts"] != {"train": 30, "val": 7, "test": 14} or \
            spec["split_protocol_id"] != "monuseg2018-train30-extra7val-test14-v1":
        raise ValueError("MoNuSeg split is not the locked 30/7/14 protocol")
    for path, expected in monuseg["external_source_hashes"].items():
        if old.sha256(Path(path)) != expected:
            raise ValueError(f"MoNuSeg source changed: {path}")

    label = f"dq_long_{arm}_ck{step}"
    shared_asset = dict(arm=label, checkpoint_id=str(step), path=str(checkpoint),
                        config=str(config), kind="dinov3", model_id="", reserve_mib=4096)
    shared = copy.deepcopy(json.loads(old.SHARED_REFERENCE.read_text()))
    shared.pop("execution_hosts", None)
    specs = [row for row in shared["datasets"]
             if row["status"] == "PASS" and row["task"] not in ("ood", "cell_tracking")]
    if len(specs) != 38 or any(row["dataset"] == "monuseg" for row in specs):
        raise ValueError("Unexpected v4 ID task inventory")
    shared.update(
        campaign_scope="V4_ID_HS6_LONG_FINITE_20261009", created_unix=time.time(),
        checkpoint_assets=[shared_asset], datasets=specs,
        checkpoint_teacher_sha256={label: checkpoint_hash},
        checkpoint_config_sha256={label: config_hash},
        old_results_relabelled=False, legacy_reuse=False, online_checkpoints=False,
        v4_aggregate_allowed=False, full_v3_aggregate_allowed=False,
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
        str(old.SHARED_SOURCE / "scripts/run_retest_fleet_20260918.py"):
            old.sha256(old.SHARED_SOURCE / "scripts/run_retest_fleet_20260918.py"),
    }
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
    result = dict(arm=f"long_{arm}_ck{step}", checkpoint=str(checkpoint),
                  checkpoint_sha256=checkpoint_hash, config_sha256=config_hash,
                  training_audit=str(train / "audit.json"),
                  shared_tasks=38, extension_tasks=9, monuseg_tasks=1,
                  monuseg_split=spec["split_protocol_id"], monuseg_counts=spec["counts"],
                  prepared_unix=time.time())
    old.save(root / "prepared.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("20tb", "100tb"), required=True)
    parser.add_argument("--step", type=int, choices=STEPS, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.arm, args.step), indent=2))


if __name__ == "__main__":
    main()
