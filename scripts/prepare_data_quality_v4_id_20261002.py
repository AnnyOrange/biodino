#!/usr/bin/env python3
"""Prepare provenance-locked v4 ID evaluation tasks for one audited 1M arm."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import time
from pathlib import Path

from audit_data_quality_1m_20261002 import ARMS, ROOT, sha256


REPO = Path("/mnt/huawei_deepcad/dinov3")
SHARED_REFERENCE = REPO / "outputs/02_eval_inputs/shared_fleet_reference_20260930/corrected_campaign_manifest.json"
MONUSEG_REFERENCE = REPO / "outputs/02_eval_runs/monuseg_train30val7_test14_retest_20260929/campaign_manifest.json"
SHARED_SOURCE = Path("/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918")
MONUSEG_SOURCE = Path("/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929")
V4_MODULE = REPO / "scripts/run_fourmodel_v4_completion_20260924.py"


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temp, path)


def module_from(path: Path):
    spec = importlib.util.spec_from_file_location("data_quality_v4_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(arm: str) -> dict:
    train = ROOT / "training" / arm
    audit = json.loads((train / "audit.json").read_text())
    if audit["status"] != "PASS" or audit["arm"] != arm or audit["consumed_unique_images"] != 999_424:
        raise ValueError("Training audit did not pass")
    checkpoint, config = Path(audit["checkpoint"]), Path(audit["config"])
    if sha256(checkpoint) != audit["checkpoint_sha256"] or sha256(config) != audit["config_sha256"]:
        raise ValueError("Training checkpoint or config changed after audit")
    v4 = module_from(V4_MODULE)
    for path, expected in v4.LOCKED_INPUTS.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Locked v4 input changed: {path}")
    monuseg = copy.deepcopy(json.loads(MONUSEG_REFERENCE.read_text()))
    spec = monuseg["datasets"][0]
    if spec["counts"] != {"train": 30, "val": 7, "test": 14} or \
            spec["split_protocol_id"] != "monuseg2018-train30-extra7val-test14-v1":
        raise ValueError("MoNuSeg reference split is not 30/7/14")
    for path, expected in monuseg["external_source_hashes"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"MoNuSeg reference source changed: {path}")
    root = ROOT / "eval" / arm
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    label = f"dq_{arm}"
    shared_asset = dict(arm=label, checkpoint_id="975", path=str(checkpoint),
                        config=str(config), kind="dinov3", model_id="", reserve_mib=4096)

    shared = copy.deepcopy(json.loads(SHARED_REFERENCE.read_text()))
    shared.pop("execution_hosts", None)
    specs = [row for row in shared["datasets"]
             if row["status"] == "PASS" and row["task"] not in ("ood", "cell_tracking")]
    if len(specs) != 38 or any(row["dataset"] == "monuseg" for row in specs):
        raise ValueError("Unexpected v4 shared ID task inventory")
    shared.update(
        campaign_scope="V4_ID_SHARED_DATA_QUALITY_1M",
        explicit_user_authorization="2026-10-02 matched 1M quality comparison, ID v4 evaluation",
        created_unix=time.time(), checkpoint_assets=[shared_asset], datasets=specs,
        checkpoint_teacher_sha256={label: audit["checkpoint_sha256"]},
        checkpoint_config_sha256={label: audit["config_sha256"]},
        old_results_relabelled=False, legacy_reuse=False, online_checkpoints=False,
        v4_aggregate_allowed=False, full_v3_aggregate_allowed=False,
    )
    shared["tasks"] = [dict(
        key=f"{label}_ck975__{row['task']}__{row['dataset']}" +
            (f"__primary-last__{row['split']}" if row["task"] == "segmentation" else ""),
        asset=shared_asset, dataset=row) for row in specs]
    shared["inventory"] = [row for row in shared["inventory"]
                           if row["task"] not in ("ood", "cell_tracking")]
    shared["external_source_hashes"] = {
        str(REPO / "Evaluation Rules/protocol_v4.json"): sha256(REPO / "Evaluation Rules/protocol_v4.json"),
        str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        str(SHARED_SOURCE / "scripts/run_retest_fleet_20260918.py"):
            sha256(SHARED_SOURCE / "scripts/run_retest_fleet_20260918.py"),
    }
    save(root / "shared/campaign_manifest.json", shared)

    v4.ROOT = root / "extension"
    asset = dict(arm=label, checkpoint_id=975, checkpoint=checkpoint, config=config,
                 checkpoint_sha256=audit["checkpoint_sha256"], shared_evidence=root / "shared")
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
                    checkpoint_sha256=audit["checkpoint_sha256"], config=str(config),
                    config_sha256=audit["config_sha256"],
                    source_entry_sha256=v4.source_digest_for(task))
        tasks.append(task)
        save(root / "extension/tasks" / f"{task['id']}.json", task)
        if task["done_kind"] == "rxrx3_json":
            save(Path(task["output"]) / "campaign_manifest.json", dict(
                protocol_id=v4.PROTOCOL_ID, protocol_sha256=sha256(v4.PROTOCOL),
                fixed_split_protocol_id=v4.RXRX3_PROTOCOL,
                fixed_split_sha256=v4.LOCKED_INPUTS[str(v4.RXRX3_CACHE / "split_manifest.jsonl")],
                checkpoint=str(checkpoint), checkpoint_sha256=audit["checkpoint_sha256"],
                config=str(config), config_sha256=audit["config_sha256"],
                teacher_branch=True, batch_size=64, query=734, gallery=734,
                created_utc=v4.now()))
    save(root / "extension/campaign_manifest.json", dict(
        protocol_id=v4.PROTOCOL_ID, protocol_sha256=sha256(v4.PROTOCOL),
        scope="admitted v4 ID extensions except MoNuSeg, which uses the 30/7/14 evaluator",
        checkpoint=str(checkpoint), checkpoint_sha256=audit["checkpoint_sha256"],
        config=str(config), config_sha256=audit["config_sha256"],
        task_ids=[task["id"] for task in tasks], source_snapshot=str(v4.SOURCE)))

    monuseg_asset = dict(shared_asset, reserve_mib=16000)
    monuseg.pop("execution_hosts", None)
    monuseg.pop("asset_additions", None)
    monuseg.pop("supersedes", None)
    monuseg.pop("superseded_split", None)
    monuseg.pop("external_sources_changed_since_legacy", None)
    monuseg.pop("checkpoint_selection_rule_20tb", None)
    monuseg.update(
        protocol_id="data-quality-monuseg-train30val7-test14-v1",
        explicit_user_authorization="2026-10-02 MoNuSeg 30/7/14 ID evaluation",
        checkpoint_assets=[monuseg_asset],
        training_roots={label: str(train)},
        online_checkpoints=False, legacy_reuse=False,
        created_unix=time.time(),
        tasks=[dict(key=f"{label}_ck975__segmentation__monuseg__primary-last__{spec['split']}",
                    asset=monuseg_asset, dataset=spec)],
    )
    monuseg["external_source_hashes"][str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    save(root / "monuseg/campaign_manifest.json", monuseg)
    result = dict(arm=arm, checkpoint=str(checkpoint),
                  checkpoint_sha256=audit["checkpoint_sha256"],
                  training_audit=str(train / "audit.json"),
                  shared_tasks=len(shared["tasks"]), extension_tasks=len(tasks),
                  monuseg_tasks=1, monuseg_split=spec["split_protocol_id"],
                  monuseg_counts=spec["counts"], prepared_unix=time.time())
    save(root / "prepared.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.arm), indent=2))


if __name__ == "__main__":
    main()
