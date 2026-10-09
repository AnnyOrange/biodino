#!/usr/bin/env python3
"""MoNuSeg train30 / extra7 val / test14 re-test on the machine that holds the checkpoints.

User instruction 2026-09-30: checkpoints that live only on their training machine (HXW, lyx-xr,
suxin-8H100-1) are tested there instead of being transferred.  Each site runs its own local
campaign with a copy of the derived evaluator snapshot dinov3_monuseg_train30val7_snapshot_20260929
(same file hashes, verified by the queue worker) and a site-local copy of the MoNuSeg files.
prepare refuses to start unless the site data reproduce the split identity of the main campaign
(monuseg_train30val7_test14_retest_20260929), so results are merged as the same protocol.
The preflight split identity hashes absolute image paths, so it necessarily differs per site; the
site check compares benchmark-relative split membership plus the SHA-256 of every image and mask
against a reference produced from /mnt/huawei_deepcad/benchmark with the same code (mode reference).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
SPLIT_IDENTITY = "7935be272a1feb009fae136264810921147887b30309e0e09b3f50873dd9982c"
SPLIT_PROTOCOL = "monuseg2018-train30-extra7val-test14-v1"
MAIN_CAMPAIGN = "/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/monuseg_train30val7_test14_retest_20260929"
MANIFEST_NAME = "monuseg2018_train30_extra7val_test14_manifest.json"


def load_site(args) -> dict:
    site = json.loads(Path(args.site).read_text())
    for key in ("name", "snapshot", "benchmark", "output", "gpus", "assets"):
        if key not in site:
            raise KeyError(f"site config lacks {key}")
    return site


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fleet_module(site: dict):
    os.environ["DINOV3_CODE_ROOT"] = site["snapshot"]
    sys.path.insert(0, str(Path(site["snapshot"]) / "scripts"))
    import run_retest_fleet_20260918 as fleet  # noqa: E402  (derived snapshot copy, hash-verified)
    return fleet


def membership(site: dict) -> dict:
    """Benchmark-relative split membership and content hashes of every MoNuSeg image and mask."""
    from dinov3.eval.bio_frozen_eval.external_fm_protocol import dense_recipe
    from dinov3.eval.bio_segmentation.feature_extractor import _build_dataset
    from dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline import _resolve_data_root
    benchmark = Path(site["benchmark"]).resolve()
    recipe = dense_recipe("dinov2", "monuseg", split_protocol=None, observation=None)
    root = _resolve_data_root(benchmark / "segmentation", "monuseg")
    out = {}
    for split in ("train", "val", "test"):
        ds = _build_dataset("monuseg", str(root), split, recipe["image_size"], recipe["resize_mode"],
                            augment=False, do_normalize=False, dataset_split_protocol=recipe["split_protocol"])
        out[split] = [[str(Path(p).resolve().relative_to(benchmark)), sha256(Path(p)),
                       str(Path(m).resolve().relative_to(benchmark)), sha256(Path(m))]
                      for p, m in zip(ds.img_paths, ds.mask_paths)]
    return out


def reference(args) -> None:
    site = load_site(args)
    fleet_module(site)
    Path(args.out).write_text(json.dumps(membership(site), indent=1) + "\n")
    print("reference", args.out, sha256(Path(args.out)), flush=True)


def prepare(args) -> None:
    site = load_site(args)
    output, snapshot, benchmark = Path(site["output"]), Path(site["snapshot"]), Path(site["benchmark"])
    if (output / "campaign_manifest.json").exists():
        raise FileExistsError("Existing campaign; refusing overwrite")
    fleet = fleet_module(site)
    fleet.queue.verify_campaign_source(dict(source_snapshot=json.loads((snapshot / "source_snapshot.json").read_text())))
    from benchmark_eval.rules_dense_preflight import audit_dense
    if Path(audit_dense.__code__.co_filename).resolve().parents[2] != (snapshot / "evaluation_external").resolve():
        raise RuntimeError("dense preflight not imported from the snapshot copy")
    spec = audit_dense("monuseg", benchmark)
    if spec["counts"] != {"train": 30, "val": 7, "test": 14}:
        raise RuntimeError(f"unexpected MoNuSeg counts {spec['counts']}")
    ref_path = Path(site["reference"])
    if membership(site) != json.loads(ref_path.read_text()):
        raise RuntimeError("site MoNuSeg membership/content differs from the main-campaign reference")
    spec.update(comparison_view="primary-last", reserve_mib=16000 if spec["image_size"] >= 512 else 8192,
                component="segmentation/monuseg", split_protocol_id=SPLIT_PROTOCOL)
    assets = site["assets"]
    missing = [a["path"] for a in assets if not Path(a["path"]).exists()]
    if missing:
        raise FileNotFoundError(f"missing assets: {missing[:3]}")
    source = json.loads((snapshot / "source_snapshot.json").read_text())
    external = {str(p): sha256(p) for p in (HERE, snapshot / "scripts/run_retest_fleet_20260918.py",
                                            benchmark / "segmentation/monuseg/extracted" / MANIFEST_NAME)}
    manifest = dict(
        protocol_id="monuseg2018-train30-extra7val-test14-retest-v1",
        explicit_user_authorization="2026-09-29: re-test all MoNuSeg with the official 30/7/14 split; "
                                    "2026-09-30: test checkpoints on their original machine",
        site=site["name"], main_campaign=MAIN_CAMPAIGN, main_campaign_split_identity=SPLIT_IDENTITY,
        site_split_identity_note="absolute-path identity; equivalence proven by site_membership_reference",
        site_membership_reference=dict(path=str(ref_path), sha256=sha256(ref_path), matched=True),
        git_commit=source["git_commit"], source_snapshot=source, source_snapshot_path=str(snapshot),
        numerical_environment=fleet.NUMERICAL, external_source_hashes=external,
        checkpoint_assets=assets, datasets=[spec], benchmark_root=str(benchmark), batch_size=64,
        autocast_dtype="bf16", seed=0, num_workers=2, full_v3_aggregate_allowed=False, legacy_reuse=False,
        online_checkpoints=False, isolate_runtime_failures=True, no_checkpoint_or_data_transfers=True,
        created_unix=time.time())
    manifest["tasks"] = fleet.tasks_for(assets, [spec])
    output.mkdir(parents=True, exist_ok=True)
    fleet.queue.save(output / "campaign_manifest.json", manifest)
    (output / "_state/inputs").mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(output=str(output), tasks=len(manifest["tasks"]),
                          split_identity=spec["split_identity_sha256"][:12])), flush=True)


def worker(args) -> None:
    site = load_site(args)
    fleet = fleet_module(site)
    gpus = [int(g) for g in site["gpus"]]
    ns = argparse.Namespace(output=Path(site["output"]), host=site["name"], gpus=gpus, target_per_gpu=1,
                            max_host_jobs=len(gpus), max_global_jobs=400, task_family="segmentation",
                            admission_guard=lambda gpu, actual, task: task["dataset"]["task"] == "segmentation")
    fleet.worker(ns)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("reference", "prepare", "worker"))
    p.add_argument("--site", required=True)
    p.add_argument("--out", help="reference mode: output JSON")
    a = p.parse_args()
    {"reference": reference, "prepare": prepare, "worker": worker}[a.mode](a)
