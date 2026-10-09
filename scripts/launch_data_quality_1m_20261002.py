#!/usr/bin/env python3
"""Launch a matched, finite one-pass HS6 ViT-L quality comparison arm."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "plot/fig2/data_quality"
BASE = ROOT / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_seed0_20260818/config.yaml"
ARMS = {
    "1tb": ["/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*.tar"],
    "5tb": [
        "/mnt/huawei_blm/deepcad_5t_v1/quality_1m_mix30_70_20260929/old_1tb_300k/filtered_mixed_train*.tar",
        "/mnt/huawei_blm/deepcad_5t_v1/quality_1m_mix30_70_20260929/new_4tb_700k/filtered_mixed_train*.tar",
    ],
    "20tb": ["/mnt/huawei_blm/deepcad_20tb_quality_1m_20261002/filtered_mixed_train_*.tar"],
    "20tb_route2_ddp": ["/mnt/huawei_blm/deepcad_20tb_quality_1m_20261002/filtered_mixed_train_*.tar"],
    "20tb_route1": [
        "/mnt/huawei_blm/deepcad_20tb_route1_quality_1m_20261006/packed/filtered_projection_20TB_nested-r*.tar",
        "/mnt/huawei_blm/deepcad_20tb_route1_quality_1m_20261006/backfill_final/filtered_projection_20TB_nested-r*.tar",
    ],
    "20tb_route1_ddp": [
        "/mnt/huawei_blm/deepcad_20tb_route1_quality_1m_20261006/packed/filtered_projection_20TB_nested-r*.tar",
        "/mnt/huawei_blm/deepcad_20tb_route1_quality_1m_20261006/backfill_final/filtered_projection_20TB_nested-r*.tar",
    ],
    "100tb": ["/mnt/huawei_blm/random_1pb_100tb_global_v3/100tb_final_1m_repaired_20260928/filtered_mixed_train*.tar"],
    "1pb": ["/mnt/huawei_blm/random_1pb_100tb_global_v3/1pb_final_1m_mounted_20260928/filtered_mixed_train*.tar"],
}
EXPECTED_SHARDS = {"1tb": 326, "5tb": 500, "20tb": 500, "20tb_route2_ddp": 500, "100tb": 500, "1pb": 500}
PYTHON = os.environ.get("DQ_PYTHON", "/home/bbnc/anaconda3/envs/dinov3/bin/python")
UPDATES = 976
MICROBATCH = 8
ACCUM = 32
ROUTE1_ASSIGNMENT = Path("/mnt/huawei_blm/deepcad_20tb_route1_quality_1m_20261006/rank_shard_assignment.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--gpu-group", required=True, help="Four comma-separated GPU indices")
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    gpus = [x.strip() for x in args.gpu_group.split(",")]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("Exactly four distinct GPUs are required for effective batch 1024")
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    if args.arm in ("20tb", "20tb_route2_ddp", "20tb_route1", "20tb_route1_ddp"):
        sample_dir = ("deepcad_20tb_quality_1m_20261002" if args.arm in ("20tb", "20tb_route2_ddp")
                      else "deepcad_20tb_route1_quality_1m_20261006")
        completion = Path("/mnt/huawei_blm") / sample_dir / "extraction_complete.json"
        if not completion.is_file():
            raise FileNotFoundError(completion)
        record = json.loads(completion.read_text())
        expected_shards = EXPECTED_SHARDS.get(args.arm, record.get("shards"))
        if record.get("status") != "PASS" or record.get("samples") != 1_000_000 or \
                not isinstance(expected_shards, int) or expected_shards < 4 or record.get("shards") != expected_shards:
            raise ValueError(f"20TB extraction verification failed: {record}")
    shards = sorted(set(path for pattern in ARMS[args.arm] for path in glob.glob(pattern)))
    expected_shards = EXPECTED_SHARDS.get(args.arm, record["shards"] if args.arm.startswith("20tb_route1") else None)
    if len(shards) != expected_shards:
        raise ValueError(f"{args.arm}: expected {expected_shards} shards, found {len(shards)}")
    if any(Path(path).stat().st_size == 0 for path in shards):
        raise ValueError(f"{args.arm}: empty source tar")
    if args.arm == "20tb_route1_ddp" and not args.smoke:
        assignment = json.loads(ROUTE1_ASSIGNMENT.read_text())
        if assignment.get("status") != "PASS" or assignment.get("sample_count") != 1_000_000 or \
                min(assignment["rank_sample_counts"]) < 249_856 or \
                sorted(path for group in assignment["rank_shards"] for path in group) != shards:
            raise ValueError("Route-1 balanced shard assignment is invalid")

    run = OUT / ("smoke" if args.smoke else "training") / args.arm
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    overrides = {
        "train.dataset_path": "packwds_once_robust:" + ";".join(ARMS[args.arm]) + "::pct=1,99",
        "train.batch_size_per_gpu": MICROBATCH,
        "optim.gradient_accumulation_steps": ACCUM,
        "train.num_workers": 1,
        "train.wds_deterministic_resampling": True,
        "train.wds_shuffle_buffer": 50,
        "train.checkpointing": True,
        "train.checkpointing_full": True,
        "train.checkpointing_blocks": 24,
        "train.pin_memory": False,
        "train.prefetch_factor": 1,
        "train.compile": False,
        "train.OFFICIAL_EPOCH_LENGTH": UPDATES,
        "train.max_updates": 1 if args.smoke else None,
        "train.seed": 0,
        "optim.epochs": 1,
        "optim.warmup_epochs": 100 / UPDATES,
        "optim.freeze_last_layer_epochs": 50 / UPDATES,
        "teacher.warmup_teacher_temp_epochs": 100 / UPDATES,
        "evaluation.eval_period_iterations": 0 if args.smoke else UPDATES,
        "checkpointing.period": UPDATES,
        "checkpointing.max_to_keep": 2,
        "crops.rgb_mean": [0.5, 0.5, 0.5],
        "crops.rgb_std": [0.35, 0.35, 0.35],
    }
    if args.arm in ("20tb_route1_ddp", "20tb_route2_ddp"):
        overrides["compute_precision.distributed_mode"] = "ddp"

    def cli_value(value):
        return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))

    command = [
        PYTHON, "-m", "torch.distributed.run", "--nproc_per_node=4",
        f"--master_port={args.master_port}", "dinov3/train/train.py",
        "--config-file", str(BASE), "--output-dir", str(run), "--no-resume", "--seed", "0",
        *(f"{key}={cli_value(value)}" for key, value in overrides.items()),
    ]
    manifest = {
        "arm": args.arm,
        "source_patterns": ARMS[args.arm],
        "source_shards": len(shards),
        "source_inventory_sha256": hashlib.sha256(
            "\n".join(f"{path}:{Path(path).stat().st_size}" for path in shards).encode()
        ).hexdigest(),
        "base_config_sha256": sha256(BASE),
        "gpu_group": gpus,
        "effective_batch": len(gpus) * MICROBATCH * ACCUM,
        "optimizer_updates": 1 if args.smoke else UPDATES,
        "target_image_visits": (1 if args.smoke else UPDATES) * len(gpus) * MICROBATCH * ACCUM,
        "finite_loader": True,
        "distributed_mode": overrides.get("compute_precision.distributed_mode", "fsdp"),
        "shard_assignment": str(ROUTE1_ASSIGNMENT) if args.arm == "20tb_route1_ddp" else None,
        "command": command,
        "overrides": overrides,
        "created_unix": time.time(),
    }
    (run / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(gpus), PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", NCCL_IB_DISABLE="1", NCCL_P2P_DISABLE="1",
    )
    if args.arm == "20tb_route1_ddp" and ROUTE1_ASSIGNMENT.is_file():
        env["DQ_FINITE_SHARD_ASSIGNMENT"] = str(ROUTE1_ASSIGNMENT)
    with (run / "console.log").open("w", buffering=1) as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (run / "exit.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
    print(json.dumps({"arm": args.arm, "run": str(run), "returncode": result.returncode}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
