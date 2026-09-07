#!/usr/bin/env python3
"""Run the validated BioDINO post-processing checks on an audited worker.

This is evaluation-only: it loads each completed Frozen BioDINO decoder head,
uses the already confirmed ViT-L/16 threshold pair for that dataset, and never
searches thresholds or enables TTA.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "bbnc@172.16.1.206"
REMOTE_ROOT = "/mnt/huawei_deepcad/dinov3"
PYTHON = "/home/bbnc/anaconda3/envs/dinov3/bin/python"
CHECKPOINT = "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/ckpt/100/checkpoint.pth"
CONFIG = "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/config.yaml"
RUN = ROOT / "outputs/instance_seg_tuning/biodinov3_validated_run"
DATASETS = {
    "bbbc038": {"gpu": 3, "fg": 0.46, "energy": 0.58, "metric": "ObjectAP"},
    "tissuenet": {"gpu": 5, "fg": 0.57, "energy": 0.40, "metric": "ObjectAP"},
    "pannuke": {"gpu": 4, "fg": 0.46, "energy": 0.45, "metric": "bPQ"},
    "conic": {"gpu": 3, "fg": 0.46, "energy": 0.43, "metric": "mPQ"},
}


def data_root(dataset: str) -> str:
    return "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell" if dataset == "livecell" else f"/mnt/huawei_deepcad/benchmark/segmentation/{dataset}/extracted"


def remote_eval(dataset: str, cfg: dict) -> int:
    out = RUN / "quick_post" / dataset
    out.mkdir(parents=True, exist_ok=True)
    head = ROOT / "outputs/instance_seg_tuning/biodinov3_seven_dataset_results_run" / dataset / "no_trick_seed0" / "best_head.pth"
    result = out / "results.json"
    log = RUN / "logs" / f"{dataset}_quick_post.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"/tmp/bio_validated_{dataset}"
    # Use the package entrypoint so eval_full.py's relative imports resolve.
    cmd = [PYTHON, "-u", "-m", "dinov3.eval.bio_segmentation.instance_seg.eval_full",
           "--dataset", dataset, "--data-root", data_root(dataset),
           "--checkpoint", CHECKPOINT, "--train-config", CONFIG,
           "--head-path", str(head), "--checkpoint-kind", "decoder",
           "--output", str(result), "--layers", "7", "15", "23", "31",
           "--feature-size", "32", "--embed-proj", "384", "--split", "val",
           "--crop-size", "256", "--stride", "192", "--fg-thresh", str(cfg["fg"]),
           "--energy-thresh", str(cfg["energy"])]
    remote = f"cd {shlex.quote(REMOTE_ROOT)} && mkdir -p {shlex.quote(tmp)} && CUDA_VISIBLE_DEVICES={cfg['gpu']} PYTHONUNBUFFERED=1 TMPDIR={shlex.quote(tmp)} TMP={shlex.quote(tmp)} TEMP={shlex.quote(tmp)} {shlex.join(cmd)}"
    with log.open("w") as handle:
        handle.write(f"GPU={cfg['gpu']} dataset={dataset} fg={cfg['fg']} energy={cfg['energy']}\n")
        handle.write(f"HEAD={head}\nCMD=ssh {HOST} {shlex.quote(remote)}\n")
        handle.flush()
        proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", HOST, remote], stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"SSH_PID={proc.pid}\n"); handle.flush()
        rc = proc.wait()
    return rc if result.is_file() else (rc or 1)


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    lock = RUN / ".task.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit(f"active atomic lock: {lock}")
    (lock / "owner.json").write_text(json.dumps({"pid": os.getpid(), "started": time.time(), "kind": "quick_post"}) + "\n")
    try:
        # GPU 3 is intentionally assigned to one task at a time; BBBC038 runs
        # there first and CoNIC is launched only after its SSH process exits.
        first = ("bbbc038", "tissuenet", "pannuke")
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {pool.submit(remote_eval, d, DATASETS[d]): d for d in first}
            for fut in as_completed(jobs):
                dataset = jobs[fut]
                rc = fut.result()
                print(f"{dataset}: {'completed' if rc == 0 else f'failed rc={rc}'}", flush=True)
        for dataset in ("conic",):
            rc = remote_eval(dataset, DATASETS[dataset])
            if rc:
                print(f"{dataset}: failed rc={rc}", flush=True)
            else:
                print(f"{dataset}: completed", flush=True)
        return 0
    finally:
        try: (lock / "owner.json").unlink()
        except FileNotFoundError: pass
        try: lock.rmdir()
        except OSError: pass


if __name__ == "__main__":
    raise SystemExit(main())
