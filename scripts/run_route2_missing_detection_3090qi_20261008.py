#!/usr/bin/env python3
"""Run missing route2 detection-proxy cells on 3090-qi with audited GPU use."""

import csv
import fcntl
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "outputs/02_eval_runs" / os.environ.get(
    "ROUTE2_DETECTION_CAMPAIGN", "route2_missing_detection_3090qi_20261008")
CURVE = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/route2_20tb_v4_20261008/family_curve.csv"
RUN = ROOT / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_20tb_route2_mix009_021_0604_0096_8x5090zxr_20260924"
TEMPLATE = ROOT / "outputs/02_eval_runs/v2_full_v4_20261007/tasks/det__bbbc038__noGRAM20tb_ck47823.json"
SOURCE = ROOT / "outputs/02_eval_runs/hs6_l5_early_late_coexistence_v4_3090_20260921/source_snapshot_dense_v4"
DATASETS = ("bbbc038", "conic", "livecell")
GPUS = tuple(int(value) for value in os.environ.get("ROUTE2_DETECTION_GPUS", "0,1,2,3,4,5,6,7").split(","))
MAX_PER_GPU = int(os.environ.get("ROUTE2_DETECTION_MAX_PER_GPU", "5"))
RESERVE_MIB = int(os.environ.get("ROUTE2_DETECTION_RESERVE_MIB", "4200"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def gpu_memory():
    lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True).splitlines()
    return {int(parts[0]): (int(parts[1]), int(parts[2]))
            for line in lines if (parts := [int(x.strip()) for x in line.split(",")])}


def valid_result(path, dataset):
    try:
        result = json.loads(path.read_text())
        if any(result.get(key) != value for key, value in
               (("dataset", dataset), ("batch_size", 8), ("image_size", 224),
                ("epochs", 5), ("seed", 0))):
            return False
        if dataset == "conic" and result.get("conic_split_protocol") != "official-baseline-fold0-nested-v1":
            return False
        return math.isfinite(float(result["test_patch_f1"]))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def prepare():
    template = json.loads(TEMPLATE.read_text())
    assert Path(template["cmd"][2]) == ROOT / "outputs/02_eval_runs/v2_full_v4_20261007/runtime/run_detection_single_rank.py"
    assert template["cmd"][template["cmd"].index("--conic-split-protocol") + 1] == "official-baseline-fold0-nested-v1"
    assert template["cmd"][template["cmd"].index("--batch-size") + 1] == "8"
    if SOURCE != Path(template["cwd"]):
        raise ValueError("Evaluator snapshot changed")
    steps = sorted({int(row["checkpoint"]) for row in csv.DictReader(CURVE.open())
                    if row["family"] == "overall"}, reverse=True)
    assert len(steps) == 34
    tasks = []
    for step in steps:
        checkpoint = RUN / "eval" / f"training_{step}" / "teacher_checkpoint.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for dataset in DATASETS:
            key = f"route2_ck{step}__{dataset}"
            prior = ROOT / "outputs/02_eval_runs/v2_full_v4_20261007"
            prior_status = prior / "claims" / f"det__{dataset}__noGRAM20tb_ck{step}" / "status.json"
            prior_result = prior / "detection" / dataset / f"noGRAM20tb_ck{step}" / "results_bio_detection.json"
            if (prior_status.is_file() and json.loads(prior_status.read_text()).get("state") == "VALID_COMPLETE"
                    and valid_result(prior_result, dataset)):
                continue
            cell = OUT / "cells" / key
            cmd = list(template["cmd"])
            cmd[cmd.index("--checkpoint") + 1] = str(checkpoint)
            cmd[cmd.index("--dataset") + 1] = dataset
            cmd[cmd.index("--output-dir") + 1] = str(cell)
            tasks.append({"key": key, "checkpoint": str(checkpoint), "step": step,
                          "dataset": dataset, "command": cmd, "result": str(cell / "results_bio_detection.json")})
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "campaign_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["tasks"] != tasks:
            raise ValueError("Existing campaign tasks differ")
    else:
        save(manifest, {"host": "3090-qi", "protocol": "bio-eval-union-v4-detection-proxy-b8",
                        "source": str(SOURCE), "reference_task": str(TEMPLATE),
                        "datasets": DATASETS, "gpus": GPUS, "max_per_gpu": MAX_PER_GPU,
                        "target_min_memory_fraction": 0.5, "tasks": tasks})
    return tasks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = prepare()
        print("PENDING", len(tasks), flush=True)
        active = {}
        failed = set()
        while True:
            for key, (process, handle, gpu, task) in list(active.items()):
                if process.poll() is None:
                    continue
                handle.close()
                okay = process.returncode == 0 and valid_result(Path(task["result"]), task["dataset"])
                save(OUT / "status" / f"{key}.json", {"state": "VALID_COMPLETE" if okay else "FAILED",
                     "returncode": process.returncode, "gpu": gpu, "pid": process.pid, "finished": time.time()})
                if not okay:
                    failed.add(key)
                print("DONE" if okay else "FAILED", key, gpu, process.returncode, flush=True)
                del active[key]
            pending = [task for task in tasks if task["key"] not in active and task["key"] not in failed
                       and not valid_result(Path(task["result"]), task["dataset"])]
            cards = gpu_memory()
            save(OUT / "gpu_samples" / f"{int(time.time())}.json",
                 {"time": time.time(), "memory": cards,
                  "active": {key: {"gpu": job[2], "pid": job[0].pid} for key, job in active.items()}})
            if not pending and not active:
                break
            per_gpu = defaultdict(int)
            for _, _, gpu, _ in active.values():
                per_gpu[gpu] += 1
            for gpu in GPUS:
                if not pending or per_gpu[gpu] >= MAX_PER_GPU:
                    continue
                used, total = cards[gpu]
                waiting = sum(1 for _, _, card, _ in active.values() if card == gpu)
                if total - max(used, waiting * RESERVE_MIB) < RESERVE_MIB + 2048:
                    continue
                task = pending.pop(0)
                key = task["key"]
                cell = Path(task["result"]).parent
                cell.mkdir(parents=True, exist_ok=True)
                save(cell / "invocation.json", {**task, "host": "3090-qi", "gpu": gpu, "started": time.time()})
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(SOURCE),
                           DINOV3_ROOT=str(SOURCE), DINOV3_CODE_ROOT=str(SOURCE),
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                handle = (cell / "run.log").open("a")
                process = subprocess.Popen(task["command"], cwd=SOURCE, env=env,
                                           stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[key] = (process, handle, gpu, task)
                per_gpu[gpu] += 1
                print("START", key, "gpu", gpu, "pid", process.pid, flush=True)
            time.sleep(5)
        print("QUEUE_FINISHED", "failed", len(failed), flush=True)


if __name__ == "__main__":
    main()
