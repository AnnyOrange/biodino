#!/usr/bin/env python3
"""Run the remaining BioDINO validated seed-0 jobs with an atomic lock.

GPU assignment is made only after an audit of the requested 3090-qi worker.
The two Full-FT jobs are one-batch memory/gradient smokes; they never alter
the input protocol and are marked blocked on OOM rather than retried with a
different method or resolution.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "bbnc@172.16.1.206"
REMOTE_ROOT = "/mnt/huawei_deepcad/dinov3"
PYTHON = "/home/bbnc/anaconda3/envs/dinov3/bin/python"
CKPT = "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/ckpt/100/checkpoint.pth"
CFG = "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/config.yaml"
OUT = ROOT / "outputs/instance_seg_tuning/biodinov3_validated_run"


def audit() -> list[dict[str, str]]:
    q = "nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits"
    apps = "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits"
    gpu = subprocess.check_output(["ssh", "-o", "BatchMode=yes", REMOTE, q], text=True)
    active = subprocess.check_output(["ssh", "-o", "BatchMode=yes", REMOTE, apps], text=True)
    pids = {line.split(",")[0].strip() for line in active.splitlines() if line.strip()}
    rows = []
    for line in gpu.splitlines():
        vals = [x.strip() for x in line.split(",")]
        if len(vals) != 4:
            continue
        idx, uuid, free, util = vals
        rows.append({"index": idx, "uuid": uuid, "free": free, "util": util,
                     "eligible": int(free) >= 22000 and int(util) == 0 and uuid not in pids})
    return rows


def run_remote(name: str, gpu: dict[str, str], args: list[str]) -> dict[str, str]:
    log = OUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"/tmp/bio_validated_{name}"
    remote = (f"cd {REMOTE_ROOT} && mkdir -p {tmp} && "
              f"CUDA_VISIBLE_DEVICES={gpu['index']} PYTHONUNBUFFERED=1 "
              f"TMPDIR={tmp} TMP={tmp} TEMP={tmp} {shlex.join(args)}")
    with log.open("w") as handle:
        handle.write(f"GPU={gpu['index']} UUID={gpu['uuid']}\nCMD={remote}\n")
        proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", REMOTE, remote],
                                stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"PID={proc.pid}\n"); handle.flush(); rc = proc.wait()
    return {"name": name, "gpu": gpu["index"], "uuid": gpu["uuid"],
            "pid": str(proc.pid), "rc": str(rc), "log": str(log)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / ".remaining.task.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit(f"active atomic lock: {lock}")
    (lock / "owner.json").write_text(json.dumps({"pid": os.getpid(), "started": time.time()}) + "\n")
    try:
        rows = audit()
        eligible = [r for r in rows if r["eligible"]]
        if len(eligible) < 2:
            raise SystemExit(f"need 2 eligible GPUs for MoNuSeg plus sequential smokes; audit={rows}")
        selected = eligible[:2]
        (OUT / "remaining_audit.json").write_text(json.dumps({"audit": rows, "selected": selected}, indent=2) + "\n")
        common = [PYTHON, "-u", "-m", "dinov3.eval.bio_segmentation.instance_seg.train",
                  "--checkpoint", CKPT, "--train-config", CFG, "--layers", "7", "15", "23", "31",
                  "--batch-size", "1", "--grad-accum-steps", "8", "--crop-size", "256", "--stride", "192",
                  "--lr", "1e-3", "--weight-decay", "1e-4", "--amp-dtype", "bf16", "--feature-size", "32",
                  "--embed-proj", "384", "--fusion-mode", "bucket_concat", "--decoder-variant", "current",
                  "--num-workers", "0", "--seed", "0", "--aug", "strong", "--mosaic-prob", "0.3",
                  "--np-loss-mode", "ce_dice", "--fg-thresh", "0.5", "--energy-thresh", "0.4",
                  "--skip-test-eval"]
        jobs = [
            ("monuseg_frozen_50ep_seed0", selected[0], common + ["--dataset", "monuseg",
             "--data-root", "/mnt/huawei_deepcad/benchmark/segmentation/monuseg/extracted",
             "--output-dir", str(OUT / "monuseg_50ep"), "--freeze-backbone", "--epochs", "50", "--eval-every", "10"]),
            ("cellpose_fullft_smoke_seed0", selected[1], common + ["--dataset", "cellpose",
             "--data-root", "/mnt/huawei_deepcad/benchmark/segmentation/cellpose/extracted",
             "--output-dir", str(OUT / "cellpose_fullft_smoke"), "--finetune", "--verify-backbone-grad",
             "--epochs", "1", "--eval-every", "1", "--max-train-batches", "1", "--max-eval-images", "1"]),
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = {pool.submit(run_remote, n, g, a): n for n, g, a in jobs}
            results = [f.result() for f in as_completed(futs)]
        # Cellpose smoke has released GPU 5. Re-audit before assigning LIVECell
        # so a newly arrived external process is never overlapped.
        rows_after = audit()
        smoke_gpu = next((r for r in rows_after if r["uuid"] == selected[1]["uuid"]), None)
        if smoke_gpu is not None and smoke_gpu["eligible"]:
            live_args = common + ["--dataset", "livecell",
                "--data-root", "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell",
                "--output-dir", str(OUT / "livecell_fullft_smoke"), "--finetune",
                "--verify-backbone-grad", "--epochs", "1", "--eval-every", "1",
                "--max-train-batches", "1", "--max-eval-images", "1"]
            results.append(run_remote("livecell_fullft_smoke_seed0", smoke_gpu, live_args))
        else:
            results.append({"name": "livecell_fullft_smoke_seed0", "rc": "not_started_gpu_no_longer_eligible",
                            "audit": rows_after})
        (OUT / "remaining_launch_results.json").write_text(json.dumps(results, indent=2) + "\n")
        return 0
    finally:
        try: (lock / "owner.json").unlink()
        except FileNotFoundError: pass
        try: lock.rmdir()
        except OSError: pass


if __name__ == "__main__":
    raise SystemExit(main())
