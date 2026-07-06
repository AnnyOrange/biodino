"""
Parallel sweep orchestrator for the instance-seg track.

Runs a frozen DINOHoVerNet for many backbones (all bio ckpt iterations + the
generic LVD baseline) across multiple GPUs with a concurrency cap, then prints a
leaderboard. Identical decoder/protocol for every row → score delta is purely the
backbone (and, for bio, the continual-training iteration).

Not wired into the package CLI on purpose — it's an experiment driver.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = "/mnt/huawei_deepcad/dinov3"
DATA = "/mnt/huawei_deepcad/benchmark/segmentation/pannuke/extracted"
CONFIG = "config.yaml"
GENERIC = "/mnt/huawei_deepcad/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"


def build_jobs(include_generic: bool):
    jobs = []
    ckpt_root = Path(ROOT) / "ckpt"
    iters = sorted(int(d.name) for d in ckpt_root.iterdir()
                   if d.is_dir() and d.name.isdigit() and (d / "checkpoint.pth").exists())
    for it in iters:
        jobs.append({"name": f"bio_{it}", "ckpt": str(ckpt_root / str(it) / "checkpoint.pth")})
    if include_generic:
        jobs.append({"name": "generic_lvd", "ckpt": GENERIC})
    return jobs


def make_cmd(job, args, gpu, out_dir):
    cmd = [
        PY, "-m", "dinov3.eval.bio_segmentation.instance_seg.train",
        "--dataset", "pannuke", "--data-root", DATA,
        "--checkpoint", job["ckpt"], "--train-config", CONFIG,
        "--output-dir", out_dir, "--layers", "7", "15", "23", "31",
        "--freeze-backbone", "--feature-size", str(args.feature_size),
        "--embed-proj", str(args.embed_proj), "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size), "--crop-size", "256", "--stride", "256",
        "--eval-every", str(args.eval_every), "--max-eval-images", str(args.max_eval_images),
        "--skip-test-eval", "--num-workers", "6",
    ]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--feature-size", type=int, default=64)
    p.add_argument("--embed-proj", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--max-eval-images", type=int, default=300)
    p.add_argument("--out-root", default="outputs/instance_seg/sweep")
    p.add_argument("--no-generic", action="store_true")
    args = p.parse_args()

    os.chdir(ROOT)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    jobs = build_jobs(include_generic=not args.no_generic)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] {len(jobs)} jobs on GPUs {gpus}: {[j['name'] for j in jobs]}", flush=True)

    pending = list(jobs)
    running = {}   # gpu -> (proc, job, logf)
    free = list(gpus)
    results = {}

    while pending or running:
        while free and pending:
            job = pending.pop(0)
            gpu = free.pop(0)
            out_dir = str(out_root / job["name"])
            os.makedirs(out_dir, exist_ok=True)
            logf = open(out_root / f"{job['name']}.log", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            proc = subprocess.Popen(make_cmd(job, args, gpu, out_dir), env=env,
                                    stdout=logf, stderr=subprocess.STDOUT)
            running[gpu] = (proc, job, logf)
            print(f"[sweep] launch {job['name']} on GPU {gpu} (pid {proc.pid})", flush=True)
        time.sleep(15)
        for gpu, (proc, job, logf) in list(running.items()):
            if proc.poll() is None:
                continue
            logf.close()
            del running[gpu]
            free.append(gpu)
            rj = out_root / job["name"] / "results.json"
            val = None
            if rj.exists():
                try:
                    val = json.load(open(rj)).get("val")
                except Exception:  # noqa: BLE001
                    val = None
            results[job["name"]] = val
            status = "ok" if val else f"FAILED(exit {proc.returncode})"
            print(f"[sweep] done {job['name']} on GPU {gpu}: {status}", flush=True)

    # Leaderboard
    rows = []
    for name, val in results.items():
        if val:
            rows.append((name, val.get("mPQ", float("nan")), val.get("AJI", float("nan")),
                         val.get("bPQ", float("nan")), val.get("AP50", float("nan"))))
        else:
            rows.append((name, float("nan"), float("nan"), float("nan"), float("nan")))
    rows.sort(key=lambda r: (-(r[1] if r[1] == r[1] else -1)))  # by mPQ desc, NaN last
    print("\n================ LEADERBOARD (val, sorted by mPQ) ================", flush=True)
    print(f"{'name':<16}{'mPQ':>8}{'AJI':>8}{'bPQ':>8}{'AP50':>8}", flush=True)
    for name, mpq, aji, bpq, ap50 in rows:
        print(f"{name:<16}{mpq:>8.4f}{aji:>8.4f}{bpq:>8.4f}{ap50:>8.4f}", flush=True)
    json.dump(results, open(out_root / "leaderboard.json", "w"), indent=2)
    print(f"\n[sweep] saved {out_root/'leaderboard.json'}", flush=True)


if __name__ == "__main__":
    main()
