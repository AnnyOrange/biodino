"""
Matrix orchestrator: run DINOHoVerNet (frozen) for {checkpoints} x {datasets}
across a GPU fleet with N processes per GPU (slot = gpu repeated N times).

Packing 2/GPU overlaps one job's GPU-idle eval (CPU-bound instance metrics) with
another job's training. Heavy datasets (livecell/monuseg) are mixed with light
256² ones to balance.

Run on deepcad:
  cd /mnt/huawei_deepcad/dinov3 && \
  ~/anaconda3/envs/dinov3/bin/python -m dinov3.eval.bio_segmentation.instance_seg.scripts._run_matrix \
    --datasets tissuenet conic bbbc038 livecell monuseg --gpus 1,2,3,4,5,6,7 --procs-per-gpu 2
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
SEG = "/mnt/huawei_deepcad/benchmark/segmentation"
CONFIG = "config.yaml"

GENERIC = "/mnt/huawei_deepcad/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"

CKPTS = {
    "bio_12299": "ckpt/12299/checkpoint.pth",
    "generic_lvd": GENERIC,
}


def all_ckpts(include_generic: bool = True) -> dict:
    """All bio H+ checkpoints in ckpt/ (+ generic), for per-dataset best-ckpt search."""
    out = {}
    root = os.path.join(ROOT, "ckpt")
    for it in sorted((int(d) for d in os.listdir(root) if d.isdigit()
                      and os.path.exists(os.path.join(root, d, "checkpoint.pth")))):
        out[f"bio_{it}"] = os.path.join("ckpt", str(it), "checkpoint.pth")
    if include_generic:
        out["generic_lvd"] = GENERIC
    return out

# per-dataset: (epochs, crop, stride, max_eval_images). Heavy/native datasets get
# fewer eval images (slow, crowded) and overlapping tiles.
DSCFG = {
    "tissuenet": (25, 256, 256, 300),
    "conic":     (25, 256, 256, 300),
    "bbbc038":   (25, 256, 192, 200),
    "livecell":  (25, 256, 192, 120),
    "monuseg":   (25, 256, 192, 60),
    "pannuke":   (25, 256, 256, 300),
}


def data_root(ds):
    return os.path.join(SEG, "LIVECell") if ds == "livecell" else os.path.join(SEG, ds, "extracted")


def make_cmd(ds, name, ckpt, gpu, out_dir, batch):
    ep, crop, stride, neval = DSCFG[ds]
    return [
        PY, "-m", "dinov3.eval.bio_segmentation.instance_seg.train",
        "--dataset", ds, "--data-root", data_root(ds),
        "--checkpoint", ckpt, "--train-config", CONFIG, "--output-dir", out_dir,
        "--layers", "7", "15", "23", "31", "--freeze-backbone",
        "--feature-size", "64", "--embed-proj", "512",
        "--epochs", str(ep), "--batch-size", str(batch),
        "--crop-size", str(crop), "--stride", str(stride),
        "--eval-every", "5", "--max-eval-images", str(neval),
        "--skip-test-eval", "--num-workers", "5",
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--gpus", default="1,2,3,4,5,6,7")
    p.add_argument("--procs-per-gpu", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out-root", default="outputs/instance_seg/matrix")
    p.add_argument("--all-ckpts", action="store_true",
                   help="Sweep ALL bio H+ checkpoints (+ generic) per dataset to find the best.")
    args = p.parse_args()

    global CKPTS
    if args.all_ckpts:
        CKPTS = all_ckpts(include_generic=True)

    os.chdir(ROOT)
    gpus = [g for g in args.gpus.split(",") if g]
    slots = gpus * args.procs_per_gpu  # e.g. 7 gpus x2 = 14 slots
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Heaviest datasets first so they start early; interleave checkpoints.
    order = sorted(args.datasets, key=lambda d: -DSCFG[d][3])  # actually crowd, but fine
    jobs = []
    skipped = 0
    for ds in args.datasets:
        for name, ck in CKPTS.items():
            if (out_root / ds / name / "results.json").exists():
                skipped += 1
                continue
            jobs.append((ds, name, ck))
    print(f"[matrix] {len(jobs)} jobs to run ({skipped} already done), slots={slots}", flush=True)

    pending = list(jobs)
    running = {}   # slot_id -> (proc, job, logf, gpu)
    free = list(range(len(slots)))
    results = {}

    while pending or running:
        while free and pending:
            sid = free.pop(0)
            gpu = slots[sid]
            ds, name, ck = pending.pop(0)
            out_dir = str(out_root / ds / name)
            os.makedirs(out_dir, exist_ok=True)
            logf = open(out_root / f"{ds}__{name}.log", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            proc = subprocess.Popen(make_cmd(ds, name, ck, gpu, out_dir, args.batch_size),
                                    env=env, stdout=logf, stderr=subprocess.STDOUT)
            running[sid] = (proc, (ds, name), logf, gpu)
            print(f"[matrix] launch {ds}/{name} on GPU{gpu} slot{sid} pid {proc.pid}", flush=True)
        time.sleep(15)
        for sid, (proc, (ds, name), logf, gpu) in list(running.items()):
            if proc.poll() is None:
                continue
            logf.close()
            del running[sid]
            free.append(sid)
            rj = out_root / ds / name / "results.json"
            val = json.load(open(rj)).get("val") if rj.exists() else None
            results[f"{ds}/{name}"] = val
            print(f"[matrix] done {ds}/{name} GPU{gpu}: {'ok' if val else 'FAILED '+str(proc.returncode)}", flush=True)

    print("\n================ MATRIX RESULTS (val) ================", flush=True)
    for ds in args.datasets:
        print(f"\n[{ds}]", flush=True)
        for name in CKPTS:
            v = results.get(f"{ds}/{name}")
            if v:
                print(f"  {name:<12} AJI={v.get('AJI',float('nan')):.4f} bPQ={v.get('bPQ',float('nan')):.4f} "
                      f"AP50={v.get('AP50',float('nan')):.4f} mPQ={v.get('mPQ',float('nan')):.4f}", flush=True)
            else:
                print(f"  {name:<12} FAILED", flush=True)
    json.dump(results, open(out_root / "matrix_leaderboard.json", "w"), indent=2)
    print(f"\n[matrix] saved {out_root/'matrix_leaderboard.json'}", flush=True)


if __name__ == "__main__":
    main()
