#!/usr/bin/env python3
"""Launch the audited BioDINO Cellpose Full-FT seed-0 run once."""
from __future__ import annotations
import json, os, shlex, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/instance_seg_tuning/biodinov3_validated_run"
LOCK = OUT / ".cellpose_formal.task.lock"
HOST = "bbnc@172.16.1.206"
GPU = "5"
UUID = "GPU-539bdd9d-ad06-4bec-ac0d-59de9d5e5e6e"
PY = "/home/bbnc/anaconda3/envs/dinov3/bin/python"
remote_root = "/mnt/huawei_deepcad/dinov3"
tmp = "/tmp/bio_validated_cellpose_formal"
cmd = [PY, "-u", "-m", "dinov3.eval.bio_segmentation.instance_seg.train",
       "--dataset", "cellpose", "--data-root", "/mnt/huawei_deepcad/benchmark/segmentation/cellpose/extracted",
       "--checkpoint", "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/ckpt/100/checkpoint.pth",
       "--train-config", "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/config.yaml",
       "--output-dir", str(OUT / "cellpose_fullft_50ep"), "--layers", "7", "15", "23", "31",
       "--finetune", "--epochs", "50", "--batch-size", "1", "--grad-accum-steps", "8",
       "--crop-size", "256", "--stride", "192", "--lr", "1e-3", "--backbone-lr", "2e-5",
       "--weight-decay", "1e-4", "--amp-dtype", "bf16", "--feature-size", "32", "--embed-proj", "384",
       "--fusion-mode", "bucket_concat", "--decoder-variant", "current", "--num-workers", "0",
       "--eval-every", "10", "--seed", "0", "--aug", "strong", "--mosaic-prob", "0.3",
       "--np-loss-mode", "ce_dice", "--fg-thresh", "0.5", "--energy-thresh", "0.4", "--skip-test-eval"]
if LOCK.exists(): raise SystemExit(f"active atomic lock: {LOCK}")
LOCK.mkdir(parents=True)
(LOCK / "owner.json").write_text(json.dumps({"pid": os.getpid(), "started": time.time(), "gpu": GPU, "uuid": UUID}) + "\n")
log = OUT / "logs" / "cellpose_fullft_50ep_seed0.log"; log.parent.mkdir(parents=True, exist_ok=True)
remote = f"cd {remote_root} && mkdir -p {tmp} && CUDA_VISIBLE_DEVICES={GPU} PYTHONUNBUFFERED=1 TMPDIR={tmp} TMP={tmp} TEMP={tmp} {shlex.join(cmd)}"
with log.open("w") as h:
    h.write(f"GPU={GPU} UUID={UUID}\nCMD={remote}\n"); h.flush()
    p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", HOST, remote], stdout=h, stderr=subprocess.STDOUT)
    h.write(f"SSH_PID={p.pid}\n"); h.flush()
try:
    rc = p.wait()
    (OUT / "cellpose_formal_launch.json").write_text(json.dumps({"rc": rc, "pid": p.pid, "gpu": GPU, "uuid": UUID, "log": str(log)}, indent=2) + "\n")
finally:
    for child in LOCK.iterdir(): child.unlink()
    LOCK.rmdir()
raise SystemExit(rc)
