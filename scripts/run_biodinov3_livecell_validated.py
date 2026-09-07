#!/usr/bin/env python3
"""Run LIVECell Full-FT smoke, then formal seed-0 only if memory-safe."""
from __future__ import annotations
import json, os, shlex, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/instance_seg_tuning/biodinov3_validated_run"
LOCK = OUT / ".livecell_validated.task.lock"
HOST = "bbnc@172.16.1.206"; GPU = "3"
UUID = "GPU-299e3bae-f266-1f85-1709-b99de36b3ad5"
PY = "/home/bbnc/anaconda3/envs/dinov3/bin/python"

def eligible() -> bool:
    q = "nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits"
    a = "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits"
    rows = subprocess.check_output(["ssh", "-o", "BatchMode=yes", HOST, q], text=True)
    apps = subprocess.check_output(["ssh", "-o", "BatchMode=yes", HOST, a], text=True)
    active = {line.split(",")[0].strip() for line in apps.splitlines() if line.strip()}
    for line in rows.splitlines():
        idx, uuid, free, util = [x.strip() for x in line.split(",")]
        if idx == GPU:
            return uuid == UUID and int(free) >= 22000 and int(util) == 0 and uuid not in active
    return False

def execute(name: str, extra: list[str]) -> tuple[int, int, Path]:
    log = OUT / "logs" / f"{name}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"/tmp/{name}"
    common = [PY, "-u", "-m", "dinov3.eval.bio_segmentation.instance_seg.train",
      "--dataset", "livecell", "--data-root", "/mnt/huawei_deepcad/benchmark/segmentation/LIVECell",
      "--checkpoint", "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/ckpt/100/checkpoint.pth",
      "--train-config", "outputs/01_training_runs/hplus_s6_e15_nosigreg_alpha1_20260812/config.yaml",
      "--layers", "7", "15", "23", "31", "--finetune", "--batch-size", "1", "--grad-accum-steps", "8",
      "--crop-size", "256", "--stride", "192", "--lr", "1e-3", "--backbone-lr", "2e-5",
      "--weight-decay", "1e-4", "--amp-dtype", "bf16", "--feature-size", "32", "--embed-proj", "384",
      "--fusion-mode", "bucket_concat", "--decoder-variant", "current", "--num-workers", "0", "--seed", "0",
      "--aug", "strong", "--mosaic-prob", "0.3", "--np-loss-mode", "ce_dice",
      "--fg-thresh", "0.5", "--energy-thresh", "0.4", "--skip-test-eval"] + extra
    remote = f"cd /mnt/huawei_deepcad/dinov3 && mkdir -p {tmp} && CUDA_VISIBLE_DEVICES={GPU} PYTHONUNBUFFERED=1 TMPDIR={tmp} TMP={tmp} TEMP={tmp} {shlex.join(common)}"
    with log.open("w") as h:
        h.write(f"GPU={GPU} UUID={UUID}\nCMD={remote}\n"); h.flush()
        p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", HOST, remote], stdout=h, stderr=subprocess.STDOUT)
        h.write(f"SSH_PID={p.pid}\n"); h.flush(); rc = p.wait()
    return rc, p.pid, log

if LOCK.exists(): raise SystemExit(f"active atomic lock: {LOCK}")
LOCK.mkdir(parents=True); (LOCK / "owner.json").write_text(json.dumps({"pid": os.getpid(), "started": time.time(), "gpu": GPU, "uuid": UUID}) + "\n")
status = {"gpu": GPU, "uuid": UUID}
try:
    if not eligible(): raise SystemExit("GPU 3 no longer eligible before LIVECell smoke")
    smoke_out = OUT / "livecell_fullft_smoke"
    rc, pid, log = execute("livecell_fullft_smoke_seed0", ["--output-dir", str(smoke_out), "--verify-backbone-grad", "--epochs", "1", "--eval-every", "1", "--max-train-batches", "1", "--max-eval-images", "1"])
    status.update({"smoke_rc": rc, "smoke_ssh_pid": pid, "smoke_log": str(log)})
    if rc: raise SystemExit(rc)
    result = json.loads((smoke_out / "results.json").read_text())
    peak = result["_meta"]["peak_cuda_memory_gib"]
    grad = result["_meta"]["backbone_gradient_l1"]
    status.update({"smoke_peak_cuda_gib": peak, "backbone_gradient_l1": grad})
    if peak >= 22.0: status["status"] = "BLOCKED_BY_MEMORY"; raise SystemExit(0)
    if not eligible(): status["status"] = "BLOCKED_BY_GPU_BUSY_AFTER_SMOKE"; raise SystemExit(0)
    formal = OUT / "livecell_fullft_50ep"
    rc, pid, log = execute("livecell_fullft_50ep_seed0", ["--output-dir", str(formal), "--epochs", "50", "--eval-every", "10"])
    status.update({"formal_rc": rc, "formal_ssh_pid": pid, "formal_log": str(log), "status": "completed" if rc == 0 else f"failed_rc_{rc}"})
finally:
    (OUT / "livecell_validated_launch.json").write_text(json.dumps(status, indent=2) + "\n")
    for child in LOCK.iterdir(): child.unlink()
    LOCK.rmdir()
