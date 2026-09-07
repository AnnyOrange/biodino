#!/usr/bin/env python3
"""Run one locked MoNuSeg control/HV-auxiliary task on one remote GPU.

This worker writes only its task directory, task log, status file and TMPDIR.
It deliberately never touches shared CSV or summary files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "outputs" / "instance_seg_tuning"
PYTHON = Path("/home/bbnc/anaconda3/envs/dinov3/bin/python")
CHECKPOINT = ROOT / "outputs/01_training_runs/5tb_idweak10_vitl16_robust_b1024_8gpu/ckpt/15374/checkpoint.pth"
CONFIG = ROOT / "dinov3/configs/train/microscopy_continual_vitl16_robust_5tb_idweak10.yaml"
DATA_ROOT = Path("/mnt/huawei_deepcad/benchmark/segmentation/monuseg/extracted")


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, data: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".status.", dir=str(path.parent), text=True)
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def gpu_uuid(index: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", index],
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""


def epoch_from_log(log_path: Path) -> str:
    try:
        matches = re.findall(r"Epoch\s+(\d+)\s*/\s*30", log_path.read_text(errors="ignore"))
        return matches[-1] if matches else "0"
    except OSError:
        return "0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("task_id")
    parser.add_argument("method", choices=("control", "hv_auxiliary"))
    parser.add_argument("seed", type=int)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--host", default="3090-qi")
    args = parser.parse_args()

    task_dir = OUT / "distributed_runs" / args.run_id / args.host / args.task_id
    log_path = OUT / "distributed_logs" / args.run_id / args.host / f"{args.task_id}.log"
    tmp_dir = OUT / "tmp" / args.run_id / args.host / args.task_id
    locks = OUT / "distributed_runs" / args.run_id / ".task_locks"
    lock_dir = locks / f"{args.task_id}.lock"
    locks.mkdir(parents=True, exist_ok=True)
    try:
        lock_dir.mkdir()
    except FileExistsError:
        # A prior or live owner has already claimed this logical task.
        return 0

    task_dir.mkdir(parents=True, exist_ok=False)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    short_key = hashlib.sha1(f"{args.run_id}|{args.task_id}".encode()).hexdigest()[:20]
    tmp_link = ROOT / ".vitl16_tmp" / short_key
    tmp_link.parent.mkdir(parents=True, exist_ok=True)
    if tmp_link.exists() or tmp_link.is_symlink():
        if not tmp_link.is_symlink() or tmp_link.resolve() != tmp_dir.resolve():
            raise RuntimeError(f"TMPDIR symlink collision: {tmp_link}")
    else:
        tmp_link.symlink_to(tmp_dir, target_is_directory=True)

    uuid = gpu_uuid(args.gpu)
    def status(state: str, pid: int = 0, epoch: str = "0", exit_code: int | None = None,
               error: str = "", result_json: str = "") -> None:
        atomic_json(task_dir / "status.json", {
            "run_id": args.run_id,
            "screening": "ViT-L/16 strategy screening; not DINOv3-7B",
            "host": args.host,
            "task_id": args.task_id,
            "dataset": "monuseg",
            "method": args.method,
            "seed": args.seed,
            "gpu_index": args.gpu,
            "gpu_uuid": uuid,
            "state": state,
            "pid": pid,
            "epoch": epoch,
            "heartbeat": stamp(),
            "exit_code": exit_code,
            "result_json": result_json,
            "error": error,
        })

    atomic_json(lock_dir / "owner.json", {"task_id": args.task_id, "pid": os.getpid(), "claimed_at": stamp()})
    # NFS ownership is shared across users; copy bytes without attempting to
    # preserve the source mode/uid, which bbnc cannot apply to an inspur-owned
    # snapshot directory.
    with CONFIG.open("rb") as source, (task_dir / "config_snapshot.yaml").open("wb") as target:
        target.write(source.read())
    with (task_dir / "environment.txt").open("w") as env_file:
        env_file.write(f"host={args.host}\ngpu_index={args.gpu}\ngpu_uuid={uuid}\n")
        env_file.write(f"TMPDIR={tmp_link}\nTMPDIR_TARGET={tmp_dir}\n")
        subprocess.run(["/usr/bin/env"], stdout=env_file, check=False)
        subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free", "--format=csv", "-i", args.gpu], stdout=env_file, check=False)
        subprocess.run(["sha256sum", str(CHECKPOINT), str(CONFIG), str(ROOT / "dinov3/eval/bio_segmentation/instance_seg/train.py"), str(ROOT / "dinov3/eval/bio_segmentation/instance_seg/decoder.py"), str(ROOT / "dinov3/eval/bio_segmentation/instance_seg/losses.py")], stdout=env_file, check=False)

    command = [
        str(PYTHON), "-m", "dinov3.eval.bio_segmentation.instance_seg.train",
        "--dataset", "monuseg", "--data-root", str(DATA_ROOT),
        "--checkpoint", str(CHECKPOINT), "--train-config", str(CONFIG),
        "--output-dir", str(task_dir), "--layers", "4", "11", "17", "23",
        "--freeze-backbone", "--epochs", "30", "--batch-size", "8", "--grad-accum-steps", "1",
        "--crop-size", "256", "--stride", "192", "--lr", "1e-3", "--weight-decay", "1e-4",
        "--warmup-ratio", "0.05", "--grad-clip-norm", "1.0", "--layer-wise-lr-decay", "0.75",
        "--select-metric", "AJI", "--amp-dtype", "bf16", "--feature-size", "32", "--embed-proj", "384",
        "--fusion-mode", "bucket_concat", "--decoder-variant", "current", "--num-workers", "4", "--eval-every", "5",
        "--seed", str(args.seed), "--aug", "strong", "--mosaic-prob", "0.3", "--fg-thresh", "0.5", "--energy-thresh", "0.4",
        "--np-loss-mode", "ce_dice", "--skip-test-eval",
    ]
    if args.method == "hv_auxiliary":
        command += ["--hv-auxiliary", "--hv-aux-weight", "1.0", "--verify-hv-aux-grad"]
    (task_dir / "command.txt").write_text(" ".join(subprocess.list2cmdline([item]) for item in command) + "\n")

    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "TMPDIR": str(tmp_link), "TMP": str(tmp_link), "TEMP": str(tmp_link), "PYTHONUNBUFFERED": "1"})
    result_json = task_dir / "results.json"
    status("starting", result_json=str(result_json))
    with log_path.open("a", buffering=1) as log:
        log.write(f"START task={args.task_id} method={args.method} seed={args.seed} at {stamp()}\n")
        log.write(f"TMPDIR={tmp_link} target={tmp_dir}\n")
        log.write("COMMAND=" + " ".join(subprocess.list2cmdline([item]) for item in command) + "\n")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        status("running", process.pid, result_json=str(result_json))
        while process.poll() is None:
            status("running", process.pid, epoch_from_log(log_path), result_json=str(result_json))
            time.sleep(30)
        exit_code = process.wait()
        log.write(f"TRAINER_EXIT rc={exit_code} at {stamp()}\n")
        log_text = log_path.read_text(errors="ignore")
        valid = exit_code == 0 and result_json.is_file() and "Results saved" in log_text and bool(re.search(r"Epoch\s+30\s*/\s*30", log_text))
        error = ""
        try:
            result = json.loads(result_json.read_text())
            for metric in ("AJI", "Dice", "bPQ"):
                value = float(result["val"][metric])
                if not value == value:
                    raise ValueError(f"non-finite {metric}")
            if args.method == "hv_auxiliary" and float(result["_meta"].get("hv_aux_gradient_l1", 0.0)) <= 0.0:
                raise ValueError("missing nonzero hv_aux_gradient_l1")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            valid = False
            error = f"invalid_results:{exc}"
        # DataLoader workers on this NFS can emit a benign multiprocessing
        # finalizer traceback while the trainer continues normally.  Only an
        # unhandled traceback other than Errno 16 should invalidate a run.
        if "Traceback" in log_text and not re.search(r"OSError: \[Errno 16\] Device or resource busy", log_text):
            valid = False
            error = error or "unhandled_traceback"
        status("completed" if valid else "failed", process.pid, "30/30" if valid else epoch_from_log(log_path), 0 if valid else exit_code, error, str(result_json))
        log.write(f"RUN_EXIT state={'completed' if valid else 'failed'} rc={0 if valid else exit_code} at {stamp()}\n")
    return 0 if valid else (exit_code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
