#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
DEFAULT_PYTHON = Path("/home/inspur/anaconda3/envs/dinov3/bin/python")
OUT = ROOT / "outputs" / "instance_seg_tuning"
CHECKPOINT = ROOT / "outputs/01_training_runs/5tb_idweak10_vitl16_robust_b1024_8gpu/ckpt/15374/checkpoint.pth"
CONFIG = ROOT / "dinov3/configs/train/microscopy_continual_vitl16_robust_5tb_idweak10.yaml"
TRAIN = ROOT / "dinov3/eval/bio_segmentation/instance_seg/train.py"

def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def atomic_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".status.", dir=str(path.parent), text=True)
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)

def gpu_uuid(index: str) -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", index],
        capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    return out[0].strip() if out else ""

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("host")
    p.add_argument("task_id")
    p.add_argument("dataset", choices=["livecell", "cellpose"])
    p.add_argument("mode", choices=["frozen", "finetune"])
    p.add_argument("seed", type=int)
    p.add_argument("--gpu", default="0")
    p.add_argument("--python", default=str(DEFAULT_PYTHON))
    args = p.parse_args()
    python = Path(args.python)

    task_dir = OUT / "distributed_runs" / args.run_id / args.host / args.task_id
    log_path = OUT / "distributed_logs" / args.run_id / args.host / f"{args.task_id}.log"
    tmp_dir = OUT / "tmp" / args.run_id / args.host / args.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # multiprocessing resource-sharer sockets have a short AF_UNIX path limit.
    # Keep the required per-task directory as the target, but expose it through
    # a short, unique project-local symlink for TMPDIR/TMP/TEMP.
    key = f"{args.run_id}|{args.host}|{args.task_id}".encode("utf-8")
    short_name = hashlib.sha1(key).hexdigest()[:20]
    # Keep the textual path well below AF_UNIX's ~108-byte socket limit.
    # The symlink target remains the required run/host/task-specific directory.
    tmp_link = ROOT / ".vitl16_tmp" / short_name
    tmp_link.parent.mkdir(parents=True, exist_ok=True)
    if tmp_link.exists() or tmp_link.is_symlink():
        if not tmp_link.is_symlink() or tmp_link.resolve() != tmp_dir.resolve():
            raise SystemExit(f"tmp link collision: {tmp_link}")
    else:
        tmp_link.symlink_to(tmp_dir, target_is_directory=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if (task_dir / "status.json").exists():
        raise SystemExit("task directory already has status.json")
    gpu = gpu_uuid(args.gpu)
    metric = "SEG" if args.dataset == "livecell" else "CellposeStyleAP"
    data_root = ROOT.parent / "benchmark" / "segmentation"
    data_root = data_root / "LIVECell" if args.dataset == "livecell" else data_root / args.dataset / "extracted"

    def status(state: str, pid: int = 0, epoch: str = "0", exit_code=None,
               result_json: str = "", selected_try: int = 0, error: str = "") -> None:
        atomic_json(task_dir / "status.json", {
            "run_id": args.run_id, "host": args.host, "task_id": args.task_id,
            "dataset": args.dataset, "method": args.mode, "seed": args.seed,
            "gpu_uuid": gpu, "state": state, "pid": pid, "epoch": epoch,
            "heartbeat": stamp(), "exit_code": exit_code, "result_json": result_json,
            "selected_try": selected_try, "error": error,
        })

    shutil.copy2(CONFIG, task_dir / "config_snapshot.yaml")
    with (task_dir / "environment.txt").open("w") as handle:
        handle.write(f"run_id={args.run_id}\nhost={args.host}\ntask_id={args.task_id}\n")
        handle.write(f"dataset={args.dataset}\nmode={args.mode}\nseed={args.seed}\ngpu_uuid={gpu}\n")
        subprocess.run(["/usr/bin/env"], stdout=handle, check=False)
        subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free",
                        "--format=csv"], stdout=handle, stderr=subprocess.STDOUT, check=False)
        subprocess.run(["sha256sum", str(CHECKPOINT), str(CONFIG), str(TRAIN)],
                       stdout=handle, stderr=subprocess.STDOUT, check=False)
    status("starting")
    with log_path.open("a", buffering=1) as log:
        log.write(f"START task={args.task_id} at {stamp()}\n")
        log.write(f"TMPDIR={tmp_link} (target={tmp_dir})\n")
        selected_try = 0
        selected_result = ""
        final_rc = 1
        final_error = ""
        for retry, (batch, accum) in enumerate(((8, 1), (4, 2))):
            if retry == 1 and args.mode != "finetune":
                break
            attempt_dir = task_dir / f"try{retry}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(python), "-m", "dinov3.eval.bio_segmentation.instance_seg.train",
                "--dataset", args.dataset, "--data-root", str(data_root),
                "--checkpoint", str(CHECKPOINT), "--train-config", str(CONFIG),
                "--output-dir", str(attempt_dir), "--layers", "4", "11", "17", "23",
                "--epochs", "50", "--batch-size", str(batch), "--grad-accum-steps", str(accum),
                "--crop-size", "256", "--stride", "192", "--lr", "1e-3",
                "--layer-wise-lr-decay", "0.75", "--weight-decay", "1e-4",
                "--warmup-ratio", "0.05", "--grad-clip-norm", "1.0",
                "--select-metric", metric, "--amp-dtype", "bf16", "--feature-size", "32",
                "--embed-proj", "384", "--fusion-mode", "bucket_concat",
                "--decoder-variant", "current", "--num-workers", "4", "--eval-every", "10",
                "--seed", str(args.seed), "--aug", "strong", "--mosaic-prob", "0.3",
                "--fg-thresh", "0.5", "--energy-thresh", "0.4", "--np-loss-mode", "ce_dice",
                "--focal-gamma", "2.0", "--tversky-alpha", "0.3", "--tversky-beta", "0.7",
                "--skip-test-eval",
            ]
            command += ["--freeze-backbone"] if args.mode == "frozen" else ["--finetune", "--backbone-lr", "1e-5"]
            with (task_dir / "command.txt").open("a") as cmdlog:
                cmdlog.write(f"attempt={retry} batch_size={batch} grad_accum_steps={accum}\n")
                cmdlog.write(" ".join(map(subprocess.list2cmdline, [command])) + "\n")
            env = dict(os.environ)
            env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "TMPDIR": str(tmp_link),
                        "TMP": str(tmp_link), "TEMP": str(tmp_link), "PYTHONUNBUFFERED": "1"})
            command[0] = str(python)
            status("running", 0, "0", None, str(attempt_dir / "results.json"), retry)
            proc = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            status("running", proc.pid, "0", None, str(attempt_dir / "results.json"), retry)
            while proc.poll() is None:
                text = log_path.read_text(errors="ignore")
                matches = re.findall(r"Epoch\s+([0-9]+)(?:/50|:)", text)
                status("running", proc.pid, matches[-1] if matches else "0", None,
                       str(attempt_dir / "results.json"), retry)
                time.sleep(30)
            rc = proc.wait()
            log.write(f"TRAINER_EXIT attempt={retry} rc={rc} at {stamp()}\n")
            final_rc = rc
            selected_try = retry
            selected_result = str(attempt_dir / "results.json")
            if rc == 0:
                break
            if args.mode != "finetune" or not re.search(r"out of memory|cuda error: out of memory",
                                                         log_path.read_text(errors="ignore"), re.I):
                final_error = f"trainer_exit_{rc}"
                break
            log.write("OOM_RETRY batch_size=4 grad_accum_steps=2\n")
        valid = final_rc == 0 and Path(selected_result).is_file()
        if valid:
            try:
                data = json.loads(Path(selected_result).read_text())
                value = float(data["val"][metric])
                valid = value == value and "Results saved" in log_path.read_text(errors="ignore")
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                valid = False
        log_text = log_path.read_text(errors="ignore")
        if not re.search(r"Epoch\s+50/50", log_text):
            valid = False
            final_error = final_error or "normal_end_marker_missing"
        if "Traceback" in log_text and "OSError: [Errno 16] Device or resource busy" not in log_text:
            valid = False
            final_error = "unhandled_traceback"
        state = "completed" if valid else "failed"
        status(state, proc.pid, "50/50" if valid else "", 0 if valid else final_rc,
               selected_result, selected_try, final_error)
        log.write(f"FINAL state={state} result={selected_result} at {stamp()}\n")
        return 0 if valid else (final_rc or 1)

if __name__ == "__main__":
    raise SystemExit(main())
