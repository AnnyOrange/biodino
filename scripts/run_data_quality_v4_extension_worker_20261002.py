#!/usr/bin/env python3
"""Run prepared v4 ID extension tasks on one physical GPU with provenance checks."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from audit_data_quality_1m_20261002 import ARMS, ROOT


MODULE = Path("/mnt/huawei_deepcad/dinov3/scripts/run_fourmodel_v4_completion_20260924.py")


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temp, path)


def free_mib(gpu: int) -> int:
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
        "-i", str(gpu),
    ], text=True)
    return int(result.strip().splitlines()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--eval-root", type=Path,
                        help="Prepared campaign root; defaults to the matched 1M arm")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Stop after this many claimed tasks; zero runs the full queue")
    args = parser.parse_args()
    if args.eval_root is None and args.arm not in ARMS:
        raise ValueError(f"Unknown matched 1M arm: {args.arm}")
    if args.max_tasks < 0:
        raise ValueError("--max-tasks must be nonnegative")
    root = (args.eval_root if args.eval_root is not None else ROOT / "eval" / args.arm) / "extension"
    manifest = json.loads((root / "campaign_manifest.json").read_text())
    task_paths = sorted((root / "tasks").glob("*.json"))
    if len(task_paths) != 9 or len(manifest["task_ids"]) != 9:
        raise ValueError("Expected exactly nine v4 extensions; MoNuSeg must use the 30/7/14 evaluator")
    spec = importlib.util.spec_from_file_location("dq_v4_extension", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    module.ASSETS = ({"arm": f"dq_{args.arm}"},)
    lock_path = root / "supervisors" / f"{platform.node()}_gpu{args.gpu}_slot{args.slot}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        claimed = 0
        for path in sorted(task_paths, key=lambda p: json.loads(p.read_text())["order"]):
            if args.max_tasks and claimed >= args.max_tasks:
                break
            task = json.loads(path.read_text())
            if task["id"] not in manifest["task_ids"]:
                raise ValueError(f"Unregistered task: {task['id']}")
            claim = root / "claims" / task["id"]
            if claim.exists():
                continue
            try:
                valid, _ = module.validate_task(task)
            except Exception:
                valid = False
            if valid:
                raise RuntimeError(f"Valid result without a registered claim: {task['id']}")
            try:
                claim.mkdir(parents=True)
            except FileExistsError:
                continue
            claimed += 1
            threshold = 16000 if task["family"] in ("segmentation", "detection") else args.min_free_mib
            while free_mib(args.gpu) < threshold:
                print(f"WAIT_GPU gpu={args.gpu} free={free_mib(args.gpu)} task={task['id']}", flush=True)
                time.sleep(30)
            output = Path(task["output"])
            output.mkdir(parents=True, exist_ok=True)
            log_path = root / "logs" / f"{task['id']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            status = dict(state="RUNNING", host=platform.node(), gpu=args.gpu,
                          pid=os.getpid(), task=task["id"], start_unix=time.time(),
                          log=str(log_path))
            save(claim / "status.json", status)
            save(output / "v4_invocation.json", dict(
                protocol_id=module.PROTOCOL_ID,
                protocol_sha256=module.sha256(module.PROTOCOL),
                task=task, host=platform.node(), gpu=args.gpu,
                start_unix=status["start_unix"],
            ))
            env = os.environ.copy()
            env.update(module.THREAD_ENV)
            env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=task["pythonpath"],
                       PYTHONUNBUFFERED="1")
            print(f"START gpu={args.gpu} task={task['id']}", flush=True)
            with log_path.open("a") as log:
                completed = subprocess.run(task["command"], cwd=task["cwd"], env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
            try:
                valid, reason = module.validate_task(task)
            except Exception as error:
                valid, reason = False, f"{type(error).__name__}: {error}"
            status.update(state="DONE" if completed.returncode == 0 and valid else "FAILED",
                          returncode=completed.returncode, validation=reason, end_unix=time.time())
            save(claim / "status.json", status)
            if status["state"] == "DONE":
                save(output / "validation_report.json", dict(
                    status="VALID_COMPLETE", protocol_id=module.PROTOCOL_ID,
                    task_id=task["id"], validation=reason,
                    checkpoint_sha256=task["checkpoint_sha256"],
                    config_sha256=task["config_sha256"],
                    source_entry_sha256=task["source_entry_sha256"],
                    validated_unix=time.time(),
                ))
            print(f"{status['state']} gpu={args.gpu} task={task['id']} rc={completed.returncode} {reason}", flush=True)
        module.summary()


if __name__ == "__main__":
    main()
