#!/usr/bin/env python3
"""Resume ck80031 evaluation queues after their already-started jobs drain."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO = Path("/mnt/huawei_deepcad/dinov3")
PAIR = REPO / "outputs/02_eval_runs/v2_20tb_ck80031_20261008"
V4 = REPO / "outputs/02_eval_runs/v2_full_v4_ck80031_20261008"
SCRIPT = REPO / "scripts/run_20tb_ck80031_v4_20261008.py"
SNAPSHOT_OLD = "/mnt/huawei_deepcad/dinov3_retest_snapshot_20260918/"
SNAPSHOT_NEW = "/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918/"
PYTHON = "/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python"
HOST = "3090-qi"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_until(check, label: str) -> None:
    while not check():
        print("WAIT", label, flush=True)
        time.sleep(15)


def run_worker(mode: str, log_path: Path, *, safe_home: Path | None = None) -> int:
    command = [PYTHON, "-u", str(SCRIPT), mode, "--host", HOST,
               "--gpus", "0", "1", "2", "3", "4", "5", "6", "7"]
    if mode == "worker-base":
        command += ["--target-per-gpu", "8", "--memory-target", "0.75", "--max-jobs", "40"]
    else:
        command += ["--max-jobs", "40", "--ram-reserve", "96"]
    env = os.environ.copy()
    if safe_home is not None:
        env["HOME"] = str(safe_home)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print("RESTARTED", mode, process.pid, flush=True)
    return process.pid


def main() -> None:
    lock = (PAIR / "logs/recovery_supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    pair_status = json.loads((PAIR / "_state/workers/3090-qi.json").read_text())
    pair_pid = int(pair_status["pid"])
    wait_until(lambda: not list((PAIR / "_state/running").glob("*.json")) and not alive(pair_pid),
               "base component jobs to drain")

    pair_manifest_path = PAIR / "campaign_manifest.json"
    pair_manifest = json.loads(pair_manifest_path.read_text())
    for task in pair_manifest["tasks"]:
        hashes = task["dataset"].get("source_hashes", {})
        replacements = {}
        for source, expected in list(hashes.items()):
            if not source.startswith(SNAPSHOT_OLD):
                continue
            target = SNAPSHOT_NEW + source[len(SNAPSHOT_OLD):]
            target_path = Path(target)
            if not target_path.is_file() or digest(target_path) != expected:
                raise RuntimeError(f"Frozen split source did not match: {target}")
            replacements[target] = expected
            del hashes[source]
        hashes.update(replacements)
    save(pair_manifest_path, pair_manifest)

    retry_history = PAIR / "_state/retry_history"
    retry_history.mkdir(exist_ok=True)
    pause = PAIR / "_state/PAUSED.json"
    if pause.exists():
        shutil.move(str(pause), str(retry_history / f"PAUSED_{int(time.time())}.json"))
    failed_key = "cls_slow2_20tb_ck80031__regression__bbbc005"
    failed_claim = PAIR / "_state/claims" / failed_key
    if failed_claim.exists():
        shutil.move(str(failed_claim), str(retry_history / f"claim_{failed_key}"))
    run_worker("worker-base", PAIR / "logs/worker_3090-qi.log")

    v4_status_path = V4 / "workers/3090-qi.json"
    v4_status = json.loads(v4_status_path.read_text())
    v4_pid = int(v4_status["pid"])
    wait_until(lambda: not json.loads(v4_status_path.read_text()).get("active"),
               "v4 jobs to drain")
    if alive(v4_pid):
        os.kill(v4_pid, signal.SIGTERM)
        wait_until(lambda: not alive(v4_pid), "v4 worker to stop")

    ctc_key = "ctc__native__cls_slow2_20tb_ck80031"
    ctc_claim = V4 / "claims" / ctc_key
    ctc_history = V4 / "claims/retry_history"
    ctc_history.mkdir(exist_ok=True)
    if ctc_claim.exists():
        shutil.move(str(ctc_claim), str(ctc_history / f"{ctc_key}_{int(time.time())}"))

    safe_home = Path("/tmp/codex_v4_home_20261008")
    safe_home.mkdir(exist_ok=True)
    subprocess.run(["git", "config", "--global", "--add", "safe.directory",
                    "/mnt/huawei_deepcad/dinov3/outputs/02_eval_runtime/py-ctcmetrics"],
                   env=dict(os.environ, HOME=str(safe_home)), check=True)
    run_worker("worker-v4", V4 / "logs/worker_3090-qi.log", safe_home=safe_home)


if __name__ == "__main__":
    main()
