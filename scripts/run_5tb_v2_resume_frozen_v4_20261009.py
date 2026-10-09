#!/usr/bin/env python3
"""Run resident v4 frozen lanes beside ongoing 5TB recovery training."""

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


SITES = {
    "lyx": {
        "arm": "global_cls",
        "run": Path("/data/xuzijing/biodino_v2_20260930/outputs/01_training_runs/hs6_l5_v2_recovery_fork29279_20260930/global_cls"),
        "eval": Path("/data/xuzijing/hs6_l5_v2_recovery_eval_20260930"),
        "repo": Path("/data/xuzijing/biodino_eval_git_20260917"),
        "python": Path("/data/xuzijing/eval_envs/hs6_protocol_v2/bin/python"),
        "worker": Path("/data/xuzijing/hs6_l5_v2_recovery_eval_20260930/bin/run_hs6_l_6m_full_eval_fleet_worker.py"),
        "benchmark": Path("/data/xuzijing/benchmark"),
        "train_config": Path("/data/xuzijing/hs6_l5_v2_recovery_eval_20260930/bin"),
        "cards": {4: (2, 3500), 5: (2, 3500)},
        "prefix": "lyx-w1-resume-v4",
    },
    "hxw": {
        "arm": "global_cls_w3",
        "run": Path("/home/xzj/biodino_v2_20260930/outputs/01_training_runs/hs6_l5_v2_recovery_fork29279_20260930/global_cls_w3"),
        "eval": Path("/data/hs6_l5_v2_recovery_eval_20260930"),
        "repo": Path("/home/xzj/biodino_eval_git_20260917"),
        "python": Path("/home/xzj/eval_envs/hs6_protocol_v2/bin/python"),
        "worker": Path("/data/hs6_hplus_5tb_eval_20260921/bin/run_hs6_l_6m_full_eval_fleet_worker.py"),
        "benchmark": Path("/data/benchmark"),
        "train_config": Path("/data/hs6_l_5tb_nogram_eval_20260921/source"),
        "cards": {0: (2, 3500), 1: (1, 2000), 2: (1, 2000), 3: (1, 2000),
                  5: (2, 3500), 6: (2, 3500), 7: (2, 3500)},
        "prefix": "hxw-w3-resume-v4",
    },
}

LANES = "classification_a,classification_b,classification_c,classification_d,regression,retrieval"
EXPECTED_TEST_MIB = 4200


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def memory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {int(row[0]): (int(row[1]), int(row[2]))
            for line in output.splitlines() if (row := [part.strip() for part in line.split(",")])}


def resident_workers(prefix):
    output = subprocess.check_output(["ps", "-eo", "args"], text=True)
    counts = {}
    pattern = re.compile(r"--gpu (\d+) --worker (\S+)")
    for line in output.splitlines():
        match = pattern.search(line)
        if match and match.group(2).startswith(prefix):
            gpu = int(match.group(1))
            counts[gpu] = counts.get(gpu, 0) + 1
    return counts


def launch(site, gpu, slot, log_root):
    arm = site["arm"]
    name = f"{site['prefix']}-gpu{gpu}-slot{slot}"
    command = [str(site["python"]), "-u", str(site["worker"]),
               "--repo", str(site["repo"]), "--train-run", str(site["train_config"]),
               "--snapshot-root", str(site["run"] / "eval"),
               "--input-root", str(site["eval"] / arm / "adapters"),
               "--output-root", str(site["eval"] / arm / "results"),
               "--benchmark-root", str(site["benchmark"]),
               "--python-bin", str(site["python"]), "--gpu", str(gpu), "--worker", name,
               "--official-epoch-length", "4098", "--full-eval-period", "488",
               "--min-local-checkpoint-id", "35623", "--expected-checkpoints", "31",
               "--ready-age-seconds", "300", "--jobs-cap", "1", "--max-attempts", "2",
               "--poll-seconds", "15", "--allow-busy-gpu", "--once", "--include-lanes", LANES]
    env = dict(os.environ, PYTHONPATH=str(site["repo"]), FROZEN_BATCH_SIZE="64",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log = log_root / f"{name}.log"
    with log.open("ab") as stream:
        process = subprocess.Popen(command, cwd=site["repo"], env=env,
                                   stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=SITES, required=True)
    args = parser.parse_args()
    site = SITES[args.site]
    log_root = site["eval"] / "logs/resume_v4_20261009"
    log_root.mkdir(parents=True, exist_ok=True)
    with (log_root / f"{args.site}_frozen_controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children = {}
        started = {}
        cooldown = {}
        while True:
            now = time.time()
            for key, process in list(children.items()):
                if (status := process.poll()) is not None:
                    print("EXIT", key, status, flush=True)
                    cooldown[key] = now + (120 if status else 45)
                    del children[key]
                    del started[key]
            cards = memory()
            counts = resident_workers(site["prefix"])
            for gpu, (limit, headroom) in site["cards"].items():
                used, total = cards[gpu]
                loading = sum(EXPECTED_TEST_MIB for key, when in started.items()
                              if key.startswith(f"gpu{gpu}-") and now - when < 90)
                if counts.get(gpu, 0) >= limit or total - used - loading < EXPECTED_TEST_MIB + headroom:
                    continue
                for slot in range(limit):
                    key = f"gpu{gpu}-slot{slot}"
                    if key in children or cooldown.get(key, 0) > now:
                        continue
                    children[key] = launch(site, gpu, slot, log_root)
                    started[key] = now
                    print("START", key, children[key].pid, "gpu_used_mib", used, flush=True)
                    break
                break
            atomic_json(log_root / f"{args.site}_frozen_status.json", {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_memory": cards, "workers": counts,
                "children": {key: process.pid for key, process in children.items()},
                "limits": {gpu: value[0] for gpu, value in site["cards"].items()},
            })
            time.sleep(20)


if __name__ == "__main__":
    main()
