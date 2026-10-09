#!/usr/bin/env python3
"""Add idle-gated GPU 0-3 workers for the newly staged H+ checkpoint 13175."""

from pathlib import Path
import subprocess


ROOT = Path("/data/hs6_hplus_5tb_eval_20260921")
WORKER = ROOT / "bin/run_hs6_l_6m_full_eval_fleet_worker.py"
PYTHON = Path("/home/xzj/eval_envs/hs6_protocol_v2/bin/python")
LOGS = ROOT / "continuation_v4/nonseg/_state/late13175_logs"


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "adapters/13175/checkpoint.pth"
    if not checkpoint.is_file() or checkpoint.stat().st_size != 1778210727:
        raise RuntimeError("H+ 13175 checkpoint not verified or missing")
    for gpu in range(4):
        name = f"hplus-13175-v4-gpu{gpu}"
        running = subprocess.run(["pgrep", "-af", str(WORKER)],
                                 capture_output=True, text=True).stdout
        if any(f"--worker {name}" in line for line in running.splitlines()):
            print(name, "already_running")
            continue
        command = [
            str(PYTHON), "-u", str(WORKER),
            "--repo", "/home/xzj/biodino_eval_git_20260917",
            "--train-run", str(ROOT / "source"),
            "--snapshot-root", str(ROOT / "source/eval"),
            "--input-root", str(ROOT / "adapters"),
            "--output-root", str(ROOT / "continuation_v4/nonseg"),
            "--benchmark-root", "/data/benchmark",
            "--python-bin", str(PYTHON),
            "--gpu", str(gpu),
            "--worker", name,
            "--official-epoch-length", "4098",
            "--full-eval-period", "488",
            "--min-local-checkpoint-id", "13175",
            "--expected-checkpoints", "13",
            "--jobs-cap", "1",
            "--max-attempts", "1",
            "--poll-seconds", "10",
            "--include-lanes", "classification_a,classification_b,classification_c,classification_d,regression,retrieval",
        ]
        with (LOGS / f"{name}.log").open("ab") as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(name, process.pid)


if __name__ == "__main__":
    main()
