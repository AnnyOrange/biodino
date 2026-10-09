#!/usr/bin/env python3
"""Run one audited arm's v4 ID queue, extensions, and MoNuSeg 30/7/14."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from audit_data_quality_1m_20261002 import ARMS, ROOT


REPO = Path("/mnt/huawei_deepcad/dinov3")
PYTHON = "/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python"
SHARED_SOURCE = Path("/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918")
MONUSEG_SOURCE = Path("/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929")


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def gpu_status(gpus: list[int], arm: str | None = None) -> list[dict]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    uuids = {}
    for line in output.splitlines():
        index, uuid, used, total, util = [part.strip() for part in line.split(",")]
        index, used, total, util = map(int, (index, used, total, util))
        if index in gpus:
            rows.append(dict(gpu=index, used_mib=used, total_mib=total,
                             memory_fraction=round(used / total, 4), utilization_pct=util,
                             own_used_mib=0, own_process_count=0))
            uuids[uuid] = rows[-1]
    if arm:
        apps = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], text=True)
        for line in apps.splitlines():
            try:
                pid_text, uuid, used_text = [part.strip() for part in line.split(",")]
                row = uuids.get(uuid)
                if row is None:
                    continue
                cmdline = Path(f"/proc/{int(pid_text)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if f"/eval/{arm}/" in cmdline or f"dq_{arm}" in cmdline:
                    row["own_used_mib"] += int(used_text.split()[0])
                    row["own_process_count"] += 1
            except (OSError, ValueError):
                continue
    for row in rows:
        row["own_memory_fraction"] = round(row["own_used_mib"] / row["total_mib"], 4)
    return rows


def completed(root: Path, group: str) -> int:
    if group == "shared":
        manifest = json.loads((root / "shared/campaign_manifest.json").read_text())
        return sum((root / "shared/cells" / task["key"] / "validation_report.json").is_file()
                   and json.loads((root / "shared/cells" / task["key"] / "validation_report.json").read_text()).get("status") == "VALID_COMPLETE"
                   for task in manifest["tasks"])
    if group == "extension":
        tasks = [json.loads(path.read_text()) for path in (root / "extension/tasks").glob("*.json")]
        return sum((Path(task["output"]) / "validation_report.json").is_file()
                   and json.loads((Path(task["output"]) / "validation_report.json").read_text()).get("status") == "VALID_COMPLETE"
                   for task in tasks)
    manifest = json.loads((root / "monuseg/campaign_manifest.json").read_text())
    return sum((root / "monuseg/cells" / task["key"] / "validation_report.json").is_file()
               and json.loads((root / "monuseg/cells" / task["key"] / "validation_report.json").read_text()).get("status") == "VALID_COMPLETE"
               for task in manifest["tasks"])


def wait_for_gpus(root: Path, gpus: list[int]) -> None:
    while True:
        cards = gpu_status(gpus)
        ready = len(cards) == len(gpus) and all(
            row["total_mib"] - row["used_mib"] >= 20_000 and row["utilization_pct"] <= 20
            for row in cards
        )
        if ready:
            return
        save(root / "driver_status.json", dict(time_unix=time.time(), state="WAIT_GPU",
                                               gpu_status=cards))
        time.sleep(180)


def run_phase(root: Path, phase: str, commands: list[list[str]], gpus: list[int], expected: int) -> None:
    arm_label = json.loads((root / "prepared.json").read_text())["arm"]
    processes = []
    for index, command in enumerate(commands):
        log_path = root / f"{phase}_worker_{index}.log"
        log = log_path.open("w")
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        processes.append((process, log, command))
    try:
        while any(process.poll() is None for process, _, _ in processes):
            done = completed(root, phase)
            cards = gpu_status(gpus, arm_label)
            record = dict(time_unix=time.time(), phase=phase, completed=done, expected=expected,
                          gpu_status=cards, under_50pct_while_pending=[row["gpu"] for row in cards
                          if row["own_memory_fraction"] < 0.5 and done < expected])
            with (root / "gpu_monitor.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            save(root / "driver_status.json", record)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), phase, f"{done}/{expected}",
                  "under_50pct", record["under_50pct_while_pending"], flush=True)
            time.sleep(180)
        codes = [process.wait() for process, _, _ in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"{phase} worker exit codes: {codes}")
        done = completed(root, phase)
        if done != expected:
            raise RuntimeError(f"{phase} validated {done}/{expected} tasks")
        save(root / "driver_status.json", dict(time_unix=time.time(), phase=phase,
                                                completed=done, expected=expected, state="COMPLETE"))
    finally:
        for _, log, _ in processes:
            log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--eval-root", type=Path,
                        help="Prepared campaign root; defaults to the matched 1M arm")
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.eval_root is None and args.arm not in ARMS:
        raise ValueError(f"Unknown matched 1M arm: {args.arm}")
    gpus = args.gpus
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("Provide distinct evaluation GPUs")
    root = args.eval_root if args.eval_root is not None else ROOT / "eval" / args.arm
    prepared = json.loads((root / "prepared.json").read_text())
    if prepared["arm"] != args.arm or prepared["monuseg_split"] != "monuseg2018-train30-extra7val-test14-v1":
        raise ValueError("Prepared arm or MoNuSeg split mismatch")
    host = platform.node()
    if any(completed(root, phase) != count for phase, count in
           (("shared", 38), ("extension", 9), ("monuseg", 1))):
        wait_for_gpus(root, gpus)

    if completed(root, "shared") != 38:
        command = [PYTHON, "-u", str(SHARED_SOURCE / "scripts/run_retest_fleet_20260918.py"),
                   "worker", "--output", str(root / "shared"), "--host", host,
                   "--gpus", *map(str, gpus), "--target-per-gpu", "5",
                   "--max-host-jobs", str(5 * len(gpus)), "--max-global-jobs", "160"]
        run_phase(root, "shared", [command], gpus, 38)
    if completed(root, "extension") != 9:
        commands = [[PYTHON, "-u", str(REPO / "scripts/run_data_quality_v4_extension_worker_20261002.py"),
                     "--arm", args.arm, "--eval-root", str(root),
                     "--gpu", str(gpu), "--slot", str(slot)]
                    for slot in range(5) for gpu in gpus[:2]]
        run_phase(root, "extension", commands, gpus[:2], 9)
    if completed(root, "monuseg") != 1:
        command = [PYTHON, "-u", str(MONUSEG_SOURCE / "scripts/run_retest_fleet_20260918.py"),
                   "worker", "--output", str(root / "monuseg"), "--host", host,
                   "--gpus", str(gpus[0]), "--target-per-gpu", "1",
                   "--max-host-jobs", "1", "--max-global-jobs", "160",
                   "--task-family", "segmentation"]
        run_phase(root, "monuseg", [command], gpus[:1], 1)
    save(root / "driver_status.json", dict(time_unix=time.time(), state="COMPLETE",
                                            arm=args.arm, shared=38, extension=9, monuseg=1,
                                            monuseg_split="train30-val7-test14"))
    print(f"v4 ID evaluation complete for {args.arm}", flush=True)


if __name__ == "__main__":
    main()
