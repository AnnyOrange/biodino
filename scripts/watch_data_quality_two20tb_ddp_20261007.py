#!/usr/bin/env python3
"""Refresh a compact status for both 20TB DDP runs every three minutes."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


OUT = Path("/mnt/huawei_deepcad/dinov3/plot/fig2/data_quality")
ARMS = ("20tb_route1_ddp", "20tb_route2_ddp")
STATUS = OUT / "two20tb_ddp_live_status.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def gpu_status() -> list[dict]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    cards = []
    uuids = {}
    for line in output.splitlines():
        gpu, uuid, used, total, util = (field.strip() for field in line.split(","))
        gpu, used, total, util = map(int, (gpu, used, total, util))
        cards.append(dict(gpu=gpu, used_mib=used, total_mib=total,
                          memory_fraction=round(used / total, 3), utilization_pct=util,
                          training_used_mib=0, other_used_mib=0))
        uuids[uuid] = cards[-1]
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True)
    for line in apps.splitlines():
        try:
            pid, uuid, memory = (field.strip() for field in line.split(","))
            card = uuids.get(uuid)
            if card is None:
                continue
            command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ")
            if any(f"/training/{arm}".encode() in command for arm in ARMS):
                card["training_used_mib"] += int(memory.split()[0])
        except (OSError, ValueError):
            continue
    for card in cards:
        card["other_used_mib"] = max(0, card["used_mib"] - card["training_used_mib"])
    return cards


def main() -> None:
    while True:
        arms = {}
        for arm in ARMS:
            training = OUT / "training" / arm
            metrics = training / "raw_loss_metrics.jsonl"
            updates = sum(1 for _ in metrics.open()) if metrics.exists() else 0
            controller_name = "20tb_route1" if arm == "20tb_route1_ddp" else arm
            arms[arm] = dict(
                optimizer_updates=updates, target_updates=976,
                training_exit=read(training / "exit.json"),
                training_audit=read(training / "audit.json").get("status"),
                controller=read(OUT / f"{controller_name}_controller_status.json"),
                evaluation=read(OUT / "eval" / arm / "driver_status.json"),
            )
        value = dict(time_unix=time.time(), arms=arms, gpus=gpu_status(),
                     finalizer=read(OUT / "two20tb_ddp_finalizer_status.json"))
        STATUS.write_text(json.dumps(value, indent=2) + "\n")
        if value["finalizer"].get("state") in ("COMPLETE", "ERROR"):
            return
        time.sleep(180)


if __name__ == "__main__":
    main()
