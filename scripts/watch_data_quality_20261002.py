#!/usr/bin/env python3
"""Record current training, sampling, and ID evaluation progress every three minutes."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3/plot/fig2/data_quality")
ARMS = ("1tb", "5tb", "20tb", "100tb", "1pb", "20tb_route1")


def last_line(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 8192))
        return stream.read().decode(errors="replace").strip().splitlines()[-1]


def arm_status(arm: str) -> dict:
    run = ROOT / "training" / arm
    if not run.exists():
        training = {"state": "NOT_STARTED"}
    else:
        exit_path = run / "exit.json"
        code = json.loads(exit_path.read_text())["returncode"] if exit_path.is_file() else None
        recent = last_line(run / "raw_loss_metrics.jsonl")
        try:
            metric = json.loads(recent) if recent else {}
        except json.JSONDecodeError:
            metric = {}
        training = dict(
            state="RUNNING" if code is None else ("COMPLETE" if code == 0 else "FAILED"),
            exit_code=code, optimizer_updates=metric.get("optimizer_updates_completed", 0),
            target_optimizer_updates=976, consumed_image_visits=metric.get("samples_seen", 0),
            target_image_visits=999_424, last_loss=metric.get("total_loss"),
            checkpoint_exists=(run / "eval/training_975/teacher_checkpoint.pth").is_file(),
            audit_status=(json.loads((run / "audit.json").read_text()).get("status")
                          if (run / "audit.json").is_file() else "PENDING"),
        )
    evaluation = ROOT / "eval" / arm
    eval_status = {}
    for group, expected in (("shared", 38), ("extension", 9), ("monuseg", 1)):
        directory = evaluation / group
        complete = 0
        if directory.exists():
            for path in directory.rglob("validation_report.json"):
                try:
                    complete += json.loads(path.read_text()).get("status") == "VALID_COMPLETE"
                except (OSError, json.JSONDecodeError):
                    pass
        eval_status[group] = dict(completed=complete, expected=expected)
    return {"training": training, "v4_id_evaluation": eval_status}


def gpu_status() -> list[dict]:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "3090-qi",
               "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=15)
    if result.returncode != 0:
        return [{"error": result.stderr.strip()[-200:]}]
    rows = []
    for line in result.stdout.splitlines():
        fields = [int(value.strip().split()[0]) for value in line.split(",")]
        if len(fields) == 4:
            index, used, total, utilization = fields
            rows.append(dict(index=index, memory_used_mib=used, memory_total_mib=total,
                             memory_used_fraction=round(used / total, 4), utilization_pct=utilization))
    return rows


def snapshot() -> dict:
    progress = last_line(ROOT / "20tb_inventory.log")
    extraction = ROOT / "20tb_extraction_status.json"
    return dict(
        time_unix=time.time(), arms={arm: arm_status(arm) for arm in ARMS},
        sampling_20tb=dict(inventory_last_line=progress,
                           extraction=json.loads(extraction.read_text()) if extraction.is_file() else None),
        gpus_3090_qi=gpu_status(),
    )


def main() -> None:
    while True:
        value = snapshot()
        path = ROOT / "RUN_STATUS.json"
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temp, path)
        print(time.strftime("%Y-%m-%d %H:%M:%S"),
              {arm: data["training"]["optimizer_updates"]
               for arm, data in value["arms"].items()
               if "optimizer_updates" in data["training"]}, flush=True)
        time.sleep(180)


if __name__ == "__main__":
    main()
