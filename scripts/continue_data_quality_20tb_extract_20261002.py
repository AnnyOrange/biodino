#!/usr/bin/env python3
"""Launch exact global 20TB extraction after the source inventory passes."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


REPO = Path("/mnt/huawei_deepcad/dinov3")
OUT = REPO / "plot/fig2/data_quality"
SAMPLE = Path("/mnt/huawei_blm/deepcad_20tb_quality_1m_20261002")
PYTHON = "/home/inspur/anaconda3/envs/dinov3/bin/python"


def status(state: str, **fields) -> None:
    path = OUT / "20tb_extraction_status.json"
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(dict(state=state, time_unix=time.time(), **fields), indent=2) + "\n")
    os.replace(temp, path)


def main() -> None:
    inventory = SAMPLE / "inventory.json"
    while not inventory.is_file():
        alive = subprocess.run(["tmux", "has-session", "-t", "dq20_inventory_20261002"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        status("WAIT_INVENTORY", inventory_session_alive=alive)
        if not alive:
            raise RuntimeError("20TB inventory stopped without a verified inventory.json")
        time.sleep(180)
    record = json.loads(inventory.read_text())
    if record["total_samples"] != 16_625_610 or record["source_tar_count"] != 5526:
        raise ValueError(f"20TB inventory is not the expected complete pool: {record['total_samples']}")
    status("EXTRACTING", total_samples=record["total_samples"])
    command = [PYTHON, "-u", str(REPO / "scripts/sample_data_quality_20tb_1m_20261002.py"),
               "extract", "--workers", "12"]
    with (OUT / "20tb_extract.log").open("w") as log:
        result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        status("ERROR", returncode=result.returncode)
        raise RuntimeError(f"20TB extraction failed with code {result.returncode}")
    complete = json.loads((SAMPLE / "extraction_complete.json").read_text())
    if complete["status"] != "PASS" or complete["samples"] != 1_000_000 or complete["shards"] != 500:
        raise ValueError("20TB extracted sample count or shard count is invalid")
    status("COMPLETE", samples=complete["samples"], shards=complete["shards"], bytes=complete["bytes"])


if __name__ == "__main__":
    main()
