#!/usr/bin/env python3
"""Wait for both DDP evaluations, then collect the six-arm ID figure."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/mnt/huawei_deepcad/dinov3")
OUT = ROOT / "plot/fig2/data_quality"
PYTHON = "/home/bbnc/anaconda3/envs/dinov3/bin/python"
ARMS = ("1tb", "5tb", "20tb_route2_ddp", "20tb_route1_ddp", "100tb", "1pb")
STATUS = OUT / "two20tb_ddp_finalizer_status.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save(state: str, **fields: object) -> None:
    STATUS.write_text(json.dumps(dict(state=state, time_unix=time.time(), **fields), indent=2) + "\n")


def run(command: list[str], name: str) -> None:
    log_path = OUT / f"two20tb_ddp_{name}.log"
    with log_path.open("w") as log:
        code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        save("ERROR", stage=name, code=code, log=str(log_path))
        raise RuntimeError(f"{name} failed: {code}; see {log_path}")


def main() -> None:
    while True:
        states = {
            arm: read(OUT / "eval" / arm / "driver_status.json")
            for arm in ARMS[2:4]
        }
        failures = {arm: read(OUT / f"{'20tb_route1' if arm.endswith('route1_ddp') else '20tb_route2_ddp'}_controller_status.json")
                    for arm in ARMS[2:4]}
        if any(failures[arm].get("state") == "ERROR" and states[arm].get("state") != "COMPLETE"
               for arm in states):
            save("ERROR", controllers=failures)
            raise RuntimeError("One 20TB DDP controller failed")
        if all(record.get("state") == "COMPLETE" and record.get("monuseg_split") == "train30-val7-test14"
               for record in states.values()):
            old_controller_active = subprocess.run(
                ["tmux", "has-session", "-t", "dq20r1_ddp_controller_20261007"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if not old_controller_active:
                break
        save("WAIT_EVAL", evaluations=states, controllers=failures)
        time.sleep(180)
    save("COLLECTING")
    run([PYTHON, "-u", str(ROOT / "scripts/collect_data_quality_v4_id_20261002.py"),
         "--arms", *ARMS, "--stem", "v4_id_two20tb"], "collect")
    save("PLOTTING")
    run([PYTHON, "-u", str(OUT / "plot_data_quality_v4_id_two20tb.py")], "plot")
    save("COMPLETE", figure=str(OUT / "data_quality_v4_id_matched_1m_two20tb.svg"),
         summary=str(OUT / "v4_id_two20tb_summary.json"))


if __name__ == "__main__":
    main()
