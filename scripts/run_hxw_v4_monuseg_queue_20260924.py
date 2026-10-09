#!/usr/bin/env python3
"""Dispatch identity-locked MoNuSeg component cells to GPUs 0-3."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import time

BASE = Path("/data/hs6_5tb_v4_monuseg_queue_20260924")
ROOTS = {"hplus": Path("/data/hs6_hplus_5tb_eval_20260921"),
         "l": Path("/data/hs6_l_5tb_nogram_eval_20260921")}
HPOINTS = (0, 487, 975, 1463, 1951, 2439, 2927, 3415, 3903, 4391, 4879, 5367, 5855, 6343, 6831,
           7319, 7807, 8295, 8783, 9271, 9759, 10247, 10735, 11223, 11711, 12199, 12687)
PYTHON = "/home/xzj/eval_envs/hs6_protocol_v2/bin/python"
RUNNER = "/data/hs6_l_5tb_nogram_eval_20260921/bin/run_hplus_l_5tb_v4_monuseg_hxw_continuation_20260924.py"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def lpoints():
    root = ROOTS["l"] / "adapters"
    return tuple(sorted(int(p.parent.name) for p in root.glob("*/checkpoint.pth")
                        if p.stat().st_size == 1401909871))


def gpu_free(gpu):
    row = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[gpu]
    return int(row.strip()) / 1024


def mem_available():
    for row in Path("/proc/meminfo").read_text().splitlines():
        if row.startswith("MemAvailable:"):
            return int(row.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable unavailable")


def tasks():
    for campaign, points in (("hplus", HPOINTS), ("l", lpoints())):
        for point in points:
            yield campaign, point


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True, type=int, choices=(0, 1, 2, 3))
    p.add_argument("--slot", required=True)
    args = p.parse_args()
    for child in ("claims", "failures"):
        (BASE / child).mkdir(parents=True, exist_ok=True)
    while True:
        free, avail = gpu_free(args.gpu), mem_available()
        if free < 15 or avail < 70 or shutil.disk_usage("/").free < 150 * 2**30:
            print(f"{now()} admission_wait gpu_free_gib={free:.1f} mem_available_gib={avail:.1f}", flush=True)
            time.sleep(20)
            continue
        selected = None
        for campaign, point in tasks():
            name = f"point_{point}__monuseg__formal-static-v1"
            cell = ROOTS[campaign] / "v3/cells" / name
            if (cell / "validation_report.json").is_file():
                continue
            key = f"{campaign}__{point}"
            if (BASE / "failures" / f"{key}.json").is_file():
                continue
            if cell.is_dir() and any(cell.iterdir()):
                continue
            claim = BASE / "claims" / key
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            if cell.is_dir() and any(cell.iterdir()):
                claim.rmdir()
                continue
            selected = campaign, point, cell, claim, key
            break
        if selected is None:
            print(f"{now()} no_unclaimed_cells", flush=True)
            time.sleep(60)
            continue
        campaign, point, cell, claim, key = selected
        cmd = [PYTHON, "-u", RUNNER, "--campaign", campaign, "--point", str(point),
               "--dataset", "monuseg", "--gpu", str(args.gpu)]
        print(f"{now()} START {key} gpu={args.gpu}", flush=True)
        with (BASE / f"{key}.log").open("a") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0 and (cell / "validation_report.json").is_file():
            claim.rmdir()
            print(f"{now()} FINISH {key}", flush=True)
        else:
            (BASE / "failures" / f"{key}.json").write_text(json.dumps({
                "key": key, "returncode": result.returncode, "cell": str(cell), "utc": now()}, indent=2) + "\n")
            print(f"{now()} FAILED {key} rc={result.returncode}", flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
