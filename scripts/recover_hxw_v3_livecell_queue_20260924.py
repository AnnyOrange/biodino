#!/usr/bin/env python3
"""Retry resource-interrupted formal LIVECell cells with bounded concurrency."""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import time

BASE = Path("/data/hs6_5tb_v3_livecell_recovery_20260924")
QUEUE = Path("/data/hs6_5tb_v3_parallel_queue_20260923")
ROOTS = {"hplus": Path("/data/hs6_hplus_5tb_eval_20260921"),
         "l": Path("/data/hs6_l_5tb_nogram_eval_20260921")}
RUNNER = ROOTS["l"] / "bin/run_hplus_l_5tb_v3_dense_hxw_20260922.py"
PYTHON = "/home/xzj/eval_envs/hs6_protocol_v2/bin/python"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def mem_available():
    for row in Path("/proc/meminfo").read_text().splitlines():
        if row.startswith("MemAvailable:"):
            return int(row.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable missing")


def gpu_free(gpu):
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                        "--format=csv,noheader,nounits"], text=True).splitlines()[gpu]) / 1024


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True, choices=(0, 1, 2, 3))
    args = p.parse_args()
    for name in ("claims", "attempted", "logs", "recovered"):
        (BASE / name).mkdir(parents=True, exist_ok=True)
    while True:
        if (mem_available() < 130 or gpu_free(args.gpu) < 7
                or shutil.disk_usage("/").free < 250 * 2**30):
            print(f"{now()} admission_wait mem_gib={mem_available():.1f} gpu_free_gib={gpu_free(args.gpu):.1f}", flush=True)
            time.sleep(30)
            continue
        selected = None
        for receipt in sorted((QUEUE / "failures").glob("*__livecell__formal_static_v1.json")):
            key = receipt.stem
            if (BASE / "attempted" / f"{key}.json").exists():
                continue
            campaign, point_str, _, _ = key.split("__", 3)
            point = int(point_str)
            cell = ROOTS[campaign] / "v3/cells" / f"point_{point}__livecell__formal-static-v1"
            report = cell / "validation_report.json"
            if report.is_file() and json.loads(report.read_text()).get("status") == "VALID_COMPLETE":
                shutil.move(str(receipt), str(BASE / "recovered" / receipt.name))
                continue
            if not (cell / "command_manifest.json").is_file():
                continue
            claim = BASE / "claims" / key
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            selected = key, receipt, campaign, point, cell, claim
            break
        if selected is None:
            print(f"{now()} no_pending_cells", flush=True)
            time.sleep(60)
            continue
        key, receipt, campaign, point, cell, claim = selected
        cmd = [PYTHON, "-u", str(RUNNER), "--campaign", campaign, "--point", str(point),
               "--dataset", "livecell", "--gpu", str(args.gpu), "--resume-existing",
               "--scratch-root", "/home/xzj/hs6_5tb_v3_scratch"]
        print(f"{now()} START {key} gpu={args.gpu}", flush=True)
        with (BASE / "logs" / f"{key}.log").open("a") as out:
            rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
        report = cell / "validation_report.json"
        valid = rc == 0 and report.is_file() and json.loads(report.read_text()).get("status") == "VALID_COMPLETE"
        if valid:
            shutil.move(str(receipt), str(BASE / "recovered" / receipt.name))
            claim.rmdir()
            print(f"{now()} FINISH {key}", flush=True)
        else:
            (BASE / "attempted" / f"{key}.json").write_text(json.dumps({
                "key": key, "gpu": args.gpu, "returncode": rc, "log": str(BASE / "logs" / f"{key}.log"),
                "utc": now()}, indent=2) + "\n")
            print(f"{now()} FAILED {key} rc={rc}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
