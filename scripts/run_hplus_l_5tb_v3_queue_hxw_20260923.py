#!/usr/bin/env python3
"""Keep one real formal-v3 test slot busy on hxw with atomic cell claims."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HROOT = Path("/data/hs6_hplus_5tb_eval_20260921")
LROOT = Path("/data/hs6_l_5tb_nogram_eval_20260921")
QUEUE = Path("/data/hs6_hplus_5tb_continuation_v3_conic_queue_20260924")
SCRATCH = Path("/home/xzj/hs6_5tb_v3_scratch")
PYTHON = Path("/home/xzj/eval_envs/hs6_protocol_v2/bin/python")
LAUNCHER = LROOT / "bin/run_hplus_l_5tb_v3_dense_hxw_continuation_20260924.py"
HPOINTS = (7319, 7807, 8295, 8783, 9271, 9759, 10247, 10735, 11223, 11711, 12199, 12687)
PANNUKE = (
    "pannuke-fold1-train-fold2-val-fold3-test",
    "pannuke-fold2-train-fold1-val-fold3-test",
    "pannuke-fold3-train-fold2-val-fold1-test",
)
# Start with the smaller last-layer caches; LIVECell is deliberately last.
DATASETS = (("conic", None),)


def lpoints():
    result = []
    for checkpoint in (LROOT / "source/eval").glob("training_*/teacher_checkpoint.pth"):
        try:
            point = int(checkpoint.parent.name.removeprefix("training_"))
        except ValueError:
            continue
        if point >= 29767 and (point - 29767) % 488 == 0 and checkpoint.stat().st_size == 1401909871:
            result.append(point)
    return tuple(sorted(result))


def slug(value):
    return value.replace("-", "_")


def cells():
    for campaign, root, points in (("hplus", HROOT, HPOINTS),):
        for dataset, fold in DATASETS:
            split = fold or ("official-baseline-fold0-nested-v1" if dataset == "conic" else "formal-static-v1")
            for point in points:
                yield campaign, root, point, dataset, fold, split


def resources_ok(gpu):
    data_gib = shutil.disk_usage("/data").free / 2**30
    system_gib = shutil.disk_usage("/").free / 2**30
    available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
    memory_gib = available_kib / 1024**2
    free_mib = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    gpu_free_gib = int(free_mib[gpu]) / 1024
    return (data_gib >= 100 and system_gib >= 180 and memory_gib >= 80
            and gpu_free_gib >= 15), data_gib, system_gib, memory_gib, gpu_free_gib


def wait_for(pid):
    if not pid:
        return
    while Path(f"/proc/{pid}").exists():
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, choices=(0, 1, 2, 3, 4))
    parser.add_argument("--slot", required=True)
    parser.add_argument("--wait-pid", type=int, default=0)
    args = parser.parse_args()
    if not args.slot.replace("_", "").isalnum():
        parser.error("Unsafe slot name")
    QUEUE.joinpath("claims").mkdir(parents=True, exist_ok=True)
    QUEUE.joinpath("failures").mkdir(parents=True, exist_ok=True)
    QUEUE.joinpath("logs").mkdir(parents=True, exist_ok=True)
    wait_for(args.wait_pid)
    while True:
        okay, data_disk, system_disk, memory, gpu_free = resources_ok(args.gpu)
        if not okay:
            print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} admission_wait "
                  f"data_gib={data_disk:.1f} system_gib={system_disk:.1f} "
                  f"mem_gib={memory:.1f} gpu_free_gib={gpu_free:.1f}", flush=True)
            time.sleep(30)
            continue
        # Refresh after every cell so newly transferred L checkpoints join v3
        # without waiting for the entire older inventory to drain.
        tasks = list(cells())
        # Rotate deterministic traversal so 20 slots do not contend for one prefix.
        shift = (args.gpu * 17 + sum(args.slot.encode())) % len(tasks)
        tasks = tasks[shift:] + tasks[:shift]
        selected = None
        for campaign, root, point, dataset, fold, split in tasks:
            name = f"point_{point}__{dataset}__{split}"
            cell = root / "v3/cells" / name
            report = cell / "validation_report.json"
            if report.exists() and json.loads(report.read_text()).get("status") == "VALID_COMPLETE":
                continue
            key = f"{campaign}__{point}__{dataset}__{slug(split)}"
            claim = QUEUE / "claims" / key
            if (QUEUE / "failures" / f"{key}.json").exists():
                continue
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            # Race defense against manually admitted or another completed cell.
            if report.exists() and json.loads(report.read_text()).get("status") == "VALID_COMPLETE":
                claim.rmdir()
                continue
            resume = cell.exists() and any(cell.iterdir())
            selected = campaign, root, point, dataset, fold, key, claim, resume
            break
        if selected is None:
            print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} no_unclaimed_cells", flush=True)
            time.sleep(60)
            continue
        campaign, root, point, dataset, fold, key, claim, resume = selected
        command = [str(PYTHON), "-u", str(LAUNCHER), "--campaign", campaign,
                   "--point", str(point), "--dataset", dataset, "--gpu", str(args.gpu),
                   "--scratch-root", str(SCRATCH)]
        if resume:
            command.append("--resume-existing")
        if fold:
            command += ["--fold", fold]
        log = QUEUE / "logs" / f"gpu{args.gpu}_{args.slot}__{key}.log"
        print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} START {key} gpu={args.gpu}", flush=True)
        with log.open("a") as output:
            rc = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
        report = next((root / "v3/cells" / f"point_{point}__{dataset}__{fold or 'formal-static-v1'}").glob("validation_report.json"), None)
        if rc == 0:
            try:
                claim.rmdir()
            except OSError:
                pass
            print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} FINISH {key}", flush=True)
        else:
            failure = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "key": key,
                       "gpu": args.gpu, "slot": args.slot, "returncode": rc,
                       "command": command, "log": str(log)}
            destination = QUEUE / "failures" / f"{key}.json"
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(failure, indent=2) + "\n")
            temporary.replace(destination)
            print(f"{failure['utc']} FAILED {key} rc={rc}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
