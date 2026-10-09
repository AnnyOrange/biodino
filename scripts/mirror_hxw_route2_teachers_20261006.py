#!/usr/bin/env python3
"""Mirror completed hxw route2 teachers into the local online evaluation run."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

RUN_NAME = ("HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e61x4098_"
            "20tb_route2_r0r9_8x3090qi_20260927")
LOCAL = Path("/mnt/huawei_deepcad/dinov3/outputs/01_training_runs") / RUN_NAME / "eval"
REMOTE = Path("/data/xuzijing/route2_20tb_20261004/run") / RUN_NAME / "eval"
HOST = "5090-lyx-xr"
JOURNAL = Path("/home/lxy/route2_20tb_20261006/teacher_mirror.jsonl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def log(event: str, **fields):
    entry = {"time_unix": time.time(), "event": event, **fields}
    with JOURNAL.open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)


def scan():
    command = (f"test ! -d {REMOTE} || find {REMOTE} -mindepth 2 -maxdepth 2 "
               "-type f -name teacher_checkpoint.pth -printf '%p\\t%s\\t%T@\\n'")
    listing = subprocess.check_output(["ssh", "-o", "BatchMode=yes", HOST, command], text=True)
    for line in listing.splitlines():
        name, size, modified = line.split("\t")
        step = int(Path(name).parent.name.removeprefix("training_"))
        if step <= 7319 or int(size) < 1_000_000_000 or time.time() - float(modified) < 180:
            continue
        target = LOCAL / f"training_{step}" / "teacher_checkpoint.pth"
        if target.exists():
            if target.stat().st_size != int(size):
                log("ERROR_EXISTING_SIZE", step=step, path=str(target))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(".teacher_checkpoint.pth.staging")
        subprocess.run(["rsync", "-a", "--partial", "--append-verify", f"{HOST}:{name}",
                        str(staging)], check=True)
        remote_hash = subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", HOST, f"sha256sum {name}"], text=True).split()[0]
        local_hash = sha256(staging)
        if local_hash != remote_hash:
            raise RuntimeError(f"Teacher hash mismatch at step {step}")
        os.replace(staging, target)
        log("VERIFIED_TEACHER", step=step, bytes=int(size), sha256=local_hash,
            source=name, target=str(target))


if __name__ == "__main__":
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            scan()
        except Exception as error:
            log("ERROR", message=str(error))
        time.sleep(120)
