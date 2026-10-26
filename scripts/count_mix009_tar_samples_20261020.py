#!/usr/bin/env python3
"""Count packed image samples in the four mix009 tar pools, resumably."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


POOLS = {
    "old1": "/data/microscopy-100k-patched/filtered_mixed_train_w*.tar",
    "old4": "/home/xzj/storage/merged/4TB/wds_patched_shuffle/filtered_mixed_train*.tar",
    "new15_r0": "/home/xzj/storage/merged/15TB_from_20TB/filtered_projection_20TB_nested-r0*.tar",
    "new15_r9": "/home/xzj/storage/merged/15TB_from_20TB/filtered_projection_20TB_nested-r9*.tar",
}


def count_tar(task):
    pool, path, expected_size, expected_mtime_ns = task
    if pool == "new15_r9":
        direct = [path.replace("/storage/merged/", f"/storage/disks/{disk}/", 1)
                  for disk in ("a", "b")]
        direct = [candidate for candidate in direct if os.path.isfile(candidate)
                  and os.stat(candidate).st_size == expected_size
                  and os.stat(candidate).st_mtime_ns == expected_mtime_ns]
        if len(direct) != 1:
            raise RuntimeError(f"Expected one matching physical tar for {path}: {direct}")
        count = members = 0
        with subprocess.Popen(["tar", "-tf", direct[0]], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) as proc:
            for name in proc.stdout:
                members += 1
                count += name.rstrip("\n").endswith(".meta.json")
            error = proc.stderr.read()
            rc = proc.wait()
        if rc or not count:
            raise RuntimeError(f"tar listing failed for {path}: {rc} {error[-500:]}")
        return dict(pool=pool, path=path, size=expected_size,
                    mtime_ns=expected_mtime_ns, samples=count, members=members)

    count = 0
    members = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        size = os.fstat(fd).st_size
        if size != expected_size:
            raise RuntimeError(f"File size changed: {path}")
        offset = 0
        while offset + 512 <= size:
            header_bytes = os.pread(fd, 512, offset)
            if len(header_bytes) != 512:
                raise RuntimeError(f"Short tar header: {path} at {offset}")
            if header_bytes == b"\0" * 512:
                break
            header = tarfile.TarInfo.frombuf(header_bytes, "utf8", "strict")
            next_offset = offset + 512 + ((header.size + 511) // 512) * 512
            if header.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
                offset = next_offset
                continue
            if not header.isfile():
                with tarfile.open(path, "r:") as archive:
                    count = 0
                    members = 0
                    for member in archive:
                        if member.isfile():
                            members += 1
                            count += member.name.endswith(".meta.json")
                        archive.members.clear()
                break
            members += 1
            if header.name.endswith(".meta.json"):
                count += 1
            offset = next_offset
        if not count:
            raise RuntimeError(f"No sample metadata: {path}")
        if os.fstat(fd).st_mtime_ns != expected_mtime_ns:
            raise RuntimeError(f"File changed during scan: {path}")
    finally:
        os.close(fd)
    return dict(pool=pool, path=path, size=expected_size,
                mtime_ns=expected_mtime_ns, samples=count, members=members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pools", nargs="+", choices=tuple(POOLS), default=list(POOLS))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    tasks = []
    for pool in args.pools:
        pattern = POOLS[pool]
        files = sorted(glob.glob(pattern))
        if not files:
            raise RuntimeError(f"No tar files for {pool}: {pattern}")
        for path in files:
            st = os.stat(path)
            tasks.append((pool, path, st.st_size, st.st_mtime_ns))
    if args.limit:
        tasks = tasks[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    recorded = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            recorded[row["path"]] = row
    todo = [task for task in tasks if not (task[1] in recorded
            and recorded[task[1]]["size"] == task[2]
            and recorded[task[1]]["mtime_ns"] == task[3])]
    print(json.dumps(dict(event="start", total=len(tasks), complete=len(tasks)-len(todo),
                          remaining=len(todo), workers=args.workers)), flush=True)

    with args.output.open("a") as output, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(count_tar, task): task for task in todo}
        done = len(tasks) - len(todo)
        for future in as_completed(futures):
            row = future.result()
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            recorded[row["path"]] = row
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(json.dumps(dict(event="progress", time=time.time(),
                                      complete=done, total=len(tasks))), flush=True)

    totals = {}
    for pool in args.pools:
        rows = [recorded[task[1]] for task in tasks if task[0] == pool]
        totals[pool] = dict(tars=len(rows), samples=sum(row["samples"] for row in rows),
                            bytes=sum(row["size"] for row in rows))
    print(json.dumps(dict(event="complete", pools=totals), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
