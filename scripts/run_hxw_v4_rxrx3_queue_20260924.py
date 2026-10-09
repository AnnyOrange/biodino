#!/usr/bin/env python3
"""Evaluate locked RxRx3 retrieval and clustering for hxw L/H+ checkpoints."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE = Path("/data/hs6_5tb_v4_rxrx3_20260924")
CACHE = BASE / "cache/rxrx3-core"
PROTOCOL = BASE / "protocol_v4.json"
EVALUATOR = BASE / "scripts/run_external4_fixedbudget_model.py"
SOURCE = Path("/data/hs6_l_5tb_nogram_eval_20260921/bin/v4_monuseg_source_snapshot_20260924")
PYTHON = Path("/home/xzj/eval_envs/hs6_protocol_v2/bin/python")
ROOTS = {
    "hplus": Path("/data/hs6_hplus_5tb_eval_20260921"),
    "l": Path("/data/hs6_l_5tb_nogram_eval_20260921"),
}
HPOINTS = (0, 487, 975, 1463, 1951, 2439, 2927, 3415, 3903, 4391, 4879, 5367, 5855, 6343, 6831,
           7319, 7807, 8295, 8783, 9271, 9759, 10247, 10735, 11223, 11711, 12199, 12687)
SPLIT = "crispr-query-guide-plate-disjoint-all-eligible-genes-v1"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def points():
    lroot = ROOTS["l"] / "source/eval"
    lpoints = sorted(int(p.parent.name.split("_")[1])
                     for p in lroot.glob("training_*/teacher_checkpoint.pth")
                     if p.stat().st_size == 1401909871)
    for campaign, sequence in (("hplus", HPOINTS), ("l", lpoints)):
        for point in sequence:
            if campaign == "hplus" and not (ROOTS[campaign] / "adapters" / str(point) / "checkpoint.pth").is_file():
                continue
            yield campaign, point


def result_path(campaign, point):
    return BASE / campaign / f"point_{point}" / "models" / f"{campaign}_{point}" / "results.json"


def valid(path, campaign, point):
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
        test = data["tests"]["rxrx3"]
    except (ValueError, KeyError, OSError):
        return False
    return (data.get("status") == "VALID_COMPLETE" and data.get("model") == f"{campaign}_{point}"
            and data.get("batch_size") == 64 and data.get("teacher_branch") == "teacher"
            and test.get("status") == "FORMAL" and test.get("proxy") is False
            and test.get("protocol_id") == SPLIT and test.get("n_query") == 734
            and test.get("n_gallery") == 734
            and all(test.get(name) is not None for name in ("recall_at_1", "mrr", "nmi")))


def resources_ok(gpu):
    mem = next(int(line.split()[1]) / 1024**2 for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:"))
    row = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    free_gpu_mib = int(row)
    return (mem >= 70 and free_gpu_mib >= 8000
            and shutil.disk_usage("/data").free >= 120 * 2**30), mem, free_gpu_mib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, choices=(0, 1, 2, 3))
    parser.add_argument("--slot", required=True)
    args = parser.parse_args()
    if not args.slot.replace("_", "").isalnum():
        parser.error("Unsafe slot name")
    for name in ("claims", "failures", "logs"):
        (BASE / name).mkdir(parents=True, exist_ok=True)
    # The frozen evaluator names this dataset ``rxrx3``; the locked input
    # inventory is stored as ``rxrx3-core`` to distinguish it from quick screens.
    alias = CACHE.parent / "rxrx3"
    if not alias.exists():
        alias.symlink_to(CACHE.name)
    if alias.resolve() != CACHE.resolve():
        raise RuntimeError("RxRx3 evaluator cache alias points to a different inventory")
    lock = json.loads(PROTOCOL.read_text())
    expected = lock["retrieval_splits"]["rxrx3-core"]["manifest_sha256"]
    if sha(CACHE / "split_manifest.jsonl") != expected:
        raise RuntimeError("RxRx3 fixed split differs from v4 lock")
    metadata = json.loads((CACHE / "metadata.json").read_text())
    rows = metadata["rows"]
    if (metadata["protocol_id"] != SPLIT
            or sum(row["split"] == "query" for row in rows) != 734
            or sum(row["split"] == "gallery" for row in rows) != 734):
        raise RuntimeError("RxRx3 full eligible gene inventory failed")
    protocol_sha = sha(PROTOCOL)
    evaluator_sha = sha(EVALUATOR)
    data_sha = {name: sha(CACHE / name) for name in
                ("metadata.json", "split_manifest.jsonl", "images.npy", "labels.npy")}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=str(SOURCE),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    cuda_libs = [
        "/home/xzj/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
        "/home/xzj/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia/cuda_cupti/lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(cuda_libs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    while True:
        okay, mem, free_gpu = resources_ok(args.gpu)
        if not okay:
            print(f"{now()} admission_wait mem={mem:.1f} gpu_free_mib={free_gpu}", flush=True)
            time.sleep(20)
            continue
        selected = None
        for campaign, point in points():
            key = f"{campaign}__{point}"
            result = result_path(campaign, point)
            if valid(result, campaign, point) or (BASE / "failures" / f"{key}.json").exists():
                continue
            claim = BASE / "claims" / key
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            if valid(result, campaign, point):
                claim.rmdir()
                continue
            if result.exists():
                claim.rmdir()
                continue  # Incomplete prior output requires an explicit review.
            selected = campaign, point, key, claim
            break
        if selected is None:
            print(f"{now()} no_unclaimed_cells", flush=True)
            time.sleep(60)
            continue
        campaign, point, key, claim = selected
        checkpoint = ROOTS[campaign] / "adapters" / str(point) / "checkpoint.pth"
        if not checkpoint.is_file():
            claim.rmdir()
            print(f"{now()} waiting_for_checkpoint {key}", flush=True)
            time.sleep(30)
            continue
        cell = BASE / campaign / f"point_{point}"
        cell.mkdir(parents=True, exist_ok=True)
        manifest = {"protocol_id": "bio-eval-union-v4", "protocol_sha256": protocol_sha,
                    "evaluator_sha256": evaluator_sha, "dataset_sha256": data_sha,
                    "campaign": campaign, "point": point, "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha(checkpoint),
                    "train_config": str(ROOTS[campaign] / "source/config.yaml"),
                    "train_config_sha256": sha(ROOTS[campaign] / "source/config.yaml"),
                    "gpu": args.gpu, "formal_batch_size": 64, "inference_microbatch": 32,
                    "created_utc": now()}
        manifest_path = cell / "campaign_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        command = [str(PYTHON), "-u", str(EVALUATOR), "--model", f"{campaign}_{point}",
                   "--campaign", str(cell), "--cache-root", str(CACHE.parent),
                   "--datasets", "rxrx3", "--checkpoint", str(checkpoint),
                   "--train-config", str(ROOTS[campaign] / "source/config.yaml"),
                   "--device", "cuda:0", "--batch-size", "32", "--logical-batch-size", "64"]
        print(f"{now()} START {key} gpu={args.gpu}", flush=True)
        with (BASE / "logs" / f"{key}.log").open("a") as log:
            rc = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        result = result_path(campaign, point)
        if rc == 0 and valid(result, campaign, point):
            claim.rmdir()
            print(f"{now()} FINISH {key}", flush=True)
        else:
            failure = {"key": key, "returncode": rc, "result": str(result), "utc": now()}
            (BASE / "failures" / f"{key}.json").write_text(json.dumps(failure, indent=2) + "\n")
            print(f"{now()} FAILED {key} rc={rc}", flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
