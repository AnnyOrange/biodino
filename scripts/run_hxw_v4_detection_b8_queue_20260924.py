#!/usr/bin/env python3
"""Run the missing matched-B8 v4 detection proxy cells on hxw."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE = Path("/data/hs6_5tb_v4_detection_queue_20260924")
SOURCE = Path("/data/hs6_l_5tb_nogram_eval_20260921/bin/v3_source_snapshot")
PYTHON = Path("/home/xzj/eval_envs/hs6_protocol_v2/bin/python")
ROOTS = {
    "hplus": Path("/data/hs6_hplus_5tb_eval_20260921"),
    "l": Path("/data/hs6_l_5tb_nogram_eval_20260921"),
}
HPOINTS = (0, 487, 975, 1463, 1951, 2439, 2927, 3415, 3903, 4391, 4879, 5367, 5855, 6343, 6831,
           7319, 7807, 8295, 8783, 9271, 9759, 10247, 10735, 11223, 11711, 12199, 12687)
DATASETS = ("bbbc038", "conic", "livecell")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lpoints():
    root = ROOTS["l"] / "source/eval"
    return tuple(sorted(int(path.parent.name.split("_")[1])
                        for path in root.glob("training_*/teacher_checkpoint.pth")
                        if path.stat().st_size == 1401909871))


def tasks():
    for campaign, points in (("hplus", HPOINTS), ("l", lpoints())):
        for dataset in DATASETS:
            for point in points:
                yield campaign, point, dataset


def output_dir(campaign, point, dataset):
    return ROOTS[campaign] / "v4/detection_b8" / f"point_{point}" / dataset


def valid(path, dataset, point):
    result = path / "results_bio_detection.json"
    if not result.is_file():
        return False
    try:
        data = json.loads(result.read_text())
    except (ValueError, OSError):
        return False
    return (data.get("dataset") == dataset and str(data.get("checkpoint")) == str(point)
            and data.get("batch_size") == 8 and data.get("epochs") == 5
            and data.get("image_size") == 224 and data.get("seed") == 0
            and "test_patch_f1" in data)


def resources_ok(gpu):
    mem = next(int(line.split()[1]) / 1024**2 for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:"))
    free_mib = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    gpu_free = int(free_mib[gpu]) / 1024
    return (mem >= 80 and shutil.disk_usage("/").free >= 180 * 2**30
            and shutil.disk_usage("/data").free >= 120 * 2**30
            and gpu_free >= 8), mem, gpu_free


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, choices=(0, 1, 2, 3))
    parser.add_argument("--slot", required=True)
    args = parser.parse_args()
    if not args.slot.replace("_", "").isalnum():
        parser.error("Unsafe slot")
    for name in ("claims", "failures", "logs"):
        (BASE / name).mkdir(parents=True, exist_ok=True)
    protocol = SOURCE / "Evaluation Rules/protocol_v4.json"
    if not protocol.is_file():
        # The code snapshot predates v4; the frozen v4 protocol is stored separately.
        protocol = Path("/data/hs6_5tb_v4_detection_queue_20260924/protocol_v4.json")
    protocol_sha = digest(protocol)
    code_sha = digest(SOURCE / "dinov3/eval/bio_detection/center_probe.py")
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
        okay, mem, gpu_free = resources_ok(args.gpu)
        if not okay:
            print(f"{now()} admission_wait mem_available_gib={mem:.1f} "
                  f"gpu_free_gib={gpu_free:.1f}", flush=True)
            time.sleep(20)
            continue
        selected = None
        for campaign, point, dataset in tasks():
            target = output_dir(campaign, point, dataset)
            if valid(target, dataset, point):
                continue
            key = f"{campaign}__{point}__{dataset}"
            if (BASE / "failures" / f"{key}.json").exists():
                continue
            claim = BASE / "claims" / key
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            if valid(target, dataset, point):
                claim.rmdir()
                continue
            selected = campaign, point, dataset, target, claim, key
            break
        if selected is None:
            print(f"{now()} no_unclaimed_cells", flush=True)
            time.sleep(60)
            continue
        campaign, point, dataset, target, claim, key = selected
        checkpoint = ROOTS[campaign] / "adapters" / str(point) / "checkpoint.pth"
        if not checkpoint.is_file():
            claim.rmdir()
            print(f"{now()} waiting_for_checkpoint {key}", flush=True)
            time.sleep(30)
            continue
        target.mkdir(parents=True, exist_ok=True)
        command = [str(PYTHON), "-u", "-m", "dinov3.eval.bio_detection.center_probe",
                   "--checkpoint", str(checkpoint), "--train-config", str(ROOTS[campaign] / "source/config.yaml"),
                   "--benchmark-root", "/data/benchmark", "--dataset", dataset,
                   "--output-dir", str(target), "--batch-size", "8", "--num-workers", "2",
                   "--image-size", "224", "--epochs", "5", "--lr", "0.001",
                   "--autocast-dtype", "bf16", "--channel-policy", "auto",
                   "--max-samples-per-split", "0", "--seed", "0",
                   "--conic-split-protocol", "official-baseline-fold0-nested-v1"]
        manifest = {"protocol_id": "bio-eval-union-v4", "protocol_sha256": protocol_sha,
                    "code_sha256": code_sha, "campaign": campaign, "point": point,
                    "dataset": dataset, "gpu": args.gpu, "command": command, "start_utc": now()}
        (target / "command_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{now()} START {key} gpu={args.gpu}", flush=True)
        with (target / "launch.log").open("a") as log:
            result = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0 and valid(target, dataset, point):
            claim.rmdir()
            print(f"{now()} FINISH {key}", flush=True)
        else:
            error = {"key": key, "returncode": result.returncode, "log": str(target / "launch.log"),
                     "utc": now()}
            (BASE / "failures" / f"{key}.json").write_text(json.dumps(error, indent=2) + "\n")
            print(f"{now()} FAILED {key} rc={result.returncode}", flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
