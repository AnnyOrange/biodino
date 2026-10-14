#!/usr/bin/env python3
"""Rerun interrupted 20TB CTC cells from the canonical repository source."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/01_training_runs/hs6_l_20tb_v2_recovery_fork38063_20261009"
OLD = ROOT / "outputs/02_eval_runs/v2_full_v4_resampling_cls_batch_20261012"
OUT = ROOT / "outputs/02_eval_runs/20tb_resampling_cls_ctc_v4_canonical_20261013"
ARCHIVE_DIR = ROOT / "outputs/03_source_archives/20tb_resampling_cls_ctc_v4_20261013"
ARCHIVE = ARCHIVE_DIR / "source.tar.gz"
MANIFEST = OUT / "campaign_manifest.json"
DATA_MANIFEST = ROOT / "outputs/02_eval_inputs/formal_v3/ctc_native_full/data_manifest.json"
RUNNER = ROOT / "scripts/run_method_native_ctc_canonical_v4_20261013.py"
PYTHON = Path(sys.executable)

ARMS = (
    "resampling_only_20tb_ck40015",
    "resampling_only_20tb_ck41967",
    "resampling_only_20tb_ck43919",
    "fixed_cls_w1_20tb_ck40015",
    "noGRAM20tb_ck40015",
    "noGRAM20tb_ck41967",
    "noGRAM20tb_ck43919",
)
SOURCE_PATHS = (
    Path("scripts"),
    Path("dinov3/eval"),
    Path("Evaluation Rules/06_CTC_Native_Protocol.md"),
    Path("Evaluation Rules/plans/deepcad_method_full_v4_fleet_20260928.md"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)


def source_files() -> list[Path]:
    files = []
    for rel in SOURCE_PATHS:
        path = ROOT / rel
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and
                         "__pycache__" not in p.parts and p.suffix != ".pyc")
        elif path.is_file():
            files.append(path)
    return sorted(set(files))


def checkpoint_assets() -> list[dict[str, object]]:
    rows = []
    for arm in ARMS:
        old_invocation = OLD / "claims" / f"ctc__native__{arm}" / "invocation.json"
        record = json.loads(old_invocation.read_text())
        asset = record["task"]["asset"]
        checkpoint = Path(asset["path"])
        config = Path(asset["config"])
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(f"Missing CTC input for {arm}: {checkpoint} / {config}")
        rows.append({
            "arm": arm,
            "checkpoint_id": asset["checkpoint_id"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "config": str(config),
            "config_sha256": sha256(config),
            "old_attempt_state": json.loads(
                (OLD / "claims" / f"ctc__native__{arm}" / "status.json").read_text()
            ).get("state"),
            "old_attempt_host": json.loads(
                (OLD / "claims" / f"ctc__native__{arm}" / "status.json").read_text()
            ).get("host"),
        })
    return rows


def prepare(refresh: bool = False) -> None:
    prior = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    if prior and not refresh:
        raise FileExistsError(f"Refusing to overwrite campaign: {MANIFEST}")
    if prior.get("workers") or any(task.get("state") == "RUNNING" for task in prior.get("tasks", [])):
        raise RuntimeError("Cannot refresh campaign source after a worker has started")
    if not DATA_MANIFEST.is_file():
        raise FileNotFoundError(DATA_MANIFEST)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        for rel in SOURCE_PATHS:
            archive.add(ROOT / rel, arcname=rel.as_posix(), recursive=True)
    dirty = subprocess.run(
        ["git", "status", "--short", "--", "scripts", "dinov3/eval", "Evaluation Rules"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    files = source_files()
    payload = {
        "campaign": "20TB_RESAMPLING_CLS_CTC_CANONICAL_V4",
        "protocol_id": "bio-eval-union-v4",
        "created_unix": time.time(),
        "canonical_source_root": str(ROOT),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True, capture_output=True, check=True).stdout.strip(),
        "dirty_source_inventory": dirty,
        "source_files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "source_archive": str(ARCHIVE),
        "source_archive_sha256": sha256(ARCHIVE),
        "data_manifest": str(DATA_MANIFEST),
        "data_manifest_sha256": sha256(DATA_MANIFEST),
        "migration_from": {
            "component_campaign": str(ROOT / "outputs/02_eval_runs/v2_20tb_resampling_cls_batch_20261012"),
            "extension_campaign": str(OLD),
            "reason": "Seven CTC cells were interrupted in the snapshot-bound campaign; this new campaign reruns them from canonical source without changing historical outputs.",
            "source_comparison": "CTC evaluator, core CTC modules, and protocol files match recorded snapshot hashes; run method is loaded from the canonical repository.",
        },
        "tasks": [{
            **asset,
            "output": str(OUT / "ctc" / asset["arm"]),
            "state": "QUEUED",
        } for asset in checkpoint_assets()],
        "workers": {},
    }
    if prior:
        payload["prelaunch_source_revisions"] = list(prior.get("prelaunch_source_revisions", [])) + [{
            "source_archive_sha256": prior.get("source_archive_sha256"),
            "source_files_sha256": prior.get("source_files_sha256"),
            "status": "PRELAUNCH_REVISION_NO_WORKER_STARTED",
            "recorded_unix": time.time(),
        }]
    save(MANIFEST, payload)
    print(json.dumps({"campaign": str(OUT), "archive": str(ARCHIVE),
                      "tasks": [row["arm"] for row in payload["tasks"]]}, indent=2))


def runtime_info(gpus: list[int]) -> dict[str, object]:
    import torch
    import torchvision
    import numpy
    import scipy
    import sklearn
    import PIL

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "host": platform.node(), "python": sys.executable,
        "python_version": platform.python_version(), "gpus": gpus,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "pillow": PIL.__version__,
        "transformers": version("transformers"), "timm": version("timm"),
        "huggingface_hub": version("huggingface-hub"),
        "tokenizers": version("tokenizers"), "safetensors": version("safetensors"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }


def worker(gpus: list[int]) -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Prepare the campaign first: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text())
    if sha256(ARCHIVE) != manifest["source_archive_sha256"]:
        raise RuntimeError("Canonical source archive changed after campaign lock")
    info = runtime_info(gpus)
    if not info["cuda_available"] or max(gpus) >= int(info["cuda_device_count"]):
        raise RuntimeError(f"Worker GPU selection is unavailable: {info}")
    manifest.setdefault("workers", {})[info["host"]] = info
    save(MANIFEST, manifest)

    pending = deque()
    for task in manifest["tasks"]:
        result = Path(task["output"]) / "models" / task["arm"] / "results.json"
        if result.is_file():
            record = json.loads(result.read_text())
            if record.get("status") == "VALID_COMPLETE" and len(record.get("domain_rows", [])) == 20:
                task["state"] = "VALID_COMPLETE"
                task["result_sha256"] = sha256(result)
                continue
        pending.append(task)

    active: dict[subprocess.Popen, tuple[dict[str, object], int, object]] = {}
    available_gpus = deque(gpus)
    while pending or active:
        while pending and available_gpus:
            task = pending.popleft()
            gpu = available_gpus.popleft()
            task["state"] = "RUNNING"
            task["worker_gpu"] = gpu
            task["started_unix"] = time.time()
            save(MANIFEST, manifest)
            log_path = OUT / "logs" / f"{task['arm']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("a", buffering=1)
            command = [str(PYTHON), "-u", str(RUNNER), "--arm", str(task["arm"]),
                       "--checkpoint", str(task["checkpoint"]), "--config", str(task["config"]),
                       "--step", str(task["checkpoint_id"]), "--output", str(task["output"])]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            active[process] = (task, gpu, log)
        finished = []
        for process, (task, gpu, log) in active.items():
            code = process.poll()
            if code is None:
                continue
            log.close()
            result = Path(task["output"]) / "models" / task["arm"] / "results.json"
            report = Path(task["output"]) / "validation_report.json"
            valid = False
            if code == 0 and result.is_file():
                data = json.loads(result.read_text())
                valid = data.get("status") == "VALID_COMPLETE" and len(data.get("domain_rows", [])) == 20
            task["state"] = "VALID_COMPLETE" if valid else "FAILED"
            task["returncode"] = code
            task["finished_unix"] = time.time()
            task["result_sha256"] = sha256(result) if valid else None
            task["validation_report"] = str(report) if report.is_file() else None
            save(MANIFEST, manifest)
            finished.append(process)
            available_gpus.append(gpu)
        for process in finished:
            del active[process]
        if not finished and active:
            time.sleep(10)
    states = {task["state"] for task in manifest["tasks"]}
    manifest["state"] = "VALID_COMPLETE" if states == {"VALID_COMPLETE"} else "INCOMPLETE"
    manifest["updated_unix"] = time.time()
    save(MANIFEST, manifest)
    print(json.dumps({"state": manifest["state"], "tasks": {
        task["arm"]: task["state"] for task in manifest["tasks"]}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--refresh-source", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--gpus", type=int, nargs="+")
    args = parser.parse_args()
    if args.prepare_only or args.refresh_source:
        prepare(refresh=args.refresh_source)
        return
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--worker requires unique --gpus IDs")
    worker(args.gpus)


if __name__ == "__main__":
    main()
