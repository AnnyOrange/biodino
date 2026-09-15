#!/usr/bin/env python3
"""Preflight and launch the 15-model external-4 screen on 3090-qi."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/02_eval_runs/external4_hplus_fm_fixedbudget_3090qi_20260910"
CACHE = CAMPAIGN / "cache"
LOGS = CAMPAIGN / "logs"
WORKER = ROOT / "scripts/run_external4_fixedbudget_model.py"
PLAN = ROOT / "Evaluation Rules/plans/external4_hplus_fm_fixedbudget_3090qi_20260910.md"
HPLUS_CKPT = ROOT / "outputs/01_training_runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818/ckpt/15374/checkpoint.pth"
HPLUS_CONFIG = ROOT / "outputs/01_training_runs/HS6_Hplus_robust_biosafe256_gb1024_lr5e5_wu3_tw30_nosig_e15_seed0_4xH100_20260818/config.yaml"
ASSIGNMENTS = {
    2: ["hs6_hplus", "cytoself", "cytoimagenet"],
    5: ["hoptimus0", "gigapath", "uni"],
    6: ["dinov2", "mae", "siglip2", "bioclip"],
    7: ["pe", "conch", "phikon2", "virchow2", "jump_cp"],
}
EXT_PYTHON = "/home/bbnc/anaconda3/envs/siglip2_env/bin/python"
HPLUS_PYTHON = "/home/bbnc/anaconda3/envs/dinov3/bin/python"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: Path) -> str:
    head = (root / ".git/HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[len("ref: "):]
    loose = root / ".git" / ref
    if loose.exists():
        return loose.read_text().strip()
    for line in (root / ".git/packed-refs").read_text().splitlines():
        if line and not line.startswith(("#", "^")):
            value, name = line.split(" ", 1)
            if name == ref:
                return value
    raise RuntimeError(f"cannot resolve Git ref {ref}")


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for name in ("ctc", "hest", "rxrx3", "midogpp")
               for p in (CACHE/name/"images.npy", CACHE/name/"labels.npy", CACHE/name/"metadata.json") if not p.exists()]
    if missing or not (CACHE / "cache_validation.json").exists():
        raise SystemExit(f"cache preflight failed; missing={missing}")
    for path in (WORKER, PLAN, HPLUS_CKPT, HPLUS_CONFIG):
        if not path.exists() or not os.access(path, os.R_OK):
            raise SystemExit(f"unreadable required input: {path}")
    gpu_state = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True).strip().splitlines()
    parsed = {int(line.split(",", 1)[0]): line for line in gpu_state}
    for gpu in ASSIGNMENTS:
        if gpu not in parsed:
            raise SystemExit(f"GPU {gpu} absent")
        used = int(parsed[gpu].split(",")[2].strip())
        if used > 1024:
            raise SystemExit(f"GPU {gpu} is not initially free: {parsed[gpu]}")
    external_env = {**os.environ, "LD_PRELOAD": "/home/bbnc/anaconda3/envs/siglip2_env/lib/libstdc++.so.6",
                    "USE_TF": "0", "TRANSFORMERS_NO_TF": "1"}
    external_env.pop("LD_LIBRARY_PATH", None)
    versions = subprocess.check_output([
        EXT_PYTHON, "-c", "import torch,numpy,scipy,sklearn,h5py,pyarrow; print(torch.__version__,numpy.__version__,scipy.__version__,sklearn.__version__,h5py.__version__,pyarrow.__version__)"
    ], text=True, env=external_env).strip()
    manifest = {
        "campaign": CAMPAIGN.name, "status": "RUNNING", "host": platform.node(), "started_unix": time.time(),
        "assignments": ASSIGNMENTS, "tests_per_model": 4, "batch_size": 4,
        "checkpoint": str(HPLUS_CKPT), "config": str(HPLUS_CONFIG),
        "gpu_preflight": [parsed[x] for x in sorted(ASSIGNMENTS)], "dependency_versions": versions,
        "git_commit": git_commit(ROOT),
        "file_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (WORKER, PLAN, ROOT/"scripts/prepare_external4_fixedbudget_cache.py")},
        "cache_validation": json.loads((CACHE / "cache_validation.json").read_text()),
        "classification": "OBSERVATIONAL / EXPERIMENTAL_NOT_REPORTABLE",
    }
    (CAMPAIGN / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    children = []
    for gpu, models in ASSIGNMENTS.items():
        for model in models:
            python = HPLUS_PYTHON if model == "hs6_hplus" else EXT_PYTHON
            cmd = [python, str(WORKER), "--model", model, "--campaign", str(CAMPAIGN), "--batch-size", "4"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            if model != "hs6_hplus":
                env.pop("LD_LIBRARY_PATH", None)
                env["LD_PRELOAD"] = "/home/bbnc/anaconda3/envs/siglip2_env/lib/libstdc++.so.6"
                env["USE_TF"] = "0"
                env["TRANSFORMERS_NO_TF"] = "1"
            env["PYTHONPATH"] = ":".join((str(ROOT), "/mnt/huawei_deepcad/benchmark_model",
                                               "/mnt/huawei_deepcad/benchmark_model/_vendor/external_gapfill_py311",
                                               "/mnt/huawei_deepcad/benchmark_model/_vendor", env.get("PYTHONPATH", "")))
            log = (LOGS / f"{model}.log").open("w")
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append((model, gpu, process, log))
            print(f"[launch] gpu={gpu} model={model} pid={process.pid}", flush=True)
    exit_codes = {}
    for model, gpu, process, log in children:
        exit_codes[model] = process.wait()
        log.close()
        print(f"[exit] gpu={gpu} model={model} code={exit_codes[model]}", flush=True)
    manifest["finished_unix"] = time.time()
    manifest["elapsed_seconds"] = manifest["finished_unix"] - manifest["started_unix"]
    manifest["exit_codes"] = exit_codes
    manifest["status"] = "COMPLETE" if all(code == 0 for code in exit_codes.values()) else "FAILED_PARTIAL"
    (CAMPAIGN / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    raise SystemExit(0 if manifest["status"] == "COMPLETE" else 2)


if __name__ == "__main__":
    main()
