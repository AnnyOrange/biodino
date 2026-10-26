#!/usr/bin/env python3
"""Record host-local evaluation runtime and check canonical entry points."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "outputs/02_eval_runs/20tb_completed_sparse_20261016"
PACKAGES = ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "Pillow",
            "transformers", "timm", "huggingface-hub", "tokenizers", "safetensors")
ENTRYPOINTS = ("dinov3.eval.bio_benchmark", "dinov3.eval.bio_frozen_eval.run_classification",
               "dinov3.eval.bio_segmentation.linear_probe")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    cfg = json.loads((CAMPAIGN / "configs" / f"{args.host}.json").read_text())
    if Path(cfg["python"]).resolve() != Path(sys.executable).resolve():
        raise RuntimeError("Running with a different Python than campaign config")
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import torch
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
                             "--format=csv,noheader"], capture_output=True, text=True)
    checks = {}
    for module in ENTRYPOINTS:
        p = subprocess.run([sys.executable, "-m", module, "--help"],
                           cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO)),
                           capture_output=True, text=True, timeout=120)
        checks[module] = {"returncode": p.returncode, "stderr_tail": p.stderr[-1000:]}
    record = {"host": args.host, "python": sys.executable, "python_version": sys.version,
              "packages": versions, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "driver": driver.stdout.strip(),
              "driver_returncode": driver.returncode, "entrypoints": checks}
    out = CAMPAIGN / "environment" / f"{args.host}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"host": args.host, "compatible": bool(record["cuda_available"] and
          driver.returncode == 0 and all(c["returncode"] == 0 for c in checks.values())),
          "environment_record": str(out)}))


if __name__ == "__main__":
    main()
