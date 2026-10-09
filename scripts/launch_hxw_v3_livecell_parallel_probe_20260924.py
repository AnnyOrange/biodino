#!/usr/bin/env python3
"""Run an independent LIVECell probe seed from an existing frozen feature cache."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/data/hs6_l_5tb_nogram_eval_20260921")
SOURCE = ROOT / "bin/v3_source_snapshot"
SCRATCH = Path("/home/xzj/hs6_5tb_v3_scratch/l")
PYTHON = "/home/xzj/eval_envs/hs6_protocol_v2/bin/python"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--point", type=int, required=True)
    p.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    p.add_argument("--budget", type=int, choices=(20, 50), default=50)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), default=2)
    args = p.parse_args()
    cell = ROOT / "v3/cells" / f"point_{args.point}__livecell__formal-static-v1"
    assert (cell / "command_manifest.json").is_file(), "Frozen cell manifest missing"
    assert not (cell / "validation_report.json").exists(), "Cell is already complete"
    cache = SCRATCH / cell.name / "primary_last_ampbf16_b32__pad_spformal_static_v1" / "livecell" / str(args.point)
    suffix = "config_last1_pad_s512_spformal_static_v1_ampbf16_b32.npz"
    caches = {split: cache / f"livecell_{split}_{suffix}" for split in ("train", "val", "test")}
    assert all(path.is_file() and path.stat().st_size > 1_000_000 for path in caches.values()), "Incomplete feature cache"
    out = cell / "results/primary_last_ampbf16_b32__pad_spformal_static_v1" / f"budget{args.budget}" / f"seed{args.seed}" / "livecell" / str(args.point)
    out.mkdir(parents=True, exist_ok=True)
    result = out / "results.json"
    assert not result.exists(), "Probe result already exists"
    cmd = [PYTHON, "-m", "dinov3.eval.bio_segmentation.linear_probe", "--dataset", "livecell",
           "--use-cached-features", "--train-cache", str(caches["train"]),
           "--val-cache", str(caches["val"]), "--output-dir", str(out),
           "--epochs", str(args.budget), "--batch-size", "32", "--lr", "0.001",
           "--weight-decay", "0.0001", "--num-workers", "2", "--eval-every", "1",
           "--seed", str(args.seed), "--test-cache", str(caches["test"])]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(SOURCE), CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    cuda_libs = ["/home/xzj/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
                 "/home/xzj/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia/cuda_cupti/lib"]
    env["LD_LIBRARY_PATH"] = ":".join(cuda_libs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    manifest = {"point": args.point, "gpu": args.gpu, "budget": args.budget, "seed": args.seed,
                "command": cmd, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "purpose": "Parallel independent frozen probe; parent pipeline reuses validated result"}
    (out / "parallel_probe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (out / "parallel_probe.log").open("w") as log:
        code = subprocess.run(cmd, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code or not result.is_file():
        raise SystemExit(f"Probe failed with exit {code}; inspect {out / 'parallel_probe.log'}")
    row = json.loads(result.read_text())
    meta = row.get("_meta", {})
    assert meta.get("probe_epochs") == args.budget and meta.get("seed") == args.seed
    print(f"VALID_PROBE point={args.point} budget={args.budget} seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
