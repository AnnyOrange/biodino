#!/usr/bin/env python3
"""Run the identity-locked MoNuSeg v4 segmentation component on hxw."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

CUDA_LIBS = (
    "/home/xzj/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
    "/home/xzj/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia/cuda_cupti/lib",
)
CAMPAIGNS = {
    "l": (Path("/data/hs6_l_5tb_nogram_eval_20260921"), {29767 + 488 * i for i in range(66)}),
    "hplus": (Path("/data/hs6_hplus_5tb_eval_20260921"), {0, 487, 975, 1463, 1951, 2439, 2927, 3415, 3903, 4391, 4879, 5367, 5855, 6343, 6831, 7319, 7807, 8295, 8783, 9271, 9759, 10247, 10735, 11223, 11711, 12199, 12687}),
}
SNAPSHOT = Path("/data/hs6_l_5tb_nogram_eval_20260921/bin/v4_monuseg_source_snapshot_20260924")
DEFAULT_SCRATCH = Path("/home/xzj/hs6_5tb_v3_scratch")
os.environ["DINOV3_CODE_ROOT"] = str(SNAPSHOT)
sys.path[:0] = [str(SNAPSHOT), str(SNAPSHOT / "evaluation_external/benchmark_model")]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True, choices=CAMPAIGNS)
    p.add_argument("--point", required=True, type=int)
    p.add_argument("--dataset", required=True, choices=("conic", "pannuke", "livecell", "tissuenet", "cellpose", "multimodal_cellseg", "monuseg"))
    p.add_argument("--fold", default=None, help="Mandatory explicit fold for PanNuke")
    p.add_argument("--gpu", required=True, type=int, choices=(0, 1, 2, 3, 4, 6, 7))
    p.add_argument("--resume-existing", action="store_true", help="Retry a verified interrupted cell without discarding extracted features")
    p.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH,
                   help="Temporary feature-cache filesystem; validated cells delete it immediately")
    args = p.parse_args()
    import torch
    import numpy
    import sklearn
    runtime = {"torch": torch.__version__, "numpy": numpy.__version__, "sklearn": sklearn.__version__}
    if not torch.__version__.startswith("2.10.0"):
        p.error(f"Formal-v3 requires frozen PyTorch 2.10.0, found {torch.__version__}")
    for directory in CUDA_LIBS:
        if not Path(directory).is_dir():
            p.error(f"Required CUDA runtime libraries unavailable: {directory}")
    from benchmark_eval.rules_dense_preflight import audit_dense
    from dinov3.eval.bio_frozen_eval.external_fm_protocol import PANNUKE_PROTOCOLS, validate_best_head

    root, points = CAMPAIGNS[args.campaign]
    if args.point not in points:
        adapter = root / "adapters" / str(args.point)
        ready = ((adapter / "verified_sha256.json").is_file() if args.point >= 13663
                 else (adapter / "checkpoint.pth").is_file())
        if args.campaign != "hplus" or not ready:
            p.error("Checkpoint not registered or not SHA-verified for this campaign")
    if args.dataset == "pannuke" and args.fold not in PANNUKE_PROTOCOLS:
        p.error("PanNuke requires one approved explicit rotation")
    if args.dataset != "pannuke" and args.fold is not None:
        p.error("Fold only applies to PanNuke")
    if not SNAPSHOT.joinpath("source_snapshot.json").is_file():
        p.error("Frozen source snapshot missing")
    ckpt = root / "adapters" / str(args.point) / "checkpoint.pth"
    if not ckpt.is_file() or not ckpt.is_symlink():
        p.error(f"Full transferred checkpoint missing: {ckpt}")
    snapshot_sha = digest(ckpt)
    split = args.fold or ("official-baseline-fold0-nested-v1" if args.dataset == "conic" else "formal-static-v1")
    spec = audit_dense(args.dataset, Path("/data/benchmark"), protocol=split)
    if spec["status"] != "PASS":
        p.error(f"Dataset identity audit failed: {spec}")
    name = f"point_{args.point}__{args.dataset}__{split}"
    cell = root / "v3" / "cells" / name
    cache_root = args.scratch_root / args.campaign / name
    if cache_root == args.scratch_root or not cache_root.is_relative_to(args.scratch_root):
        p.error(f"Unsafe cache root: {cache_root}")
    report = cell / "validation_report.json"
    if report.exists():
        if json.loads(report.read_text()).get("status") == "VALID_COMPLETE":
            print(f"Already validated: {report}", flush=True)
            return
        p.error(f"Existing cell report must be inspected: {report}")
    if cell.exists() and any(cell.iterdir()):
        if not args.resume_existing:
            p.error(f"Existing incomplete cell must be inspected: {cell}")
        manifest_path = cell / "command_manifest.json"
        if not manifest_path.is_file():
            p.error("Cannot resume a cell without an earlier manifest")
        earlier = json.loads(manifest_path.read_text())
        if (earlier.get("checkpoint_sha256") != snapshot_sha or earlier.get("split") != spec
                or earlier.get("point") != args.point or earlier.get("campaign") != args.campaign):
            p.error("Checkpoint/dataset preflight changed since the interrupted run")
        if earlier.get("runtime", {}).get("torch") != runtime["torch"]:
            # Previous diagnostic fits used a different PyTorch version.
            # Keep them as evidence, but force all six formal fits to rerun.
            for result in (cell / "results").rglob("results.json"):
                diagnostic = result.with_name("results.diagnostic_env28.json")
                if diagnostic.exists() or result.is_symlink():
                    p.error(f"Diagnostic result archive already exists: {diagnostic}")
                result.rename(diagnostic)
            print("Archived non-v3-runtime probe results; rerunning six independent fits", flush=True)
    cell.mkdir(parents=True, exist_ok=True)
    (root / "v3" / "checkpoint_holds").mkdir(parents=True, exist_ok=True)
    (root / "v3" / "checkpoint_holds" / f"point_{args.point}.hold").touch(exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline",
           "--datasets", args.dataset, "--checkpoints-dir", str(root / "adapters"),
           "--checkpoint-iters", str(args.point), "--train-config", str(root / "source/config.yaml"),
           "--data-root-base", "/data/benchmark/segmentation", "--protocol", "manual",
           "--dataset-split-protocol", split, "--feature-img-size", str(spec["image_size"]),
           "--resize-mode", spec["resize_mode"], "--layer-preset", "last1",
           "--feature-batch-size", "32", "--autocast-dtype", "bf16", "--feature-num-workers", "2",
           "--probe-epoch-grid", "20", "50", "--probe-seeds", "0", "1", "2",
           "--probe-eval-every", "1", "--probe-batch-size", "32", "--probe-num-workers", "2",
           "--probe-lr", "0.001", "--probe-weight-decay", "0.0001",
           "--probe-class-weight-mode", "sqrt_inverse" if args.dataset == "conic" else "none",
           "--channel-policy", "auto", "--chunked-cache", "--no-compress-cache",
           "--cache-root", str(cache_root), "--output-root", str(cell / "results"),
           "--run-name", "primary_last", "--gpu", str(args.gpu)]
    manifest = {"campaign": args.campaign, "point": args.point, "gpu": args.gpu,
                "checkpoint_sha256": snapshot_sha, "split": spec,
                "runtime": runtime,
                "temporary_cache_root": str(cache_root),
                "protocol_sha256": digest(SNAPSHOT / "Evaluation Rules/protocol_v4.json"),
                "source_snapshot_sha256": digest(SNAPSHOT / "source_snapshot.json"),
                "command": cmd, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    if args.resume_existing:
        manifest["previous_command"] = earlier["command"]
        manifest["resume_reason"] = "Verified interrupted formal cell; retry under the unchanged pinned runtime and protocol"
    (cell / "command_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    env = os.environ.copy()
    env.update(PYTHONPATH=f"{SNAPSHOT}:{SNAPSHOT / 'evaluation_external/benchmark_model'}",
               CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    env["LD_LIBRARY_PATH"] = ":".join(CUDA_LIBS) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    print(f"START {name} on gpu {args.gpu}", flush=True)
    with (cell / "pipeline.log").open("a" if args.resume_existing else "w") as log:
        if args.resume_existing:
            log.write("\nRESUME WITH EXISTING CACHE: verified source and preflight unchanged\n")
        ret = subprocess.run(cmd, cwd=SNAPSHOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if ret:
        print(f"FAILED {name}: exit {ret}; inspect {cell / 'pipeline.log'}", flush=True)
        sys.exit(ret)
    paths = sorted((cell / "results").rglob("results.json"))
    if len(paths) != 6:
        raise RuntimeError(f"Expected exactly six independent E20/E50 x seed0/1/2 fits, got {len(paths)}")
    signatures = set()
    hashes = {}
    for result in paths:
        row = json.loads(result.read_text())
        meta = row["_meta"]
        validate_best_head(meta)
        if meta["probe_batch_size"] != 32 or meta["probe_eval_every"] != 1:
            raise RuntimeError(f"Wrong v3 dense settings: {result}")
        if meta["full_train_samples"] != spec["counts"]["train"] or meta["used_train_samples"] != spec["counts"]["train"]:
            raise RuntimeError(f"Training-split count mismatch: {result}")
        if not math.isfinite(row["test"]["mDice"]):
            raise RuntimeError(f"Invalid test metric: {result}")
        signatures.add((meta["probe_epochs"], meta["seed"]))
        hashes[str(result.relative_to(cell))] = digest(result)
    if signatures != {(epochs, seed) for epochs in (20, 50) for seed in (0, 1, 2)}:
        raise RuntimeError(f"Incomplete independent probe coverage: {signatures}")
    report.write_text(json.dumps({"status": "VALID_COMPLETE", "scope": "FORMAL_V4_SEGMENTATION_COMPONENT_ONLY",
                                  "full_v4_aggregate_allowed": False, "result_sha256": hashes,
                                  "command_manifest_sha256": digest(cell / "command_manifest.json"),
                                  "validated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2) + "\n")
    if cache_root.exists():
        shutil.rmtree(cache_root)
    print(f"VALID_COMPLETE {name}; deleted temporary cache {cache_root}", flush=True)


if __name__ == "__main__":
    main()
