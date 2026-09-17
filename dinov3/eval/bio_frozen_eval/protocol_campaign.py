"""Auditable candidate sweeps and one-way protocol freezing on existing probes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import socket
import subprocess
import sys
import traceback

import numpy as np

from .candidate_datasets import (CandidateCountDataset, build_idcia_manifest,
                                 digest, file_digest, validate_records)
from .probes import run_regression_probe_split
from .cellfmcount import CellFMCountDataset, build_cellfm_manifest

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = ROOT / "Evaluation Rules/unprotocolized_protocols.json"
ADAPTERS = {"idcia-condition-count": (build_idcia_manifest, CandidateCountDataset),
            "cellfmcount-dapi-count": (build_cellfm_manifest, CellFMCountDataset)}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def clean_metrics(metrics):
    return {key: float(value) if np.isfinite(value) else None for key, value in metrics.items()}


def code_identity(require_clean=True):
    git = ["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT)]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output([*git, "status", "--porcelain",
                                       "--untracked-files=normal"], text=True).strip()
    if require_clean and changed:
        raise ValueError("Experiments require a clean, committed checkout; use a detached worktree")
    return {"hostname": socket.gethostname(), "git_commit": commit,
            "code_clean": not bool(changed), "python": sys.executable,
            "started_utc": datetime.now(timezone.utc).isoformat()}


def validate_sync(path, commit, registry_sha):
    data = json.loads(Path(path).read_text())
    hosts = {record["host"]: record for record in data["machines"]}
    expected = {"local", "5090-hxw-xzj", "5090-lyx-xr", "suxin-8H100-1"}
    if len(data["machines"]) != 4 or set(hosts) != expected:
        raise ValueError("Sync gate needs all four requested machines")
    for host, record in hosts.items():
        if not (record["git_commit"] == commit and record["code_clean"]
                and record["registry_sha256"] == registry_sha and record["environment_pass"]):
            raise ValueError(f"Code/registry/environment sync mismatch: {host}")
    return file_digest(path)


def feature_layers(depth, preset):
    # Reuse the dense task's architecture-dependent even4 definition.
    from dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline import _even4_layers
    if depth < 4:
        raise ValueError("Candidate features require a backbone depth >= 4")
    if preset == "last":
        return [depth - 1]
    if preset == "last4":
        return list(range(max(0, depth - 4), depth))
    if preset == "4-even":
        return _even4_layers(depth)
    raise ValueError(f"Unknown feature preset {preset}")


def choose_sweep(rows, features, alphas):
    expected = {(feature, float(alpha)) for feature in features for alpha in alphas}
    actual = [(row["feature"], float(row["alpha"])) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Sweep is incomplete or contains duplicate candidates")
    if any(row["status"] != "SUCCESS" or row["metrics"].get("mae") is None
           or not np.isfinite(row["metrics"]["mae"]) for row in rows):
        raise ValueError("Cannot freeze a failed/nonfinite sweep")
    # Stable declared search order resolves exact ties.
    return min(rows, key=lambda row: (row["metrics"]["mae"], features.index(row["feature"]),
                                     list(alphas).index(row["alpha"])))


def validate_frozen(frozen, commit, registry_sha, manifest, budget):
    signature = frozen.get("protocol_sha256")
    if signature != digest({k: v for k, v in frozen.items() if k != "protocol_sha256"}):
        raise ValueError("Frozen protocol checksum mismatch")
    if frozen["git_commit"] != commit or frozen["registry_sha256"] != registry_sha:
        raise ValueError("Frozen protocol code/registry mismatch")
    if frozen["manifest_sha256"] != digest(manifest):
        raise ValueError("Frozen split/sample/annotation/content manifest changed")
    if frozen["selection_budget"] != "1TB" or budget not in ("1TB", "5TB", "20TB"):
        raise ValueError("Invalid selection/evaluation model budget")
    if frozen["status"] != "FROZEN_DEVELOPMENT_PROTOCOL":
        raise ValueError("Protocol has not been frozen")


def validate_model(models, model_id, family, budget, checkpoint_sha, config_sha):
    model = models["models"].get(model_id)
    if model is None or not (model["model_family"] == family and model["model_budget"] == budget
                            and model["checkpoint_sha256"] == checkpoint_sha
                            and model["train_config_sha256"] == config_sha and model["branch"] == "teacher"):
        raise ValueError("Checkpoint/config identity is not admitted for this model family/budget")
    return model


def validate_previous(previous, frozen, budget):
    expected = {"5TB": "1TB", "20TB": "5TB"}.get(budget)
    if expected is None:
        return
    if not previous or not (previous["status"] == "SUCCESS" and previous["model_budget"] == expected
                            and previous["protocol_sha256"] == frozen["protocol_sha256"]
                            and previous["git_commit"] == frozen["git_commit"]
                            and previous["dataset"] == frozen["dataset"]
                            and previous["model_family"] == frozen["model_family"]
                            and previous["seed"] == frozen["seed"]):
        raise ValueError(f"{budget} requires successful matching frozen-protocol {expected} evaluation")


def software_versions():
    return {name: version(name) for name in
            ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "Pillow", "omegaconf")}


def validate_software(frozen, actual):
    if frozen.get("software_versions") != actual:
        raise ValueError("Numerical environment differs from 1TB protocol selection; standardize it first")


def extract_bank(args, config, manifest, feature, directory, identity):
    import torch
    from .encoder import Dinov3CkptEncoder, extract_features
    directory.mkdir(parents=True, exist_ok=False)
    encoder = Dinov3CkptEncoder(Path(args.checkpoint), Path(args.train_config), args.device,
                               1, True, torch.bfloat16,
                               image_size=config["preprocessing"]["image_size"],
                               resize_size=config["preprocessing"]["resize_size"])
    depth = len(encoder.model.backbone.blocks)
    if depth != identity["model_depth"]:
        raise ValueError("Loaded backbone depth differs from admitted checkpoint identity")
    layers = feature_layers(depth, feature)
    encoder.model.n_last_blocks = layers
    bank = {}
    for split in ("train", "val", "test"):
        # Selection never extracts or scores the held-out test split.
        if args.command == "sweep" and split == "test":
            continue
        dataset = ADAPTERS[args.dataset][1](args.benchmark_root, manifest, split)
        bank[split] = extract_features(dataset, encoder, directory / f"{split}.npz",
                                       args.batch_size, args.workers, True,
                                       args.model_id, save_paths=True)
    write_new(directory / "provenance.json", {**identity, "feature": feature,
              "zero_based_layers": layers, "batch_size": args.batch_size,
              "preprocessing": config["preprocessing"], "manifest_sha256": digest(manifest),
              "embedding": "selected CLS concatenation + final selected patch mean",
              "autocast_dtype": "bf16", "storage_dtype": "float16", "normalization": "L2",
              "channel_policy": "auto; RGB images only; no channel TTA",
              "cache_policy": "fresh extraction only; no weak cache reuse"})
    del encoder
    torch.cuda.empty_cache()
    return bank, layers


def run(args):
    registry = json.loads(Path(args.registry).read_text())
    config = registry["datasets"][args.dataset]
    if args.dataset not in ADAPTERS:
        raise ValueError("Dataset has no runnable candidate adapter")
    if args.command != "preflight" and config["status"] != "CANDIDATE_READY":
        raise ValueError(f"Dataset experiment admission blocked: {config['status']}")
    manifest = ADAPTERS[args.dataset][0](args.benchmark_root, config["seed"])
    validate_records(manifest["records"])
    output = Path(args.output)
    if args.command == "preflight":
        write_new(output / "dataset_manifest.json", manifest)
        write_new(output / "preflight.json", {"status": "PASS_CANDIDATE_ONLY",
                  "manifest_sha256": digest(manifest), "registry_sha256": file_digest(args.registry),
                  "stats": manifest["stats"], "limitations": manifest["limitations"]})
        print(json.dumps(manifest["stats"]))
        return
    identity = code_identity()
    if config.get("license_status") == "UNVERIFIED_NO_FORMAL_LAUNCH":
        raise ValueError("Dataset usage terms unverified: preflight only, no experiment admission")
    registry_sha = file_digest(args.registry)
    sync_sha = validate_sync(args.sync_manifest, identity["git_commit"], registry_sha)
    if not Path(args.checkpoint).is_file() or not Path(args.train_config).is_file():
        raise ValueError("Candidate runner requires a consolidated checkpoint file and train config")
    checkpoint_sha, train_config_sha = file_digest(args.checkpoint), file_digest(args.train_config)
    model_registry_path = Path(args.registry).parent / registry["model_registry"]
    model_registry = json.loads(model_registry_path.read_text())
    model = validate_model(model_registry, args.model_id, args.model_family, args.model_budget,
                           checkpoint_sha, train_config_sha)
    identity.update(dataset=args.dataset, model_id=args.model_id, model_budget=args.model_budget,
                    model_family=args.model_family, model_depth=model["depth"],
                    checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=checkpoint_sha,
                    train_config_sha256=train_config_sha, registry_sha256=registry_sha,
                    model_registry_sha256=file_digest(model_registry_path),
                    software_versions=software_versions(),
                    sync_manifest_sha256=sync_sha, seed=config["seed"], output_path=str(output.resolve()))
    if args.command == "sweep" and args.model_budget != "1TB":
        raise ValueError("Only the 1TB selection model may search protocols")
    if args.command == "evaluate":
        frozen = json.loads(Path(args.frozen).read_text())
        validate_frozen(frozen, identity["git_commit"], registry_sha, manifest, args.model_budget)
        validate_software(frozen, identity["software_versions"])
        if frozen["dataset"] != args.dataset or frozen["model_family"] != args.model_family:
            raise ValueError("Frozen dataset/model family differs (never substitute L for S+/H+)")
        if frozen["batch_size"] != args.batch_size:
            raise ValueError("Extraction batch differs from frozen protocol")
        if args.model_budget == "1TB" and checkpoint_sha != frozen["selection_checkpoint_sha256"]:
            raise ValueError("1TB evaluation checkpoint differs from selection checkpoint")
        if args.model_budget == "1TB" and train_config_sha != frozen["selection_train_config_sha256"]:
            raise ValueError("1TB training config differs from protocol selection config")
        if frozen["model_registry_sha256"] != identity["model_registry_sha256"]:
            raise ValueError("Model identity registry differs from protocol selection registry")
        previous = json.loads(Path(args.previous_result).read_text()) if args.previous_result else None
        validate_previous(previous, frozen, args.model_budget)
    else:
        frozen = None
    write_new(output / "run_manifest.json", {**identity, "status": "PREFLIGHT_PASS",
                                            "dataset_manifest_sha256": digest(manifest)})
    write_new(output / "dataset_manifest.json", manifest)
    rows = []
    try:
        if args.command == "sweep":
            for feature in config["features"]:
                bank, layers = extract_bank(args, config, manifest, feature, output / feature, identity)
                for alpha in config["alphas"]:
                    result = run_regression_probe_split(*bank["train"], *bank["val"], alpha=alpha)
                    rows.append({"feature": feature, "layers": layers, "alpha": alpha,
                                 "metrics": clean_metrics(result.metrics), "status": "SUCCESS"})
                    write_new(output / f"candidate_{len(rows):03d}.json", {**identity, **rows[-1]})
            winner = choose_sweep(rows, config["features"], config["alphas"])
            write_new(output / "full_sweep.json", {**identity, "rows": rows, "status": "SUCCESS"})
            frozen = {"status": "FROZEN_DEVELOPMENT_PROTOCOL", "dataset": args.dataset,
                      "selection_budget": "1TB", "selection_model_id": args.model_id,
                      "model_family": args.model_family, "git_commit": identity["git_commit"],
                      "registry_sha256": registry_sha, "manifest_sha256": digest(manifest),
                      "selection_checkpoint_sha256": checkpoint_sha,
                      "selection_train_config_sha256": train_config_sha,
                      "model_registry_sha256": identity["model_registry_sha256"],
                      "software_versions": identity["software_versions"],
                      "full_sweep_sha256": file_digest(output / "full_sweep.json"),
                      "batch_size": args.batch_size, "config": config, "selected": winner,
                      "selection_split": "val", "test_used_for_selection": False,
                      "seed": config["seed"], "fit_policy": "frozen train only; do not refit on validation"}
            frozen["protocol_sha256"] = digest(frozen)
            write_new(output / "frozen_protocol.json", frozen)
            result = {**identity, "status": "SUCCESS_PROTOCOL_FROZEN", "selected": winner,
                      "protocol_sha256": frozen["protocol_sha256"], "rows": len(rows)}
        else:
            winner = frozen["selected"]
            bank, layers = extract_bank(args, frozen["config"], manifest, winner["feature"],
                                        output / winner["feature"], identity)
            if layers != winner["layers"]:
                raise ValueError("Selected layer indices changed; frozen features must match at every budget")
            probe = run_regression_probe_split(*bank["train"], *bank["test"], alpha=winner["alpha"])
            result = {**identity, "status": "SUCCESS", "protocol_sha256": frozen["protocol_sha256"],
                      "feature": winner["feature"], "layers": layers, "alpha": winner["alpha"],
                      "metrics": clean_metrics(probe.metrics), "n_train": probe.n_train,
                      "n_test": probe.n_test, "aggregation": frozen["config"]["aggregation"]}
        write_new(output / "result.json", result)
        print(json.dumps(result))
    except Exception as exc:
        write_new(output / "failure.json", {**identity, "status": "FAILURE", "rows": rows,
                                            "error": repr(exc), "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "sweep", "evaluate"))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--dataset", default="idcia-condition-count")
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sync-manifest")
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-config")
    parser.add_argument("--model-id")
    parser.add_argument("--model-family", choices=("hs6-S+", "hs6-H+"))
    parser.add_argument("--model-budget", choices=("1TB", "5TB", "20TB"))
    parser.add_argument("--frozen")
    parser.add_argument("--previous-result")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command != "preflight":
        for name in ("sync_manifest", "checkpoint", "train_config", "model_id", "model_family", "model_budget"):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required")
        if args.command == "evaluate" and not args.frozen:
            parser.error("--frozen is required for evaluation")
    try:
        run(args)
    except Exception as exc:
        path = Path(args.output) / "failure.json"
        if path.exists():
            path = path.with_name("failure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
        failure = {"status": "FAILURE", "hostname": socket.gethostname(),
                   "dataset": args.dataset, "model_id": args.model_id,
                   "model_family": args.model_family, "model_budget": args.model_budget,
                   "checkpoint": args.checkpoint, "output_path": str(Path(args.output).resolve()),
                   "command": vars(args), "error": repr(exc), "traceback": traceback.format_exc()}
        try:
            failure.update(code_identity(require_clean=False))
        except (OSError, subprocess.CalledProcessError):
            failure["git_commit"] = None
        write_new(path, failure)
        raise


if __name__ == "__main__":
    main()
