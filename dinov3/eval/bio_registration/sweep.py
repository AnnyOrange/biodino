"""Save complete 1TB registration sweeps, or run one admitted frozen protocol.

Run only after the authoritative benchmark commit/environment is synchronized.
Selection is development-group ONLY; final evaluation never picks its winner.
"""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

from .datasets import anhir_pairs, cima_pairs, grouped_partition, preflight_pairs
from .api import evaluate_registration


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def aggregate(records):
    if not records:
        raise ValueError("Empty evaluation partition")
    groups = sorted({r["group"] for r in records})
    group_means = [np.mean([r["metrics"]["median_rtre"] for r in records if r["group"] == group]) for group in groups]
    return {"pairs": len(records), "groups": len(groups), "mean_group_median_rtre": float(np.mean(group_means)),
            "mean_median_rtre": float(np.mean([r["metrics"]["median_rtre"] for r in records])),
            "mean_robustness": float(np.mean([r["metrics"]["robustness"] for r in records])),
            "failed_fraction": float(np.mean([not r["success"] for r in records]))}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ANHIR", "CIMA"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-family", choices=["hs6-S+", "hs6-H+"], required=True)
    parser.add_argument("--model-budget", choices=["1TB", "5TB", "20TB"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol-candidates", default="Evaluation Rules/registration_protocol_candidates.json")
    parser.add_argument("--frozen-protocol")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--sync-manifest", required=True)
    parser.add_argument("--registry", default="Evaluation Rules/unprotocolized_protocols.json")
    parser.add_argument("--model-registry", default="Evaluation Rules/unprotocolized_models.json")
    parser.add_argument("--previous")
    return parser.parse_args()


def bind_manifest(pairs, root, partition):
    from dinov3.eval.bio_frozen_eval.candidate_datasets import file_digest
    hashes, manifest = {}, []
    root = Path(root)
    for pair in pairs:
        record = {"pair_id": pair.pair_id, "group": pair.group,
                  "reference_diagonal": pair.reference_diagonal, "coordinate_unit": pair.coordinate_unit,
                  "official_status": pair.official_status, "partition": partition[pair.pair_id]}
        for field in ("source_image", "target_image", "source_landmarks", "target_landmarks"):
            path = getattr(pair, field)
            if path is None:
                raise ValueError("Public registration evaluation requires source AND target annotations")
            if path not in hashes:
                hashes[path] = file_digest(path)
            record[field] = str(path.relative_to(root))
            record[field + "_sha256"] = hashes[path]
        manifest.append(record)
    return manifest


def protocol_signature(frozen):
    fields = ["dataset", "code_commit", "registry_hash", "manifest_hash", "model_family",
              "environment_sha256", "selected", "selected_zero_based_layers", "seed",
              "aggregation", "metric", "preprocessing", "evaluator",
              "selection_checkpoint_sha256", "selection_train_config_sha256"]
    return canonical_hash({field: frozen[field] for field in fields})


def run(args):
    from dinov3.eval.bio_frozen_eval.protocol_campaign import code_identity, validate_model, validate_sync
    from dinov3.eval.bio_frozen_eval.candidate_datasets import file_digest
    from dinov3.eval.bio_frozen_eval.expansion_campaign import environment_identity
    identity = code_identity()
    commit = identity["git_commit"]
    if commit != args.expected_commit:
        raise ValueError("Expected benchmark commit mismatch")
    if args.model_budget != "1TB" and not args.frozen_protocol:
        raise ValueError("Only 1TB may select a protocol")
    sync_hash = validate_sync(args.sync_manifest, commit, file_digest(args.registry))
    environment = environment_identity()
    checkpoint_hash, train_config_hash = file_digest(args.checkpoint), file_digest(args.train_config)
    admitted = validate_model(json.loads(Path(args.model_registry).read_text()), args.model_id,
        args.model_family, args.model_budget, checkpoint_hash, train_config_hash)
    candidate_bytes = Path(args.protocol_candidates).read_bytes()
    registry_hash = hashlib.sha256(candidate_bytes).hexdigest()
    candidates = json.loads(candidate_bytes)
    from dinov3.eval.bio_frozen_eval.expansion_campaign import ROOT
    candidate_key = str(Path(args.protocol_candidates).resolve().relative_to(ROOT))
    for row in json.loads(Path(args.sync_manifest).read_text())["machines"]:
        if row.get("environment_sha256") != environment["environment_sha256"]:
            raise ValueError(f"Pinned evaluation environment mismatch: {row['host']}")
        if row.get("registries", {}).get(candidate_key) != registry_hash:
            raise ValueError(f"Registration candidate registry not synchronized: {row['host']}")
    seed = candidates["search"]["seed"]
    pairs = anhir_pairs(args.root) if args.dataset == "ANHIR" else cima_pairs(args.root, "scale-25pc")
    checked = preflight_pairs(pairs)
    if not checked["success"]:
        raise ValueError(f"Data preflight failed: {checked['failures']}")
    excluded = [p.group for p in pairs if p.group.startswith(("lung-", "mammary-"))] if args.dataset == "ANHIR" else []
    partition = grouped_partition(pairs, seed, exclude_development_groups=excluded)
    manifest = bind_manifest(pairs, args.root, partition)
    manifest_hash = canonical_hash(manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Refuse accidental overwrite of full-sweep evidence.
    with (output / "invocation.json").open("x") as handle:
        json.dump(vars(args) | {"hostname": socket.gethostname(), "code_commit": commit,
            "registry_hash": registry_hash, "manifest_hash": manifest_hash, "preflight": checked}, handle, indent=2)
    from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone
    from dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline import _even4_layers
    backbone = load_dinov3_backbone(args.checkpoint, args.train_config, device="cuda", freeze=True)
    if len(backbone.blocks) != admitted["depth"]:
        raise ValueError("Loaded architecture depth differs from admitted checkpoint")
    representations = {"last": 1, "4-even": _even4_layers(len(backbone.blocks))}
    cache = {}
    provenance = identity | {"hostname": socket.gethostname(), "code_commit": commit, "registry_hash": registry_hash,
        "manifest_hash": manifest_hash, "model": args.model_id, "checkpoint": args.checkpoint,
        "model_family": args.model_family, "model_budget": args.model_budget, "dataset": args.dataset, "seed": seed,
        "checkpoint_sha256": checkpoint_hash, "train_config_sha256": train_config_hash,
        "sync_manifest_sha256": sync_hash, "environment": environment,
        "environment_sha256": environment["environment_sha256"]}
    if args.frozen_protocol:
        frozen = json.loads(Path(args.frozen_protocol).read_text())
        if not frozen.get("frozen") or frozen["dataset"] != args.dataset or frozen["manifest_hash"] != manifest_hash:
            raise ValueError("Frozen protocol or data-manifest mismatch")
        if frozen["registry_hash"] != registry_hash:
            raise ValueError("Frozen registry mismatch")
        if frozen["model_family"] != args.model_family:
            raise ValueError("Use the frozen protocol of the same model family")
        if frozen["code_commit"] != commit or frozen["environment_sha256"] != environment["environment_sha256"]:
            raise ValueError("Frozen benchmark commit/environment mismatch")
        if frozen["protocol_sha256"] != protocol_signature(frozen):
            raise ValueError("Frozen protocol signature mismatch")
        actual_layers = [len(backbone.blocks) - 1] if frozen["selected"]["representation"] == "last" else representations["4-even"]
        if frozen["selected_zero_based_layers"] != actual_layers:
            raise ValueError("Frozen layer indices differ from admitted architecture")
        if args.model_budget == "1TB" and (frozen["selection_checkpoint_sha256"] != checkpoint_hash
                                           or frozen["selection_train_config_sha256"] != train_config_hash):
            raise ValueError("Frozen 1TB evaluation must use its admitted selection checkpoint")
        if args.model_budget in {"5TB", "20TB"}:
            previous = json.loads(Path(args.previous).read_text()) if args.previous else {}
            expected_budget = "1TB" if args.model_budget == "5TB" else "5TB"
            if not (previous.get("status") == "SUCCESS" and previous.get("phase") == "evaluation"
                    and previous.get("model_budget") == expected_budget and previous.get("code_commit") == commit
                    and previous.get("protocol_sha256") == frozen["protocol_sha256"]
                    and previous.get("environment_sha256") == environment["environment_sha256"]):
                raise ValueError("Budget progression requires successful same-protocol previous-budget evaluation")
        configurations = [frozen["selected"]]
        phase = "evaluation"
    else:
        configurations = [{"representation": rep, "ratio": ratio, "threshold_fraction": threshold}
            for rep, ratio, threshold in product(candidates["feature_candidates"],
                candidates["search"]["lowe_euclidean_ratio"],
                candidates["search"]["ransac_threshold_fraction_target_diagonal"])]
        phase = "development"
    summaries = []
    with (output / "pair_results.jsonl").open("x") as handle:
        for config in configurations:
            results = []
            config_id = canonical_hash(config)
            for pair in pairs:
                if partition[pair.pair_id] != phase:
                    continue
                started = time.monotonic()
                evaluated = evaluate_registration(backbone, [pair], representations[config["representation"]],
                    device="cuda", max_side=candidates["search"].get("max_side", 512),
                    ratio=config["ratio"], threshold_fraction=config["threshold_fraction"], seed=seed,
                    descriptor_cache=cache)
                result = evaluated["pair_results"][0]
                result.update(provenance | {"status": "SUCCESS", "config_id": config_id, "feature_choice": config["representation"],
                    "hyperparameters": config, "phase": phase, "output_path": str(output),
                    "total_seconds": time.monotonic() - started})
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                results.append(result)
            summaries.append({"configuration": config, "config_id": config_id, "aggregate": aggregate(results)})
    if not args.frozen_protocol:
        selected = min(summaries, key=lambda row: (row["aggregate"]["mean_group_median_rtre"], row["config_id"]))
        freeze = provenance | {"frozen": True, "selected_using_1tb": True,
            "selected": selected["configuration"], "selection_partition": "development",
            "aggregation": "mean_group_mean_pair_median_rtre_including_failed_pairs", "metric": "mean_group_median_rtre",
            "manifest": manifest, "complete_sweep": summaries, "protocol_id": selected["config_id"]}
        freeze.update({"selected_zero_based_layers": [len(backbone.blocks) - 1] if freeze["selected"]["representation"] == "last" else representations["4-even"],
            "selection_checkpoint_sha256": checkpoint_hash, "selection_train_config_sha256": train_config_hash,
            "preprocessing": candidates["preprocessing"], "evaluator": "native_dino_patch_affine_registration"})
        freeze["protocol_sha256"] = protocol_signature(freeze)
        with (output / "frozen_protocol.json").open("x") as handle:
            json.dump(freeze, handle, indent=2)
    with (output / "summary.json").open("x") as handle:
        json.dump(provenance | {"status": "SUCCESS", "phase": phase, "configurations": summaries,
            "protocol_sha256": frozen["protocol_sha256"] if args.frozen_protocol else freeze["protocol_sha256"]}, handle, indent=2)
    print(json.dumps({"output": str(output), "phase": phase, "configurations": len(summaries)}))


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAILURE", "dataset": args.dataset, "model_id": args.model_id,
                   "model_budget": args.model_budget, "command": vars(args),
                   "error": str(error), "traceback": traceback.format_exc()}
        try:
            from dinov3.eval.bio_frozen_eval.protocol_campaign import code_identity
            failure.update(code_identity(require_clean=False))
        except Exception:
            pass
        path = output / "failure.json"
        if path.exists():
            path = output / f"failure_{time.time_ns()}.json"
        with path.open("x") as handle:
            json.dump(failure, handle, indent=2)
        raise


if __name__ == "__main__":
    main()
