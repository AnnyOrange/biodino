#!/usr/bin/env python3
"""Fit the shared deterministic LC25000 probe on published original-tile groups."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from dinov3.eval.bio_frozen_eval.group_keys import LC25000_GROUPS_SHA256, group_keys
from dinov3.eval.bio_frozen_eval.make_group_splits import group_split_indices, load_split
from dinov3.eval.bio_frozen_eval.probes import run_classification_probe_split
from dinov3.eval.bio_frozen_eval.registry import build_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/02_eval_runs/unified_paper"
INPUT = ROOT / "outputs/02_eval_inputs/formal_v4"
PROTOCOL = "lc25000-clean-original-tile-80-20-seed0-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_queue():
    config = json.loads((OUT / "config.json").read_text())
    split = load_split("lc25000")
    assert split["annotation_sha256"] == LC25000_GROUPS_SHA256
    tasks = []
    for spec in config["models"]:
        model = spec["model"]
        source = OUT / "results" / model / "retrieval_clustering/lc25000"
        key = f"{model}__global__lc25000"
        tasks.append({
            "key": key, "model": model, "task": "classification", "dataset": "lc25000",
            "entrypoint": Path(__file__).name, "args": ["--model", model],
            "execution_mode": "deterministic_frozen_feature_probe",
            "claim_key": f"lc25000_grouped_{model}",
            "allow_feature_only_remote_host": True, "min_free_mib": 0,
            "requires": [str(source / "features.npz"), str(source / "component_result.json")],
            "result": str(OUT / "results" / model / "global/lc25000/component_result.json"),
            "log": str(OUT / "logs" / f"{key}.log"),
        })
    queue = OUT / "lc25000_grouped_queue.json"
    save(queue, {"tasks": tasks, "protocol": PROTOCOL,
                 "split": str(Path(__file__).resolve().parents[1] / "dinov3/eval/bio_frozen_eval/splits/lc25000.json"),
                 "annotation_sha256": LC25000_GROUPS_SHA256,
                 "scope": "All configured models; CPU-only shared deterministic probe after validated frozen feature extraction"})
    print("PREPARED", queue, len(tasks), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--prepare-queue", action="store_true")
    args = parser.parse_args()
    if args.prepare_queue:
        prepare_queue()
        return
    if not args.model:
        parser.error("--model is required unless --prepare-queue is used")
    from unified_paper_gpu_admission import reserve_for_current_command
    reserve_for_current_command()
    config = json.loads((OUT / "config.json").read_text())
    spec = next(model for model in config["models"] if model["model"] == args.model)
    source = OUT / "results" / args.model / "retrieval_clustering/lc25000"
    feature_path = source / "features.npz"
    feature_result = json.loads((source / "component_result.json").read_text())
    assert feature_result["model"] == args.model and feature_result["checkpoint"] == spec["checkpoint"]
    source_cache = feature_result["feature_cache"]
    if source_cache != str(feature_path):
        assert spec.get("execution_host") and source_cache.endswith(f"/results/{args.model}/retrieval_clustering/lc25000/features.npz")
    assert feature_result["feature_sha256"] == sha(feature_path)
    out = OUT / "results" / args.model / "global/lc25000"
    target = out / "component_result.json"
    if target.exists():
        existing = json.loads(target.read_text())
        assert existing["feature_sha256"] == feature_result["feature_sha256"]
        assert existing["protocol"] == PROTOCOL and existing["checkpoint"] == spec["checkpoint"]
        return
    manifest = json.loads((INPUT / "lc25000_retrieval_manifest.json").read_text())
    sample_ids = [row["sample_id"] for row in manifest["records"]]
    labels = np.asarray([row["label"] for row in manifest["records"]], dtype=np.int64)
    with np.load(feature_path) as pack:
        assert str(pack["model"]) == args.model
        assert pack["paths"].astype(str).tolist() == sample_ids
        assert np.array_equal(pack["labels"], labels)
        features = pack["features"].astype(np.float32)
    assert features.shape[0] == 25000 and np.isfinite(features).all()
    dataset, task = build_dataset("lc25000", "train", None, None, benchmark_root="/mnt/huawei_deepcad/benchmark")
    assert task == "classification"
    assert [str(path) for path, _ in dataset.samples] == sample_ids
    train, test = group_split_indices("lc25000", dataset)
    groups = np.asarray(group_keys("lc25000", dataset))
    assert len(set(groups[train]) & set(groups[test])) == 0
    assert len(train) == 19949 and len(test) == 5051
    with threadpool_limits(limits=2):
        result = run_classification_probe_split(features[train], labels[train], features[test], labels[test])
    assert result.predictions is not None and result.n_test == len(test)
    out.mkdir(parents=True, exist_ok=True)
    prediction = out / "test_predictions.npz"
    temporary = out / "test_predictions.pending.npz"
    np.savez_compressed(temporary, **result.predictions,
                        sample_id=np.asarray(sample_ids)[test], group_id=groups[test])
    temporary.replace(prediction)
    split = load_split("lc25000")
    assert split["annotation_sha256"] == LC25000_GROUPS_SHA256
    membership = {
        "protocol": PROTOCOL,
        "annotation_source": split["annotation_source"],
        "annotation_sha256": LC25000_GROUPS_SHA256,
        "split_seed": split["seed"],
        "independent_unit": split["independent_unit"],
        "train": [(sample_ids[i], int(labels[i])) for i in train],
        "test": [(sample_ids[i], int(labels[i])) for i in test],
        "test_groups": groups[test].tolist(),
        "group_note": "Published original-tile groups prevent augmented descendants crossing train/test; patient and slide identities unavailable",
    }
    sample_manifest = out / "sample_manifest.json"
    save(sample_manifest, membership)
    row = {
        **result.to_dict(), "model": args.model, "checkpoint": spec["checkpoint"],
        "dataset": "lc25000", "task": "classification", "protocol": PROTOCOL,
        "split": PROTOCOL, "training_seed": "not_applicable", "solver_seed": 0,
        "required_seeds": 0, "probe": "StandardScaler+balanced-LBFGS-LogisticRegression",
        "feature_readout": feature_result["feature_readout"],
        "image_size": feature_result["image_size"], "resize_size": feature_result["resize_size"],
        "feature_cache": str(feature_path), "feature_sha256": feature_result["feature_sha256"],
        "original_feature_cache": source_cache,
        "prediction_path": str(prediction), "prediction_sha256": sha(prediction),
        "sample_manifest": str(sample_manifest), "split_sha256": sha(sample_manifest),
        "group_note": membership["group_note"], "annotation_sha256": LC25000_GROUPS_SHA256,
        "checkpoint_selection": "fixed predeclared checkpoint",
    }
    save(target, row)
    print("COMPLETE", args.model, "lc25000", result.metrics, flush=True)


if __name__ == "__main__":
    main()
