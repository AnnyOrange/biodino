#!/usr/bin/env python3
"""Evaluate a new frozen FM on the locked v4 RxRx3-core split."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score

from run_new_fm_v4_linear_20261016 import MODELS, NewEncoder, ROOT
from benchmark_eval.encoders import MODEL_REGISTRY
from run_fm_rules import StandardExternalEncoder


PROTOCOL = "crispr-query-guide-plate-disjoint-all-eligible-genes-v1"
INPUT = ROOT / "outputs/02_eval_inputs/formal_v3/rxrx3-core"
OUTPUT = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS) + tuple(MODEL_REGISTRY))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    out = ((OUTPUT / args.model) if args.model in MODELS else
           (ROOT / "outputs/02_eval_runs/fm14_rxrx3_five_seed_20261019" / args.model)) / "rxrx3_core_five_seed"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(args, out)


def run(args: argparse.Namespace, out: Path) -> None:
    metadata = json.loads((INPUT / "metadata.json").read_text())
    protocol = json.loads((ROOT / "Evaluation Rules/protocol_v4.json").read_text())
    split_sha = sha256(INPUT / "split_manifest.jsonl")
    assert metadata["protocol_id"] == PROTOCOL
    assert split_sha == metadata["manifest_sha256"] == protocol["retrieval_splits"]["rxrx3-core"]["manifest_sha256"]
    rows = metadata["rows"]
    assert len(rows) == 1468
    gallery = np.asarray([row["split"] == "gallery" for row in rows])
    assert int(gallery.sum()) == 734 and int((~gallery).sum()) == 734
    assert not ({row["well_id"] for row in rows if row["split"] == "gallery"} &
                {row["well_id"] for row in rows if row["split"] == "query"})

    feature_path = out / "features.npz"
    if feature_path.exists():
        with np.load(feature_path) as saved:
            assert str(saved["split_sha256"]) == split_sha
            features = saved["features"].astype(np.float32)
    else:
        images = np.load(INPUT / "images.npy", mmap_mode="r")
        assert len(images) == len(rows)
        encoder = (NewEncoder(args.model, "cuda:0", args.batch_size)
                   if args.model in MODELS else StandardExternalEncoder(args.model, 224, 256))
        parts = []
        for start in range(0, len(images), args.batch_size):
            batch = [Image.fromarray(np.asarray(image)) for image in images[start:start + args.batch_size]]
            parts.append(encoder.encode_images(batch))
            if start == 0 or (start // args.batch_size) % 10 == 0:
                print(f"[extract] {args.model}: {min(start + args.batch_size, len(images))}/{len(images)}", flush=True)
        features = np.concatenate(parts).astype(np.float32)
        assert len(features) == len(rows) and np.isfinite(features).all()
        np.savez_compressed(feature_path, features=features.astype(np.float16), split_sha256=split_sha)
        if args.model not in MODELS:
            encoder.features.close()
        del encoder

    labels = np.load(INPUT / "labels.npy")
    assert len(labels) == len(features)
    features /= np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-12)
    gallery_x, query_x = features[gallery], features[~gallery]
    gallery_y, query_y = labels[gallery], labels[~gallery]
    scores = query_x @ gallery_x.T
    order = np.argsort(-scores, axis=1)
    ranks = np.asarray([np.flatnonzero(gallery_y[index] == query_y[i])[0] + 1
                        for i, index in enumerate(order)])
    retrieval = {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
    }
    cluster_input = np.concatenate((gallery_x, query_x))
    cluster_labels = np.concatenate((gallery_y, query_y))
    for seed in range(5):
        result = out / f"seed{seed}.json"
        if result.exists():
            existing = json.loads(result.read_text())
            if existing.get("split_sha256") == split_sha and existing.get("seed") == seed:
                print(f"[skip] {args.model} seed{seed}", flush=True)
                continue
        pred = MiniBatchKMeans(n_clusters=734, n_init=5, random_state=seed,
                               batch_size=1024, max_iter=200).fit_predict(cluster_input)
        payload = {
            "model": args.model, "dataset": "rxrx3-core", "protocol_id": PROTOCOL,
            "v4_protocol_id": "bio-eval-union-v4", "split_sha256": split_sha,
            "seed": seed, "n_gallery": 734, "n_query": 734,
            "retrieval": retrieval,
            "clustering": {"nmi": float(normalized_mutual_info_score(cluster_labels, pred)),
                           "method": "MiniBatchKMeans(n_init=5,max_iter=200)"},
            "feature_file": str(feature_path), "status": "VALID_COMPLETE",
        }
        temp = result.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temp, result)
        print(f"[complete] {args.model} seed{seed}", flush=True)


if __name__ == "__main__":
    main()
