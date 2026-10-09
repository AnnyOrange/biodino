#!/usr/bin/env python3
"""Paired, validation-only 20TB representation diagnosis.

Extract frozen features on GPU, then compare checkpoints using the same image
IDs and fixed training/validation partitions. This is exploratory and does not
write official v4 scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sample_id(dataset, index, path):
    # NCT parquet has null image.path; row order in the fixed shard is its ID.
    return f"{dataset}:{index}" if dataset == "nct-crc-he" else str(path)


def allen_metadata(paths, meta_path, field):
    lookup = {}
    with meta_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("train_test_split") == "Train" and row.get("file_path"):
                lookup[str(meta_path.parent.parent / row["file_path"])] = str(row.get(field, ""))
    groups = [lookup.get(str(path), "") for path in paths]
    if any(not group for group in groups):
        raise ValueError(f"Allen {field} missing for an extracted image")
    return np.asarray(groups)


def encode_raw(encoder, images):
    import torch
    if torch.is_tensor(images[0]):
        tensors = [encoder._resize_center_crop_tensor(image) for image in images]
        channels = max(int(tensor.shape[0]) for tensor in tensors)
        batch = torch.zeros(len(tensors), channels, encoder.image_size, encoder.image_size)
        valid = torch.zeros(len(tensors), channels, dtype=torch.bool)
        for index, tensor in enumerate(tensors):
            batch[index, :len(tensor)] = tensor
            valid[index, :len(tensor)] = True
        batch = encoder._collapse_to_three_channels_once(batch, valid, "auto")
        batch = encoder._normalize_tensor_batch(batch).to(encoder.device)
    else:
        batch = torch.stack([encoder.transform(image) for image in images]).to(encoder.device)
    with torch.inference_mode():
        return encoder.model(batch).float().cpu().numpy()


def extract(args):
    import torch
    from torch.utils.data import DataLoader
    from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder, pil_collate
    from dinov3.eval.bio_frozen_eval.registry import build_dataset
    from dinov3.eval.bio_frozen_eval.retrieval_clustering import build_retrieval_dataset

    encoder = Dinov3CkptEncoder(
        checkpoint=args.checkpoint, train_config=args.train_config, device="cuda",
        n_last_blocks=1, use_avgpool=True, autocast_dtype=torch.bfloat16,
        image_size=224, resize_size=256, channel_policy="auto",
    )
    for dataset_name in args.datasets:
        output = args.output_dir / args.model / f"{dataset_name}.npz"
        if output.exists() and not args.overwrite:
            print(f"[skip] {output}", flush=True)
            continue
        if dataset_name == "nct-crc-he-1k":
            ds, _ = build_retrieval_dataset(dataset_name, benchmark_root=args.benchmark_root)
            task = "classification"
        else:
            ds, task = build_dataset(
                dataset_name, split="train", max_samples=None,
                max_per_class=args.max_per_class, benchmark_root=args.benchmark_root,
            )
        if task != "classification":
            raise ValueError(f"Expected classification dataset, got {task}")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=pil_collate)
        features, raw_cls, labels, paths = [], [], [], []
        for batch_index, (images, batch_labels, batch_paths) in enumerate(loader):
            raw = encode_raw(encoder, images)
            dim = raw.shape[1] // 2
            raw_cls.append(raw[:, :dim].astype(np.float16))
            features.append((raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)).astype(np.float16))
            labels.extend(int(label) for label in batch_labels)
            paths.extend(str(path) for path in batch_paths)
            if batch_index % 20 == 0:
                print(f"[extract] {args.model} {dataset_name}: {len(paths)}/{len(ds)}", flush=True)
        full = np.concatenate(features).astype(np.float16)
        if len(full) != len(ds) or full.shape[1] % 2:
            raise ValueError("Feature count or CLS+patchmean feature width is invalid")
        ids = np.asarray([sample_id(dataset_name, i, path) for i, path in enumerate(paths)])
        groups = (allen_metadata(paths, args.benchmark_root / "Classification/CHAMMI/Allen/enriched_meta.csv", "FOVId")
                  if dataset_name == "chammi-allen-task2" else np.asarray(["unknown"] * len(paths)))
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, full=full, cls=full[:, :full.shape[1] // 2],
                 raw_cls=np.concatenate(raw_cls),
                 labels=np.asarray(labels, dtype=np.int64), ids=ids, groups=groups,
                 checkpoint=str(args.checkpoint), train_config=str(args.train_config))
        print(f"[saved] {output} n={len(ds)} dim={full.shape[1]}", flush=True)


def split_indices(labels, groups, dataset, seed):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    ids = np.arange(len(labels))
    if dataset == "chammi-allen-task2":
        fit, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
                        .split(ids, labels, groups))
    else:
        fit, val = train_test_split(ids, test_size=.25, random_state=seed,
                                    stratify=labels)
    if set(groups[fit]) & set(groups[val]) and dataset == "chammi-allen-task2":
        raise AssertionError("Allen FOV leakage")
    return np.asarray(fit), np.asarray(val)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def kmeans_rows(model, dataset, x, y, val, seeds):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    a, b = normalize(x[val]), y[val]
    rows = []
    for seed in range(seeds):
        predicted = MiniBatchKMeans(n_clusters=len(np.unique(b)), random_state=seed,
                                    batch_size=2048, n_init="auto").fit_predict(a)
        rows.append(dict(model=model, dataset=dataset, seed=seed,
                         nmi=float(normalized_mutual_info_score(b, predicted)),
                         ari=float(adjusted_rand_score(b, predicted))))
    return rows


def probe_rows(model, dataset, representation, x, y, fit, val, seed, fractions):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    rows = []
    for fraction in fractions:
        for repeat in range(3):
            if fraction == 1:
                subset = fit
            else:
                subset, _ = train_test_split(fit, train_size=fraction,
                                              stratify=y[fit], random_state=seed + repeat)
            scaler = StandardScaler().fit(x[subset])
            train_x = scaler.transform(x[subset]).astype(np.float32)
            val_x = scaler.transform(x[val]).astype(np.float32)
            head = RidgeClassifier(alpha=10.0, solver="lsqr").fit(train_x, y[subset])
            prediction = head.predict(val_x)
            rows.append(dict(model=model, dataset=dataset, representation=representation,
                             fraction=fraction, repeat=repeat, n_train=len(subset),
                             balanced_accuracy=float(balanced_accuracy_score(y[val], prediction)),
                             macro_f1=float(f1_score(y[val], prediction, average="macro"))))
    return rows


def recovery_row(dataset, current, anchor, fit, val, device, domains=None):
    import torch
    x = torch.as_tensor(current, dtype=torch.float32, device=device)
    y = torch.as_tensor(anchor, dtype=torch.float32, device=device)
    x_fit, y_fit, x_val, y_val = x[fit], y[fit], x[val], y[val]
    # Fit a ridge residual map using only calibration observations.
    design = torch.cat((x_fit, torch.ones_like(x_fit[:, :1])), dim=1)
    normal = design.T @ design / len(fit)
    target = design.T @ (y_fit - x_fit) / len(fit)
    delta = torch.linalg.solve(normal + .1 * torch.eye(normal.shape[0], device=device), target)
    predicted = x_val + torch.cat((x_val, torch.ones_like(x_val[:, :1])), dim=1) @ delta
    variance = y_fit.var(dim=0, unbiased=False).clamp_min(.05)
    residual = (predicted - y_val) / variance.sqrt()
    second = residual.T @ residual / len(val)
    # Largest eigenvalue bounds a bounded linear head's squared error.
    worst = torch.linalg.eigvalsh(second)[-1]
    singular = torch.linalg.svdvals(torch.eye(x.shape[1], device=device) + delta[:-1])
    identity_residual = (x_val - y_val) / variance.sqrt()
    per_domain = []
    if domains is not None:
        for domain in sorted(set(domains[val])):
            mask = torch.as_tensor(domains[val] == domain, device=device)
            chunk = residual[mask]
            per_domain.append(dict(dataset=dataset, domain_type="InstrumentId", domain=str(domain),
                                   n_val=int(mask.sum()), normalized_mse=float(chunk.square().mean()),
                                   worst_direction=(float(torch.linalg.eigvalsh(chunk.T @ chunk / len(chunk))[-1])
                                                    if len(chunk) >= 30 else float("nan"))))
    return dict(dataset=dataset, n_fit=len(fit), n_val=len(val), dimension=x.shape[1],
                normalized_mse=float(residual.square().mean()),
                identity_normalized_mse=float(identity_residual.square().mean()),
                worst_direction=float(worst), decoder_norm=float(singular[0]),
                decoder_min_singular=float(singular[-1]),
                decoder_condition=float(singular[0] / singular[-1].clamp_min(1e-8)),
                per_domain=per_domain)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    banks = {}
    for model in (args.anchor, args.current):
        for dataset in args.datasets:
            path = args.input_dir / model / f"{dataset}.npz"
            with np.load(path, allow_pickle=False) as bank:
                banks[model, dataset] = {key: bank[key] for key in ("full", "cls", "raw_cls", "labels", "ids", "groups")}
    kmeans, probes, recovery, source_recovery, splits = [], [], [], [], []
    for dataset in args.datasets:
        old, new = banks[args.anchor, dataset], banks[args.current, dataset]
        for key in ("labels", "ids", "groups"):
            if not np.array_equal(old[key], new[key]):
                raise ValueError(f"Unpaired {key} for {dataset}")
        y = old["labels"]
        fit, val = split_indices(y, old["groups"], dataset, args.seed)
        splits.append(dict(dataset=dataset, n_fit=len(fit), n_val=len(val),
                           fit_ids_sha256=hashlib.sha256("\n".join(old["ids"][fit]).encode()).hexdigest(),
                           val_ids_sha256=hashlib.sha256("\n".join(old["ids"][val]).encode()).hexdigest()))
        for model in (args.anchor, args.current):
            bank = banks[model, dataset]
            for key in ("full", "cls"):
                features = normalize(bank[key])
                probes.extend(probe_rows(model, dataset, key, features, y, fit, val,
                                         args.seed, args.fractions))
                if key == "full":
                    kmeans.extend(kmeans_rows(model, dataset, features, y, val, args.kmeans_seeds))
        domains = (allen_metadata(old["ids"], args.benchmark_root / "Classification/CHAMMI/Allen/enriched_meta.csv", "InstrumentId")
                   if dataset == "chammi-allen-task2" else None)
        row = recovery_row(dataset, new["raw_cls"], old["raw_cls"], fit, val, args.device, domains)
        source_recovery.extend(row.pop("per_domain"))
        recovery.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "kmeans_seeds.csv", kmeans)
    write_csv(args.output_dir / "probe_learning_curves.csv", probes)
    write_csv(args.output_dir / "recovery_domains.csv", recovery)
    write_csv(args.output_dir / "recovery_sources.csv", source_recovery)
    manifest = dict(scope="exploratory train-pool diagnosis, not official v4",
                    anchor=args.anchor, current=args.current, seed=args.seed,
                    splits=splits, kmeans_seeds=args.kmeans_seeds,
                    probe_alpha=10.0, probe_fractions=args.fractions,
                    recovery_ridge=.1, source_groups="dataset proxy; Allen validation is FOV-grouped; NCT source IDs unavailable",
                    recovery=recovery)
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def cluster(args):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rows = []
    reference = None
    for model in (args.anchor, args.current):
        with np.load(args.input_dir / model / f"{args.dataset}.npz", allow_pickle=False) as bank:
            if args.representation == "patchmean":
                values = bank["full"][:, bank["cls"].shape[1]:]
            else:
                values = bank[args.representation]
            x, y, ids = normalize(values), bank["labels"], bank["ids"]
        if reference is None:
            reference = (y, ids)
        elif not np.array_equal(y, reference[0]) or not np.array_equal(ids, reference[1]):
            raise ValueError("Unpaired benchmark rows")
        for seed in range(args.kmeans_seeds):
            prediction = MiniBatchKMeans(n_clusters=len(np.unique(y)), random_state=seed,
                                        batch_size=2048, n_init="auto").fit_predict(x)
            rows.append(dict(model=model, dataset=args.dataset, seed=seed,
                             nmi=float(normalized_mutual_info_score(y, prediction)),
                             ari=float(adjusted_rand_score(y, prediction))))
    suffix = "" if args.representation == "full" else f"_{args.representation}"
    write_csv(args.output_dir / f"benchmark_kmeans_seeds{suffix}.csv", rows)
    write_json(args.output_dir / f"benchmark_kmeans_manifest{suffix}.json",
               dict(scope="descriptive benchmark test sensitivity only; no fitting of probes/decoders",
                    anchor=args.anchor, current=args.current, dataset=args.dataset,
                    representation=args.representation,
                    rows=len(reference[0]), seeds=args.kmeans_seeds))
    print(f"[saved] {args.output_dir / f'benchmark_kmeans_seeds{suffix}.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--train-config", type=Path, required=True)
    e.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    e.add_argument("--output-dir", type=Path, required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--datasets", nargs="+", default=["nct-crc-he", "chammi-allen-task2"])
    e.add_argument("--max-per-class", type=int, default=300)
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--workers", type=int, default=2)
    e.add_argument("--overwrite", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--input-dir", type=Path, required=True)
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--benchmark-root", type=Path, default=Path("/mnt/huawei_deepcad/benchmark"))
    a.add_argument("--anchor", default="20tb_ck26351")
    a.add_argument("--current", default="20tb_ck47823")
    a.add_argument("--datasets", nargs="+", default=["nct-crc-he", "chammi-allen-task2"])
    a.add_argument("--seed", type=int, default=20261009)
    a.add_argument("--kmeans-seeds", type=int, default=20)
    a.add_argument("--fractions", type=float, nargs="+", default=[.05, .1, .25, .5, 1.0])
    a.add_argument("--device", default="cuda")
    c = sub.add_parser("cluster")
    c.add_argument("--input-dir", type=Path, required=True)
    c.add_argument("--output-dir", type=Path, required=True)
    c.add_argument("--anchor", default="20tb_ck26351")
    c.add_argument("--current", default="20tb_ck47823")
    c.add_argument("--dataset", default="nct-crc-he-1k")
    c.add_argument("--representation", choices=["full", "cls", "patchmean"], default="full")
    c.add_argument("--kmeans-seeds", type=int, default=50)
    args = parser.parse_args()
    if args.mode == "extract":
        extract(args)
    elif args.mode == "analyze":
        analyze(args)
    else:
        cluster(args)


if __name__ == "__main__":
    main()
