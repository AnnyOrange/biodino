"""Pinned, full-sweep selection and frozen evaluation for repaired datasets.

Local 1TB development can precede distributed synchronization. Freezing and all
5TB/20TB evaluation require the complete code/environment synchronization gate.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import traceback

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import torch

from .candidate_datasets import digest, file_digest
from .protocol_campaign import (ROOT, code_identity, feature_layers, validate_model,
                                validate_sync, write_new)


def implementation_identity():
    paths = ["dinov3", "environments/hs6_protocol_v2.txt", "scripts/evaluation_environment.py"]
    rows = subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT),
                                    "ls-tree", "-r", "HEAD", "--", *paths], text=True)
    return digest(rows)


def environment_identity():
    import importlib.util

    spec = importlib.util.spec_from_file_location("hs6_environment", ROOT / "scripts/evaluation_environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fingerprint()


def admit(args, config):
    identity = code_identity()
    published = subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT),
                                         "ls-remote", "origin", "refs/heads/main"], text=True).split()[0]
    if published != identity["git_commit"]:
        raise ValueError("Benchmark commit is not the published authoritative GitHub commit")
    models = json.loads(Path(args.models).read_text())
    checkpoint_sha, config_sha = file_digest(args.checkpoint), file_digest(args.train_config)
    model = models["models"][args.model_id]
    validate_model(models, args.model_id, model["model_family"], args.budget, checkpoint_sha, config_sha)
    environment = environment_identity()
    identity.update(model_id=args.model_id, model_family=model["model_family"], model_budget=args.budget,
                    checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=checkpoint_sha,
                    train_config_sha256=config_sha, model_depth=model["depth"],
                    implementation_sha256=implementation_identity(), environment=environment,
                    dataset=args.dataset, config_sha256=digest(config), seed=config["seed"])
    if args.command in {"freeze", "evaluate"} or not args.local_selection:
        if not args.sync_manifest:
            raise ValueError("Distributed runs/freezing require a four-machine synchronization manifest")
        registry_sha = file_digest(ROOT / "Evaluation Rules/unprotocolized_protocols.json")
        identity["sync_manifest_sha256"] = validate_sync(args.sync_manifest, identity["git_commit"], registry_sha)
        sync = json.loads(Path(args.sync_manifest).read_text())
        for row in sync["machines"]:
            if row.get("environment_sha256") != environment["environment_sha256"]:
                raise ValueError(f"Pinned environment synchronization mismatch: {row['host']}")
            active_sha = row.get("registries", {}).get(str(Path(args.registry).relative_to(ROOT)))
            if active_sha != file_digest(args.registry):
                raise ValueError(f"Active protocol registry synchronization mismatch: {row['host']}")
    elif args.budget != "1TB" or not str(ROOT).startswith("/mnt/huawei_deepcad/"):
        raise ValueError("Pre-sync development is limited to 1TB in a local authoritative clean worktree")
    return identity


def read_manifest(args, config):
    if not args.manifest:
        raise ValueError("An approved, fully validated data manifest is required")
    manifest = json.loads(Path(args.manifest).read_text())
    expected = {"CellFMCount": ("cellfmcount-dapi-count", "regression"),
                "AllenCell_Morphology": ("allen-cell-volume-well-grouped", "regression"),
                "CytoImageNet": ("cytoimagenet-source-grouped", "classification"),
                "OpenCell": ("opencell-major-localization-protein-heldout", "classification"),
                "VGG_Cell_Counting": ("vgg-synthetic-cell-count", "regression"),
                "FILM": ("FILM", "c_elegans_age_classification")}
    if args.dataset in expected and (manifest.get("dataset"), manifest.get("task")) != expected[args.dataset]:
        raise ValueError("Data manifest belongs to a different dataset/task")
    if manifest.get("seed", config["seed"]) != config["seed"]:
        raise ValueError("Manifest seed differs from protocol split seed")
    if manifest.get("manifest_sha256"):
        signature = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        if signature != manifest["manifest_sha256"]:
            raise ValueError("Manifest signature mismatch")
    if config["adapter"] == "hest":
        if not (manifest["status"] == "PASS" and manifest["release_status"] == "PASS"
                and manifest["payload_verified"] and args.tissue in manifest["tasks"]):
            raise ValueError("Full release/payload/target HEST validation is required")
    elif config["adapter"] == "grouped":
        from .grouped_benchmarks import validate_grouped_manifest
        if not manifest["content_audit"]["completed"]:
            raise ValueError("Full decoded-pixel/source-group audit is required")
        validate_grouped_manifest(manifest)
    elif config["adapter"] in {"cellfm", "opencell"}:
        from .candidate_datasets import validate_records
        validate_records(manifest["records"])
        if config["adapter"] == "cellfm" and not manifest.get("source_audit_sha256"):
            raise ValueError("CellFMCount requires a checksum-verified complete official-source audit")
    return manifest


def verify_data(args, config, manifest):
    root = Path(args.benchmark_root)
    if config["adapter"] in {"vgg", "film"}:
        if config["adapter"] == "vgg":
            from .vgg_count import build_vgg_manifest
            rebuilt = build_vgg_manifest(root, config["seed"], n_train=32)
        else:
            from .film import build_film_manifest
            rebuilt = build_film_manifest(root / "ood/ood_classification/datasets/FILM", tuple(config["repetition_seeds"]))
        approved = {k: v for k, v in manifest.items() if k != "loader_smoke"}
        if rebuilt != approved:
            raise ValueError("Approved CV dataset no longer matches source pixels, annotations and folds")
        return None
    if config["adapter"] == "hest":
        release_path = Path(args.manifest).parent / "release_verification.json"
        release = json.loads(release_path.read_text())
        if release["status"] != "PASS" or not release["files"]:
            raise ValueError("HEST release identity verification missing")
        for row in release["files"]:
            path = root / "external_benchmarks_20260901/HEST_benchmark" / row["path"]
            if file_digest(path) != row["sha256"] or row["status"] != "PASS":
                raise ValueError(f"HEST published payload changed: {path}")
        return file_digest(release_path)
    folders = {"cellfm": "ood/ood_regression/datasets/CellFMCount/extracted",
               "opencell": "external_benchmarks_20260901/OpenCell_projections"}
    if config["adapter"] == "grouped":
        folder = "Classification/CHAMMI" if manifest["task"] == "regression" else "Representation/CytoImageNet/extracted"
    else:
        folder = folders[config["adapter"]]
    for row in manifest["records"]:
        if file_digest(root / folder / row["path"]) != row["image_sha256"]:
            raise ValueError(f"Approved image payload changed: {row['sample_id']}")
        if row.get("annotation_sha256"):
            path = root / folder / "ground_truth" / f"{row['sample_id']}.csv"
            if file_digest(path) != row["annotation_sha256"]:
                raise ValueError(f"Approved annotation changed: {row['sample_id']}")
    if config["adapter"] == "cellfm":
        from .cellfmcount import OFFICIAL_ARCHIVE_MD5, build_cellfm_manifest

        audit_path = Path(args.manifest).parent / "cellfmcount_source_audit.json"
        audit = json.loads(audit_path.read_text())
        if digest(audit) != manifest["source_audit_sha256"]:
            raise ValueError("CellFMCount official-source audit artifact changed")
        if not audit["archive_md5_verified"] or audit["official_archive_md5"] != OFFICIAL_ARCHIVE_MD5:
            raise ValueError("CellFMCount official source identity missing")
        regenerated = build_cellfm_manifest(args.benchmark_root, config["seed"], source_audit=audit)
        if regenerated != manifest:
            raise ValueError("CellFMCount source annotations/metadata/targets no longer reproduce approved manifest")
    return None


def dataset(args, config, manifest, split):
    if config["adapter"] == "vgg":
        from .vgg_count import VGGCountDataset
        return VGGCountDataset(args.benchmark_root, manifest, split)
    if config["adapter"] == "film":
        from .film import FILMAgeDataset
        return FILMAgeDataset(Path(args.benchmark_root) / "ood/ood_classification/datasets/FILM", manifest, split)
    if config["adapter"] == "cellfm":
        from .cellfmcount import CellFMCountDataset
        return CellFMCountDataset(args.benchmark_root, manifest, split)
    if config["adapter"] == "opencell":
        from .opencell import OpenCellDataset
        return OpenCellDataset(args.benchmark_root, manifest, split)
    if config["adapter"] == "grouped":
        from .grouped_benchmarks import GroupedBenchmarkDataset
        return GroupedBenchmarkDataset(args.benchmark_root, manifest, split)
    raise ValueError("Unsupported global-feature adapter")


def make_encoder(args, config, identity, feature):
    from .encoder import Dinov3CkptEncoder
    encoder = Dinov3CkptEncoder(Path(args.checkpoint), Path(args.train_config), args.device,
        1, config["preprocessing"]["patch_avgpool"], torch.bfloat16,
        image_size=config["preprocessing"]["image_size"],
        resize_size=config["preprocessing"]["resize_size"],
        channel_policy=config["preprocessing"].get("channel_policy", "auto"))
    if len(encoder.model.backbone.blocks) != identity["model_depth"]:
        raise ValueError("Loaded model depth differs from admitted checkpoint")
    layers = feature_layers(identity["model_depth"], feature)
    encoder.model.n_last_blocks = layers
    return encoder, layers


def extract_global(args, config, manifest, identity, feature, splits):
    from .encoder import extract_features
    encoder, layers = make_encoder(args, config, identity, feature)
    bank, output = {}, Path(args.output) / "features" / feature
    output.mkdir(parents=True)
    for split in splits:
        path = output / f"{split}.npz"
        if path.exists():
            raise ValueError("Fresh selection extraction required; do not reuse an unverified cache")
        data = dataset(args, config, manifest, split)
        bank[split] = extract_features(data, encoder, path, args.batch_size, args.workers,
                                      True, args.model_id, save_paths=True)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return bank, layers


def probe(config, parameters, train, test):
    from .probes import run_classification_probe_split, run_regression_probe_split
    if config["evaluator"] == "ridge":
        return run_regression_probe_split(*train, *test, alpha=parameters["alpha"]).metrics
    if config["evaluator"] == "logistic":
        return run_classification_probe_split(*train, *test, C=parameters["C"], seed=config["seed"]).metrics
    if config["evaluator"] == "knn":
        from dinov3.eval.bio_classification.knn import knn_predict_scores
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        scores = knn_predict_scores(train_features=torch.from_numpy(train[0]),
            train_labels=torch.from_numpy(train[1]), test_features=torch.from_numpy(test[0]),
            num_classes=len(np.unique(train[1])), k=parameters["k"], temperature=parameters["temperature"], chunk_size=128)
        predicted = scores.argmax(dim=1).numpy()
        return {"accuracy": float(accuracy_score(test[1], predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(test[1], predicted))}
    raise ValueError("Unsupported probe")


def candidates(config):
    keys = list(config["search"]["hyperparameters"])
    return [dict(zip(keys, values)) for values in itertools.product(
        *(config["search"]["hyperparameters"][key] for key in keys))]


def validate_search(rows, config, depth):
    expected = {(feature, digest(parameters)) for feature in config["search"]["features"]
                for parameters in candidates(config)}
    actual = [(row["feature"], digest(row["hyperparameters"])) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Sweep candidate grid incomplete, duplicated or altered")
    for row in rows:
        if row["zero_based_layers"] != feature_layers(depth, row["feature"]):
            raise ValueError("Candidate feature/layer definitions differ from registry")


def choose(rows, metric, direction, expected):
    if len(rows) != expected or any(row["status"] != "SUCCESS" or
        not np.isfinite(row["metrics"].get(metric, np.nan)) for row in rows):
        raise ValueError("Cannot select or freeze incomplete/failed/nonfinite sweeps")
    multiplier = 1 if direction == "min" else -1
    return min(rows, key=lambda row: multiplier * row["metrics"][metric])


def hest_bank(args, config, manifest, identity, feature):
    from .hest import HESTPatchDataset, _records
    from .encoder import extract_features
    task = manifest["tasks"][args.tissue]
    encoder, layers = make_encoder(args, config, identity, feature)
    root = Path(args.benchmark_root) / "external_benchmarks_20260901/HEST_benchmark" / args.tissue
    bank, samples, source_records = {}, {}, {}
    folds = sorted(task["folds"], key=int)
    # All official folds reuse the same source samples. Extract each sample once,
    # then reconstruct every fold in exact CSV/sample/barcode order.
    for split in ("train", "test"):
        data = HESTPatchDataset(root, split=split, fold=int(folds[0]))
        source_records.update({row["sample_id"]: row for row in data.records})
        path = Path(args.output) / "features" / feature / f"source_fold{folds[0]}_{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        x, y = extract_features(data, encoder, path, args.batch_size, args.workers,
                                True, args.model_id, save_paths=True)
        ids = np.asarray([item[2] for item in data.samples])
        for sample in set(ids):
            samples[sample] = (x[ids == sample], y[ids == sample])
        data.close()
    for fold in folds:
        for split in ("train", "test"):
            rows, _ = _records(root, split, int(fold))
            ids = [row["sample_id"] for row in rows]
            if not set(ids).issubset(samples):
                raise ValueError("HEST fold zero does not cover every fold's source samples")
            if any(any(row[key] != source_records[row["sample_id"]][key]
                       for key in ("patches_path", "expr_path")) for row in rows):
                raise ValueError("A HEST sample's patch/expression identity changes between folds")
            bank[(fold, split)] = tuple(np.concatenate([samples[sample][axis] for sample in ids]) for axis in (0, 1))
    del encoder
    return bank, layers


def hest_score(config, parameters, bank):
    from .hest import run_hest_probe_split
    reports = []
    for fold in sorted({key[0] for key in bank}, key=int):
        train, test = bank[(fold, "train")], bank[(fold, "test")]
        report = run_hest_probe_split(*train, *test, latent_dim=256,
                                     alpha_multiplier=parameters["alpha_multiplier"], seed=config["seed"])
        reports.append({"fold": fold, "metrics": report["metrics"], "hyperparameters": report["hyperparameters"],
                        "metric_valid": report["metric_valid"], "n_train": report["n_train"], "n_test": report["n_test"]})
    if not all(row["metric_valid"] for row in reports):
        raise ValueError("HEST official undefined-gene Pearson propagation invalidates this candidate")
    return {"gene_wise_pearson": float(np.mean([row["metrics"]["gene_wise_pearson"] for row in reports])),
            "mae": float(np.mean([row["metrics"]["mae"] for row in reports]))}, reports


def cv_bank(args, config, manifest, identity, feature, evaluation=False):
    splits = ("all",) if config["adapter"] == "film" else (("development", "test") if evaluation else ("development",))
    bank, layers = extract_global(args, config, manifest, identity, feature, splits)
    return bank, layers


def cv_score(config, parameters, bank, manifest, evaluation=False):
    if config["adapter"] == "vgg":
        from .probes import run_regression_probe_split
        records = [r for r in manifest["records"] if r["source_split"] == "development"]
        if evaluation:
            records += [r for r in manifest["records"] if r["source_split"] == "test"]
        x = np.concatenate([value[0] for value in bank.values()])
        y = np.concatenate([value[1] for value in bank.values()])
        index = {r["sample_id"]: i for i, r in enumerate(records)}
        reports = []
        for fold in manifest["folds"]:
            train = [index[s] for s in fold["train"]]
            test = [index[s] for s in fold["test" if evaluation else "val"]]
            metrics = run_regression_probe_split(x[train], y[train], x[test], y[test], alpha=parameters["alpha"]).metrics
            reports.append({"fold": fold["fold"], "metrics": metrics})
    else:
        from .film import run_film_fold_probe
        reports = []
        for repetition, entry in enumerate(manifest["repetitions"]):
            for fold in range(len(entry["folds"])):
                inner = [run_film_fold_probe(bank["all"][0], manifest, repetition, fold,
                          C=parameters["C"], inner_fold=i) for i in range(3)]
                reports.append({"repetition": repetition, "fold": fold,
                    "metrics": {k: float(np.mean([r[k] for r in inner])) for k in
                                ("accuracy", "balanced_accuracy", "macro_f1")}, "inner_folds": inner})
    return {key: float(np.mean([r["metrics"][key] for r in reports]))
            for key in reports[0]["metrics"]}, reports


def select_protocol(rows, config, depth):
    validate_search(rows, config, depth)
    winner = choose(rows, config["primary_metric"], config["direction"],
                    len(config["search"]["features"]) * len(candidates(config)))
    if config["adapter"] != "film":
        return winner
    protocols = []
    expected = {(r, f) for r in range(3) for f in range(3)}
    for row in rows:
        if {(f["repetition"], f["fold"]) for f in row["folds"]} != expected or len(row["folds"]) != 9:
            raise ValueError("FILM requires every unique outer fold's inner-validation report")
    for repetition, fold in sorted(expected):
        choices = []
        for row in rows:
            report = next(f for f in row["folds"] if (f["repetition"], f["fold"]) == (repetition, fold))
            value = report["metrics"][config["primary_metric"]]
            if not np.isfinite(value):
                raise ValueError("Nonfinite FILM inner-validation metric")
            choices.append((value, row))
        _, selected = max(choices, key=lambda item: item[0])
        protocols.append({"repetition": repetition, "fold": fold,
                          **{k: selected[k] for k in ("feature", "hyperparameters", "zero_based_layers")}})
    return {"feature": "fold-specific", "fold_protocols": protocols,
            "selection": "Independent inner grouped CV within each outer training partition"}


def run(args):
    registry = json.loads(Path(args.registry).read_text())
    config = registry["datasets"][args.dataset]
    identity = admit(args, config)
    manifest = read_manifest(args, config)
    release_sha = verify_data(args, config, manifest)
    identity.update(manifest_sha256=file_digest(args.manifest), protocol_config_id=config["id"],
                    release_verification_sha256=release_sha,
                    preprocessing=config["preprocessing"], aggregation=config["aggregation"], tissue=args.tissue,
                    extraction_batch_size=args.batch_size,
                    output_path=str(Path(args.output).resolve()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_new(output / "invocation.json", {**identity, "command": args.command})
    if args.command == "freeze":
        sweep = json.loads(Path(args.sweep).read_text())
        if args.budget != "1TB" or sweep.get("model_budget") != "1TB" or sweep.get("status") != "SELECTION_COMPLETE_NOT_YET_FROZEN":
            raise ValueError("Only a completed 1TB sweep can be frozen")
        for key in ("implementation_sha256", "config_sha256", "manifest_sha256", "release_verification_sha256", "model_id", "dataset", "tissue",
                    "checkpoint_sha256", "train_config_sha256", "model_family", "model_depth", "extraction_batch_size"):
            if sweep[key] != identity[key]:
                raise ValueError(f"Selected sweep differs from frozen admission: {key}")
        if sweep["environment"]["environment_sha256"] != identity["environment"]["environment_sha256"]:
            raise ValueError("Selection environment changed")
        winner = select_protocol(sweep["rows"], config, identity["model_depth"])
        frozen = {**identity, "status": "FROZEN", "selection_budget": "1TB", "selection_git_commit": sweep["git_commit"],
                  "selection_sweep_sha256": file_digest(args.sweep), "selected": winner,
                  "config": config, "selection_model": {k: identity[k] for k in
                    ("model_id", "checkpoint_sha256", "train_config_sha256", "model_family", "model_depth")}}
        frozen["protocol_sha256"] = digest(frozen)
        write_new(output / "frozen.json", frozen)
        return
    if args.command == "sweep":
        if args.budget != "1TB":
            raise ValueError("Only 1TB can select protocols")
        rows = []
        for feature in config["search"]["features"]:
            hest = config["adapter"] == "hest"
            cv = config["adapter"] in {"vgg", "film"}
            layers = feature_layers(identity["model_depth"], feature)
            extraction_error = None
            try:
                if hest:
                    bank, layers = hest_bank(args, config, manifest, identity, feature)
                elif cv:
                    bank, layers = cv_bank(args, config, manifest, identity, feature)
                else:
                    bank, layers = extract_global(args, config, manifest, identity, feature, ("train", "val"))
            except Exception as error:
                extraction_error = str(error)
            for parameters in candidates(config):
                try:
                    if extraction_error:
                        raise RuntimeError(extraction_error)
                    if hest:
                        metrics, folds = hest_score(config, parameters, bank)
                    elif cv:
                        metrics, folds = cv_score(config, parameters, bank, manifest)
                    else:
                        metrics, folds = probe(config, parameters, bank["train"], bank["val"]), None
                    row = {"status": "SUCCESS", "metrics": {k: float(v) if np.isfinite(v) else None for k, v in metrics.items()}, "folds": folds}
                except Exception as error:
                    row = {"status": "FAILURE", "error": str(error), "metrics": {}, "folds": None}
                row.update({"feature": feature, "zero_based_layers": layers,
                       "hyperparameters": parameters,
                       "config_id": config["id"], "seed": config["seed"]}
                )
                write_new(output / "candidates" / f"candidate_{len(rows):03d}.json", {**identity, **row})
                rows.append(row)
                print(json.dumps(row), flush=True)
        try:
            winner = select_protocol(rows, config, identity["model_depth"])
        except ValueError:
            winner = None
        write_new(output / "sweep.json", {**identity, "status": "SELECTION_COMPLETE_NOT_YET_FROZEN" if winner else "FAILURE", "rows": rows,
            "selected": winner, "selection_split": "official outer test folds, development only" if config["adapter"] == "hest" else ("per-outer-fold inner grouped CV" if config["adapter"] == "film" else "validation"),
            "test_used_for_development": config["adapter"] == "hest"})
        if winner is None:
            raise ValueError("Full sweep saved but failed/nonfinite candidates prevent freezing")
        return
    frozen = json.loads(Path(args.frozen).read_text())
    if frozen["protocol_sha256"] != digest({k: v for k, v in frozen.items() if k != "protocol_sha256"}):
        raise ValueError("Frozen protocol checksum mismatch")
    for key in ("git_commit", "implementation_sha256", "config_sha256", "manifest_sha256", "release_verification_sha256", "model_family", "model_depth", "dataset", "tissue", "extraction_batch_size"):
        if frozen[key] != identity[key]:
            raise ValueError(f"Frozen protocol mismatch: {key}")
    if frozen["status"] != "FROZEN" or frozen["environment"]["environment_sha256"] != identity["environment"]["environment_sha256"]:
        raise ValueError("Frozen protocol/environment admission mismatch")
    if args.budget == "1TB":
        for key in ("checkpoint_sha256", "train_config_sha256"):
            if frozen["selection_model"][key] != identity[key]:
                raise ValueError("1TB evaluation must use the selection checkpoint/config")
    else:
        previous = json.loads(Path(args.previous).read_text()) if args.previous else {}
        expected_budget = "1TB" if args.budget == "5TB" else "5TB"
        if not (previous.get("status") == "SUCCESS" and previous.get("model_budget") == expected_budget
                and previous.get("git_commit") == identity["git_commit"]
                and previous.get("protocol_sha256") == frozen["protocol_sha256"]):
            raise ValueError("Budget progression requires matching successful evaluation at the previous budget")
    selected = frozen["selected"]
    if config["adapter"] == "film":
        from .film import run_film_fold_probe
        banks = {}
        folds = []
        for entry in selected["fold_protocols"]:
            feature = entry["feature"]
            if feature not in banks:
                banks[feature], layers = cv_bank(args, config, manifest, identity, feature, evaluation=True)
                if layers != entry["zero_based_layers"]:
                    raise ValueError("Frozen FILM taps changed")
            report = run_film_fold_probe(banks[feature]["all"][0], manifest, entry["repetition"],
                                        entry["fold"], C=entry["hyperparameters"]["C"])
            folds.append({**entry, "metrics": {k: report[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}})
        metrics = {k: float(np.mean([f["metrics"][k] for f in folds])) for k in folds[0]["metrics"]}
        seed_means = [np.mean([f["metrics"]["balanced_accuracy"] for f in folds if f["repetition"] == r]) for r in range(3)]
        metrics["balanced_accuracy_seed_std"] = float(np.std(seed_means, ddof=0))
        write_new(output / "result.json", {**identity, "status": "SUCCESS", "feature": "fold-specific",
            "fold_protocols": selected["fold_protocols"], "metric": config["primary_metric"],
            "metrics": metrics, "folds": folds, "protocol_sha256": frozen["protocol_sha256"], "retuned": False})
        return
    if config["adapter"] == "hest":
        bank, layers = hest_bank(args, config, manifest, identity, selected["feature"])
        metrics, folds = hest_score(config, selected["hyperparameters"], bank)
    elif config["adapter"] == "vgg":
        bank, layers = cv_bank(args, config, manifest, identity, selected["feature"], evaluation=True)
        metrics, folds = cv_score(config, selected["hyperparameters"], bank, manifest, evaluation=True)
        metrics["mae_draw_std"] = float(np.std([f["metrics"]["mae"] for f in folds], ddof=0))
    else:
        bank, layers = extract_global(args, config, manifest, identity, selected["feature"], ("train", "test"))
        metrics, folds = probe(config, selected["hyperparameters"], bank["train"], bank["test"]), None
    if layers != selected["zero_based_layers"]:
        raise ValueError("Frozen feature taps changed")
    write_new(output / "result.json", {**identity, "status": "SUCCESS", "feature": selected["feature"],
        "hyperparameters": selected["hyperparameters"], "metric": config["primary_metric"],
        "metrics": metrics, "folds": folds, "protocol_sha256": frozen["protocol_sha256"], "retuned": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sweep", "freeze", "evaluate"])
    for field in ("dataset", "model-id", "checkpoint", "train-config", "benchmark-root", "manifest", "output"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--registry", default=str(ROOT / "Evaluation Rules/repaired_protocol_candidates.json"))
    parser.add_argument("--models", default=str(ROOT / "Evaluation Rules/unprotocolized_models.json"))
    parser.add_argument("--budget", choices=["1TB", "5TB", "20TB"], default="1TB")
    parser.add_argument("--tissue")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sync-manifest")
    parser.add_argument("--local-selection", action="store_true")
    parser.add_argument("--frozen")
    parser.add_argument("--sweep")
    parser.add_argument("--previous")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAILURE", "dataset": args.dataset, "model_id": args.model_id,
                   "model_budget": args.budget, "command": vars(args), "error": str(error), "traceback": traceback.format_exc()}
        try:
            failure.update(code_identity(require_clean=False))
        except Exception:
            pass
        write_new(output / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
