"""Native frozen-backbone detection selection, freezing and budget progression."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
import random
import traceback

import numpy as np
import torch

from dinov3.eval.bio_frozen_eval.candidate_datasets import digest, file_digest
from dinov3.eval.bio_frozen_eval.expansion_campaign import admit
from dinov3.eval.bio_frozen_eval.protocol_campaign import ROOT, feature_layers, validate_model, write_new
from .native import NativeDetectionDataset, build_detector, coco_bbox_metrics, validate_manifest


CANONICAL_ROOT = Path("/mnt/huawei_deepcad/benchmark")


def configuration(registry, dataset):
    if dataset not in {"BCCD", "BBBC041"} or dataset not in registry["datasets"]:
        raise ValueError("Unsupported native detection dataset")
    config = {key: copy.deepcopy(value) for key, value in registry.items() if key != "datasets"}
    config["dataset"] = dataset
    config["dataset_spec"] = registry["datasets"][dataset]
    return config


def grid(config):
    search = config["search"]
    return [dict(zip(("feature", "learning_rate", "epochs", "weight_decay", "momentum"), values))
            for values in itertools.product(search["features"], search["learning_rate"],
                                            search["epochs"], search["weight_decay"], search["momentum"])]


def select(rows, config):
    expected = {digest(recipe) for recipe in grid(config)}
    actual = [digest(row["recipe"]) for row in rows]
    metric = config["primary_metric"]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate native detection sweep grid")
    if any(row["status"] != "SUCCESS" or not np.isfinite(row["metrics"].get(metric, np.nan))
           or len(row["epochs"]) != row["recipe"]["epochs"] for row in rows):
        raise ValueError("Failed/nonfinite/incomplete detection sweep cannot freeze")
    return max(rows, key=lambda row: row["metrics"][metric])


def remap_path(path, benchmark_root):
    path = Path(path)
    if path.is_absolute():
        if not path.is_relative_to(CANONICAL_ROOT):
            raise ValueError("Native manifest absolute path is outside canonical benchmark root")
        path = path.relative_to(CANONICAL_ROOT)
    resolved = (Path(benchmark_root) / path).resolve()
    if not resolved.is_relative_to(Path(benchmark_root).resolve()):
        raise ValueError("Native manifest path escapes benchmark root")
    return resolved


def load_data(args):
    approved = json.loads(Path(args.manifest).read_text())
    if approved.get("dataset") != args.dataset:
        raise ValueError("Native manifest belongs to a different dataset")
    if approved.get("seed", 0) != 0:
        raise ValueError("Native split seed must be zero")
    if approved.get("manifest_sha256") and approved["manifest_sha256"] != digest(
            {k: v for k, v in approved.items() if k != "manifest_sha256"}):
        raise ValueError("Native manifest signature mismatch")
    local = copy.deepcopy(approved)
    # Rebuild from the released annotations/splits, never accept self-signed
    # manifests as sufficient evidence that boxes/classes were joined correctly.
    from .native import build_bccd_manifest, build_bbbc041_manifest
    for row in local["records"]:
        row["path"] = str(remap_path(row["path"], args.benchmark_root))
        if row.get("image_sha256") and file_digest(row["path"]) != row["image_sha256"]:
            raise ValueError("Approved native image content changed")
    first = Path(local["records"][0]["path"])
    if args.dataset == "BCCD":
        source_root = first.parent.parent
        rebuilt = build_bccd_manifest(source_root)
    else:
        source_root = first.parent
        while not (source_root / "training.json").is_file():
            if source_root == Path(args.benchmark_root).resolve() or source_root == source_root.parent:
                raise ValueError("BBBC041 source annotation root unavailable")
            source_root = source_root.parent
        rebuilt = build_bbbc041_manifest(source_root, seed=0)
    compare = ("sample_id", "split", "group", "path", "width", "height", "boxes", "labels", "decoded_pixel_sha256", "annotation_sha256")
    if local["classes"] != rebuilt["classes"] or local["source_hashes"] != rebuilt["source_hashes"]:
        raise ValueError("Native released class map or source annotations/splits changed")
    old = {row["sample_id"]: row for row in local["records"]}
    new = {row["sample_id"]: row for row in rebuilt["records"]}
    if set(old) != set(new) or any(old[s].get(k) != new[s].get(k) for s in old for k in compare):
        raise ValueError("Native approved boxes, membership, pixels or paths differ from released source")
    if args.dataset == "BCCD" and local.get("quarantined_annotations") != rebuilt.get("quarantined_annotations"):
        raise ValueError("Native BCCD source quarantine changed")
    validate_manifest(local, decode=True)
    return local


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detector(args, config, identity, feature, smoke=False):
    from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder
    encoder = Dinov3CkptEncoder(Path(args.checkpoint), Path(args.train_config), args.device,
                                1, False, torch.bfloat16)
    if len(encoder.model.backbone.blocks) != identity["model_depth"]:
        raise ValueError("Admitted detection backbone depth differs")
    layers = feature_layers(identity["model_depth"], feature)
    prep = config["preprocessing"]
    model = build_detector(encoder.model.backbone, layers, len(identity["classes"]) + 1,
                           min_size=64 if smoke else prep["min_size"],
                           max_size=96 if smoke else prep["max_size"]).to(args.device)
    return model, layers


def collate(batch):
    return tuple(zip(*batch))


def evaluate(model, data, args):
    model.eval()
    predictions = []
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.workers, collate_fn=collate)
    with torch.inference_mode():
        for images, _ in loader:
            predictions.extend({key: value.cpu() for key, value in pred.items()}
                               for pred in model([image.to(args.device) for image in images]))
    return coco_bbox_metrics(data, predictions, data.classes)


def train_recipe(args, config, manifest, identity, recipe, directory, score_split):
    directory.mkdir(parents=True, exist_ok=False)
    seed_all(config["seed"])
    model, layers = detector(args, config, identity, recipe["feature"])
    train = NativeDetectionDataset(manifest, "train")
    scoring = NativeDetectionDataset(manifest, score_split)
    scoring.classes = manifest["classes"]
    generator = torch.Generator().manual_seed(config["seed"])
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True,
                 num_workers=args.workers, collate_fn=collate, generator=generator)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
        lr=recipe["learning_rate"], momentum=recipe["momentum"], weight_decay=recipe["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    history = []
    for epoch in range(recipe["epochs"]):
        model.train()
        steps = []
        learning_rate = optimizer.param_groups[0]["lr"]
        for images, targets in loader:
            targets = [{key: value.to(args.device) for key, value in target.items()} for target in targets]
            losses = model([image.to(args.device) for image in images], targets)
            total = sum(losses.values())
            if not torch.isfinite(total):
                raise ValueError("Nonfinite native detector loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            steps.append({key: float(value.detach().cpu()) for key, value in losses.items()})
        scheduler.step()
        metrics = evaluate(model, scoring, args) if score_split == "val" or epoch + 1 == recipe["epochs"] else None
        checkpoint = directory / f"epoch_{epoch + 1:03d}.pth"
        # Frozen DINO weights remain separately admitted; save every trained
        # neck/RPN/RoI state without duplicating the multi-GB backbone per epoch.
        head_state = {key: value.cpu() for key, value in model.state_dict().items()
                      if not key.startswith("backbone.body.")}
        torch.save({"head_state": head_state, "epoch": epoch + 1, "recipe": recipe,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "checkpoint_sha256": identity["checkpoint_sha256"], "seed": config["seed"]}, checkpoint)
        row = {"epoch": epoch + 1, "learning_rate": learning_rate, "step_losses": steps,
               "metrics": metrics, "head_checkpoint": str(checkpoint.resolve()),
               "head_checkpoint_sha256": file_digest(checkpoint)}
        history.append(row)
        write_new(directory / f"epoch_{epoch + 1:03d}.json", row)
        print(json.dumps({"epoch": epoch + 1, "metrics": metrics, "recipe": recipe}), flush=True)
    return {"status": "SUCCESS", "recipe": recipe, "zero_based_layers": layers,
            "epochs": history, "metrics": history[-1]["metrics"], "score_split": score_split}


def validate_freeze(sweep, identity, config):
    if identity["model_budget"] != "1TB" or sweep.get("model_budget") != "1TB" or sweep.get("status") != "SUCCESS":
        raise ValueError("Only successful actual 1TB sweeps may freeze")
    for key in ("git_commit", "implementation_sha256", "config_sha256", "registry_sha256", "manifest_sha256",
                "model_id", "checkpoint_sha256", "train_config_sha256", "model_family", "model_depth", "seed", "dataset",
                "batch_size", "classes"):
        if sweep.get(key) != identity.get(key):
            raise ValueError(f"Native selection identity differs: {key}")
    if sweep["environment"]["environment_sha256"] != identity["environment"]["environment_sha256"]:
        raise ValueError("Native selection environment differs")
    winner = select(sweep["rows"], config)
    for row in sweep["rows"]:
        if row["zero_based_layers"] != feature_layers(identity["model_depth"], row["recipe"]["feature"]):
            raise ValueError("Native feature taps changed")
        for epoch in row["epochs"]:
            if file_digest(epoch["head_checkpoint"]) != epoch["head_checkpoint_sha256"]:
                raise ValueError("Native trained head artifact changed")
    return winner


def validate_frozen(frozen, identity, previous=None):
    if frozen.get("status") != "FROZEN" or frozen.get("selection_budget") != "1TB" or frozen.get("protocol_sha256") != digest(
            {k: v for k, v in frozen.items() if k != "protocol_sha256"}):
        raise ValueError("Invalid frozen native detection protocol")
    for key in ("git_commit", "implementation_sha256", "config_sha256", "registry_sha256", "manifest_sha256", "seed", "dataset",
                "model_family", "model_depth", "batch_size", "classes"):
        if frozen.get(key) != identity.get(key):
            raise ValueError(f"Frozen native identity mismatch: {key}")
    if frozen["environment"]["environment_sha256"] != identity["environment"]["environment_sha256"]:
        raise ValueError("Frozen native environment differs")
    if identity["model_budget"] == "1TB":
        for key in ("model_id", "checkpoint_sha256", "train_config_sha256"):
            if frozen[key] != identity[key]:
                raise ValueError("1TB native evaluation selector model differs")
    else:
        expected = "1TB" if identity["model_budget"] == "5TB" else "5TB"
        if not previous or any(previous.get(key) != value for key, value in {
            "status": "SUCCESS", "model_budget": expected, "protocol_sha256": frozen["protocol_sha256"],
            "git_commit": identity["git_commit"], "dataset": identity["dataset"], "model_family": identity["model_family"],
            "manifest_sha256": identity["manifest_sha256"], "seed": identity["seed"]}.items()):
            raise ValueError("Native budget progression requires matching previous frozen evaluation")


def smoke(args, config, manifest):
    if args.device != "cpu" or args.budget != "1TB":
        raise ValueError("Precommit native smoke is CPU/1TB only")
    models = json.loads(Path(args.models).read_text())
    selected = models["models"][args.model_id]
    checkpoint_sha, config_sha = file_digest(args.checkpoint), file_digest(args.train_config)
    validate_model(models, args.model_id, selected["model_family"], "1TB", checkpoint_sha, config_sha)
    identity = {"model_depth": selected["depth"], "classes": manifest["classes"]}
    seed_all(0)
    model, layers = detector(args, config, identity, "last", smoke=True)
    image, target = NativeDetectionDataset(manifest, "train")[0]
    model.train()
    losses = model([image], [target])
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise ValueError("Native real-backbone smoke loss nonfinite")
    total.backward()
    projection_grad = model.backbone.projection.weight.grad
    if projection_grad is None or not torch.isfinite(projection_grad).all() or not projection_grad.abs().sum():
        raise ValueError("Native trainable neck receives no finite gradient")
    if any(p.grad is not None for p in model.backbone.body.parameters()):
        raise ValueError("Native DINO backbone is not frozen")
    model.eval()
    with torch.inference_mode():
        prediction = model([image])[0]
    report = {"status": "PASS_REAL_HS6_CPU_NATIVE_DETECTION_SMOKE", "not_a_benchmark_score": True,
              "model_id": args.model_id, "checkpoint_sha256": checkpoint_sha, "train_config_sha256": config_sha,
              "layers": layers, "smoke_resize": [64, 96], "losses": {k: float(v.detach()) for k,v in losses.items()},
              "detections": len(prediction["boxes"]), "finite_trainable_neck_gradient": True, "frozen_backbone": True}
    write_new(Path(args.output) / "smoke.json", report)
    print(json.dumps(report))


def run(args):
    registry = json.loads(Path(args.registry).read_text())
    config = configuration(registry, args.dataset)
    manifest = load_data(args)
    if args.command == "smoke":
        return smoke(args, config, manifest)
    if args.local_selection:
        raise ValueError("Native full sweeps require four-machine synchronization, not pre-sync local selection")
    if torch.device(args.device).type != "cuda":
        raise ValueError("Frozen native bf16-backbone recipe requires CUDA; CPU is smoke-only FP32")
    identity = admit(args, config)
    identity.update(registry_sha256=file_digest(args.registry), manifest_sha256=file_digest(args.manifest),
                    batch_size=args.batch_size, classes=manifest["classes"], output_path=str(Path(args.output).resolve()),
                    head_retraining="Fresh seeded detector neck/RPN/RoI for each budget, identical frozen recipe",
                    protocol_config_id=f"native-{args.dataset}-v{config['schema_version']}",
                    precision="CUDA frozen-backbone bf16 autocast, trainable heads float32")
    output = Path(args.output)
    write_new(output / "invocation.json", {**identity, "arguments": vars(args)})
    if args.command == "sweep":
        if args.budget != "1TB":
            raise ValueError("Native protocol search is restricted to 1TB")
        rows = []
        for i, recipe in enumerate(grid(config)):
            try:
                row = train_recipe(args, config, manifest, identity, recipe, output / f"candidate_{i:03d}", "val")
            except Exception as error:
                row = {"status": "FAILURE", "recipe": recipe, "error": str(error), "traceback": traceback.format_exc()}
            rows.append(row)
            write_new(output / f"candidate_{i:03d}.json", row)
        try:
            winner = select(rows, config)
        except ValueError:
            winner = None
        write_new(output / "sweep.json", {**identity, "status": "SUCCESS" if winner else "FAILURE", "rows": rows, "selected": winner})
        if winner is None:
            raise ValueError("Full native detection sweep saved; failed candidates prevent freezing")
        return
    if args.command == "freeze":
        sweep = json.loads(Path(args.sweep).read_text())
        winner = validate_freeze(sweep, identity, config)
        frozen = {**identity, "status": "FROZEN", "selection_budget": "1TB", "config": config,
                  "selected": winner, "selection_sweep_sha256": file_digest(args.sweep)}
        frozen["protocol_sha256"] = digest(frozen)
        write_new(output / "frozen.json", frozen)
        return
    frozen = json.loads(Path(args.frozen).read_text())
    previous = json.loads(Path(args.previous).read_text()) if args.previous else None
    validate_frozen(frozen, identity, previous)
    report = train_recipe(args, frozen["config"], manifest, identity, frozen["selected"]["recipe"], output / "frozen_head", "test")
    if report["zero_based_layers"] != frozen["selected"]["zero_based_layers"]:
        raise ValueError("Frozen native layer taps differ")
    write_new(output / "result.json", {**identity, **report, "protocol_sha256": frozen["protocol_sha256"], "retuned": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "sweep", "freeze", "evaluate"])
    for key in ("dataset", "model-id", "checkpoint", "train-config", "benchmark-root", "manifest", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--registry", default=str(ROOT / "Evaluation Rules/native_detection_candidates.json"))
    parser.add_argument("--models", default=str(ROOT / "Evaluation Rules/unprotocolized_models.json"))
    parser.add_argument("--budget", choices=["1TB", "5TB", "20TB"], default="1TB")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sync-manifest")
    parser.add_argument("--local-selection", action="store_true")
    parser.add_argument("--sweep")
    parser.add_argument("--frozen")
    parser.add_argument("--previous")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        write_new(Path(args.output) / "failure.json", {"status": "FAILURE", "arguments": vars(args),
                                                       "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
