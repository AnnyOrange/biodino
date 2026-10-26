#!/usr/bin/env python3
"""Extend one audited FM14 v4 dense cell to seeds 3-4 from frozen caches."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import torch

from dinov3.eval.bio_frozen_eval.external_fm_protocol import validate_best_head
from dinov3.eval.bio_segmentation.linear_probe import DATASET_CONFIGS, run_cached_linear_probe


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/02_eval_runs/fm14_segmentation_five_seed_20261019"


def run_cell(cell: dict, cell_index: int) -> None:
    recipe = json.loads(Path(cell["recipe"]).read_text())
    torch.set_num_threads(2)
    for budget in (20, 50):
        for seed in (3, 4):
            output = OUT / "results" / cell["model"] / cell["dataset"] / cell["split_protocol"] / f"E{budget}" / f"seed{seed}"
            result = output / "results.json"
            if result.is_file():
                payload = json.loads(result.read_text())
                if int(payload["_meta"]["seed"]) != seed:
                    raise ValueError(f"Wrong seed in {result}")
                validate_best_head(payload["_meta"])
                print(f"[skip] {cell_index}/{budget}/{seed}", flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            extension = {"model": cell["model"], "dataset": cell["dataset"],
                         "split_protocol": cell["split_protocol"], "budget": budget,
                         "seed": seed, "source_recipe": cell["recipe"],
                         "source_result": cell["source_result"], "caches": cell["caches"],
                         "protocol_id": "bio-eval-union-v4", "status": "OBSERVATIONAL"}
            (output / "five_seed_extension.json").write_text(json.dumps(extension, indent=2) + "\n")
            run_cached_linear_probe(
                train_cache=cell["caches"]["train"], val_cache=cell["caches"]["val"],
                test_cache=cell["caches"]["test"], output_dir=str(output),
                **DATASET_CONFIGS[cell["dataset"]], epochs=budget,
                lr=recipe.get("lr", 1e-3), batch_size=recipe["probe_batch_size"],
                weight_decay=recipe.get("weight_decay", 1e-4),
                dropout=recipe.get("dropout", 0.1), num_workers=0,
                eval_every=1, seed=seed, class_weight_mode=recipe["class_weight_mode"],
            )
            payload = json.loads(result.read_text())
            validate_best_head(payload["_meta"])
            print(f"[complete] {cell_index} {cell['model']}/{cell['dataset']}/{cell['split_protocol']}/E{budget}/seed{seed}",
                  flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args()
    audit = json.loads((OUT / "audit.json").read_text())["cells"]
    if not 0 <= args.cell_index < len(audit):
        parser.error("cell index out of range")
    cell = audit[args.cell_index]
    if cell["status"] != "READY_CACHED":
        raise RuntimeError(f"Cell {args.cell_index} is not cache-ready")
    lock_dir = OUT / "locks"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f"cell{args.cell_index}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run_cell(cell, args.cell_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
