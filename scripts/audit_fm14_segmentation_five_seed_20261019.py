#!/usr/bin/env python3
"""Inventory v4-preferred FM14 dense cells and reusable frozen caches."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dinov3.eval.bio_frozen_eval.external_fm_protocol import MODELS


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/v4_models"
OUTPUT = ROOT / "outputs/02_eval_runs/fm14_segmentation_five_seed_20261019"


def main() -> None:
    cells = []
    for model in MODELS:
        source = json.loads((REPORT / f"fm_{model}.json").read_text())
        observed = source["checkpoints"]["pretrained"]["cells"]["segmentation"]
        for dataset, entry in sorted(observed.items()):
            preferred = entry.get("preferred")
            candidates = ([preferred] if preferred else []) + entry.get("observations", [])
            results = []
            seen_roots = set()
            for candidate in candidates:
                for name in candidate.get("source", "").split(";"):
                    path = Path(name)
                    if path.name == "results.json" and path.is_file() and path.parents[2] not in seen_roots:
                        seen_roots.add(path.parents[2])
                        results.append(path)
                    if results and dataset != "pannuke":
                        break
                if results and dataset != "pannuke":
                    break
            if not results:
                cells.append({"model": model, "dataset": dataset, "source_result": None,
                              "status": "NO_SOURCE_RESULT"})
            for result in results:
                record = {"model": model, "dataset": dataset, "source_result": str(result)}
                source_root = result.parents[2]
                recipe_file = source_root / "recipe.json"
                if not recipe_file.is_file() and (model, dataset) == ("cytoimagenet", "monuseg"):
                    source_root = ROOT / "outputs/02_eval_runs/fm14_cytoimagenet_monuseg_20261016"
                    recipe_file = source_root / "E20/seed0/recipe.json"
                if not recipe_file.is_file():
                    record["status"] = "NO_RECIPE"
                    cells.append(record)
                    continue
                recipe = json.loads(recipe_file.read_text())
                if recipe.get("model") != model or recipe.get("dataset") != dataset:
                    record["status"] = "RECIPE_MISMATCH"
                    cells.append(record)
                    continue
                caches = {part: source_root / "cache" / dataset / model / f"{part}.npz"
                          for part in ("train", "val", "test")}
                record.update(source_root=str(source_root), recipe=str(recipe_file),
                              split_protocol=recipe.get("split_protocol"),
                              caches={part: str(path) for part, path in caches.items()},
                              existing={str(budget): [seed for seed in range(5)
                                                       if (source_root / f"E{budget}" / f"seed{seed}" / "results.json").is_file()]
                                        for budget in (20, 50)})
                record["status"] = "READY_CACHED" if all(path.is_file() for path in caches.values()) else "NO_CACHE"
                cells.append(record)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {"cells": cells, "status_counts": dict(Counter(row["status"] for row in cells))}
    (OUTPUT / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"cells": len(cells), **payload["status_counts"]}))


if __name__ == "__main__":
    main()
