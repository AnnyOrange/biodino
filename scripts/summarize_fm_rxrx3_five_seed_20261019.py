#!/usr/bin/env python3
"""Validate and summarize the FM14/new-FM RxRx3-core five-seed results."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from pathlib import Path

from dinov3.eval.bio_frozen_eval.external_fm_protocol import MODELS as FM14


ROOT = Path(__file__).resolve().parents[1]
FM14_RUN = ROOT / "outputs/02_eval_runs/fm14_rxrx3_five_seed_20261019"
NEW_RUN = ROOT / "outputs/02_eval_runs/new_fm_v4_matched_20261016"
NEW_MODELS = ("distillpath", "morphem", "genbio_pathfm", "openpath")
FIELDS = ("group", "model", "dataset", "n_seeds", "n_gallery", "n_query",
          "recall_at_1", "recall_at_5", "recall_at_10", "mrr",
          "seed0_nmi", "seed1_nmi", "seed2_nmi", "seed3_nmi", "seed4_nmi",
          "mean_nmi", "sd_nmi", "protocol_id", "split_sha256")


def summarize(group: str, model: str, folder: Path) -> dict:
    payloads = [json.loads((folder / f"seed{seed}.json").read_text()) for seed in range(5)]
    reference = payloads[0]
    nmis = []
    for seed, payload in enumerate(payloads):
        if (payload["model"], payload["dataset"], payload["seed"], payload["status"]) != (
                model, "rxrx3-core", seed, "VALID_COMPLETE"):
            raise ValueError(f"Invalid model/dataset/seed/status: {folder}/seed{seed}.json")
        if (payload["protocol_id"], payload["split_sha256"],
                payload["n_gallery"], payload["n_query"], payload["retrieval"]) != (
                reference["protocol_id"], reference["split_sha256"], 734, 734,
                reference["retrieval"]):
            raise ValueError(f"Split or retrieval changed across seeds: {folder}")
        nmi = float(payload["clustering"]["nmi"])
        if not math.isfinite(nmi):
            raise ValueError(f"Nonfinite NMI: {folder}/seed{seed}.json")
        nmis.append(nmi)
    return {"group": group, "model": model, "dataset": "rxrx3-core", "n_seeds": 5,
            "n_gallery": 734, "n_query": 734, **reference["retrieval"],
            **{f"seed{seed}_nmi": nmi for seed, nmi in enumerate(nmis)},
            "mean_nmi": statistics.mean(nmis), "sd_nmi": statistics.stdev(nmis),
            "protocol_id": reference["protocol_id"],
            "split_sha256": reference["split_sha256"]}


def write(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    fm14 = [summarize("FM14", model, FM14_RUN / model / "rxrx3_core_five_seed")
            for model in FM14]
    new = [summarize("new FM", model, NEW_RUN / model / "rxrx3_core_five_seed")
           for model in NEW_MODELS]
    write(FM14_RUN / "seed_summary.csv", fm14)
    write(NEW_RUN / "rxrx3_seed_summary.csv", new)
    print(f"FM14={len(fm14)}/{len(FM14)} new_FM={len(new)}/{len(NEW_MODELS)}")


if __name__ == "__main__":
    main()
