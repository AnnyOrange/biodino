#!/usr/bin/env python3
"""CPU metadata audit across complete randomly chosen shards, not shard prefixes.

This is a sampled inventory audit, not a measurement of consumed training batches.
No image decoding, feature extraction or benchmark data is involved.
"""
import argparse
from collections import Counter
import csv
import glob
import json
from pathlib import Path
import random
import re
import tarfile
import time

POOLS = {
    "1tb": "/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*.tar",
    "5tb": "/mnt/huawei_blm/deepcad_5t_v1/wds_patched_shuffle/filtered_mixed_train*.tar",
    "route2_r0": "/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923/route2_15tb_no_old5_micro_tars/filtered_projection_20TB_nested-r0*.tar",
    "route2_r9": "/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923/route2_15tb_no_old5_micro_tars/filtered_projection_20TB_nested-r9*.tar",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards-per-pool", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261016)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    summary = dict(seed=args.seed, started=time.time(), sampling="uniform random shards; full metadata scan",
                   limitation="Sampled shards, not global census or consumed-batch audit; source path is not necessarily specimen ID",
                   pools={})
    with (args.output / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pool", "shard", "key", "dataset", "source_image", "channels"])
        writer.writeheader()
        for pool, pattern in POOLS.items():
            shards = sorted(glob.glob(pattern))
            chosen = rng.sample(shards, min(args.shards_per_pool, len(shards)))
            datasets, sources, channels = Counter(), Counter(), Counter()
            missing = 0
            completed, errors = [], []
            for shard in chosen:
                try:
                    # Seek over TIFF payloads instead of reading/decompressing them.
                    with tarfile.open(shard, "r:") as archive:
                        for member in archive:
                            if not member.name.endswith(".meta.json"):
                                continue
                            d = json.load(archive.extractfile(member))
                            source = d.get("source_path") or d.get("original_path") or ""
                            dataset = d.get("source_dataset") or d.get("dataset_name") or ""
                            if not dataset:
                                match = re.search(r"/(idr\d{4}[^/]*|\d+-[^/]+)/", source)
                                dataset = match.group(1) if match else "unknown"
                            channel = str(d.get("channel_count", d.get("source_channel_count", "unknown")))
                            datasets[dataset] += 1
                            channels[channel] += 1
                            if source:
                                sources[source] += 1
                            else:
                                missing += 1
                            writer.writerow(dict(pool=pool, shard=Path(shard).name,
                                                 key=member.name[:-len(".meta.json")], dataset=dataset,
                                                 source_image=source, channels=channel))
                    completed.append(shard)
                except Exception as exc:
                    errors.append(dict(shard=shard, error=str(exc)))
                f.flush()
                total = sum(datasets.values())
                summary["pools"][pool] = dict(total_shards=len(shards), chosen_shards=chosen,
                    completed_shards=completed, errors=errors, tiles=total,
                    known_source_tiles=sum(sources.values()), unknown_source_tiles=missing,
                    unique_known_sources=len(sources),
                    tiles_per_known_source=sum(sources.values()) / len(sources) if sources else None,
                    dataset_counts=dict(datasets.most_common()), channel_counts=dict(channels.most_common()))
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(pool, len(completed), "shards", total, "tiles", len(sources), "sources", flush=True)
    summary["finished"] = time.time()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
