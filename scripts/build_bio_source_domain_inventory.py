#!/usr/bin/env python3
"""Inventory source datasets recoverable from packed-data path mappings."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


IDR_COMPONENT = re.compile(r"^idr\d{4}(?:[-_][^/]+)?$", re.IGNORECASE)
CHANNEL_COMPONENT = re.compile(r"^ch(\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mappings", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def source_domain(src_path: str) -> tuple[str, str]:
    parts = [part for part in PurePosixPath(src_path).parts if part not in {"", "/"}]
    for part in parts:
        if IDR_COMPONENT.match(part):
            return f"idr:{part.lower()}", "idr"

    for marker, family in (
        ("0-large-model-dataset", "external_collection"),
        ("000-LM-dataset-preprocessed", "preprocessed_collection"),
    ):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                component = parts[index + 1]
                if component != "0-large-model-dataset":
                    return f"{family}:{component.lower()}", family
                if index + 2 < len(parts):
                    return f"external_collection:{parts[index + 2].lower()}", "external_collection"

    if parts:
        return f"unresolved:{parts[-2].lower() if len(parts) > 1 else parts[0].lower()}", "unresolved"
    return "unresolved:empty", "unresolved"


def main() -> None:
    args = parse_args()
    domain_counts = Counter()
    family_counts = Counter()
    channel_counts = defaultdict(Counter)
    examples = defaultdict(list)
    total_rows = 0
    malformed_rows = 0

    stop = False
    for mapping in args.mappings:
        with mapping.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if args.max_rows is not None and total_rows >= args.max_rows:
                    stop = True
                    break
                if len(row) < 3:
                    malformed_rows += 1
                    continue
                _, channel_text, src_path = row[:3]
                domain, family = source_domain(src_path)
                channel_match = CHANNEL_COMPONENT.match(channel_text.strip())
                channel_count = int(channel_match.group(1)) if channel_match else 0
                domain_counts[domain] += 1
                family_counts[family] += 1
                channel_counts[domain][channel_count] += 1
                if len(examples[domain]) < 3 and src_path not in examples[domain]:
                    examples[domain].append(src_path)
                total_rows += 1
        if stop:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain, count in domain_counts.most_common():
        family = domain.split(":", 1)[0]
        row = {
            "domain": domain,
            "family": family,
            "mapping_rows": count,
            "fraction": count / total_rows if total_rows else 0.0,
            "dominant_channel_count": channel_counts[domain].most_common(1)[0][0],
            "channel_histogram": json.dumps(dict(sorted(channel_counts[domain].items()))),
            "example_src_path": examples[domain][0] if examples[domain] else "",
        }
        rows.append(row)

    csv_path = args.output_dir / "source_domains.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["domain"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "mappings": [str(path) for path in args.mappings],
        "total_mapping_rows": total_rows,
        "malformed_rows": malformed_rows,
        "num_source_domains": len(domain_counts),
        "family_counts": dict(family_counts),
        "global_channel_histogram": dict(
            sorted(
                sum((counts for counts in channel_counts.values()), Counter()).items()
            )
        ),
        "top_domains": rows[:50],
        "interpretation": (
            "Counts are packed mapping rows, not unique biological specimens. "
            "Acquisition modality and organism still require a curated source-domain catalog."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
