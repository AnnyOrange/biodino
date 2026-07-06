"""
Assemble the Line-2 final table without mixing val/test sources.

The script reads already generated JSON files for:
  - frozen bio-DINOv3 + HoVerNet decoder,
  - generic DINOv3 + the same decoder,
  - cpsam zero-shot,
  - cpsam fine-tuned.

By default it only accepts entries for ``--split test``. Missing cells stay
missing instead of silently falling back to val results.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DATASETS = ("pannuke", "tissuenet", "conic", "bbbc038", "livecell", "monuseg")
MODELS = ("frozen_bio", "generic_dinov3", "cpsam_zero_shot", "cpsam_finetuned")


@dataclass
class Entry:
    split: str
    metrics: Dict[str, float]
    source: str


def _candidates(dataset: str) -> Dict[str, List[str]]:
    common_bio = f"outputs/instance_seg/test_eval/{dataset}/bio.json"
    common_generic = f"outputs/instance_seg/test_eval/{dataset}/generic_lvd.json"
    common_cpsam = f"outputs/instance_seg/test_eval/{dataset}/cpsam/results.json"
    common_cpsam_ft = [
        f"outputs/instance_seg/test_eval/{dataset}/cpsam_ft/results.json",
        f"outputs/instance_seg/cpsam_ft_test/{dataset}/results.json",
        f"outputs/instance_seg/cpsam_ft/{dataset}/test_results.json",
        # Deliberately last: existing files may be val-only and will be rejected
        # unless --allow-non-test is set.
        f"outputs/instance_seg/cpsam_ft/{dataset}/results.json",
    ]

    out = {
        "frozen_bio": [common_bio],
        "generic_dinov3": [common_generic],
        "cpsam_zero_shot": [common_cpsam],
        "cpsam_finetuned": common_cpsam_ft,
    }
    if dataset == "pannuke":
        out["frozen_bio"] = ["outputs/instance_seg/full_eval/bio_12299.json", common_bio]
        out["generic_dinov3"] = ["outputs/instance_seg/full_eval/generic_lvd.json", common_generic]
        out["cpsam_zero_shot"] = [
            "outputs/instance_seg/full_eval/cpsam_zs_full/results.json",
            common_cpsam,
            "outputs/instance_seg/specialist/cpsam/pannuke/results.json",
        ]
    elif dataset == "monuseg":
        out["frozen_bio"] = ["outputs/instance_seg/monuseg_test/bio_test.json", common_bio]
        out["cpsam_zero_shot"] = ["outputs/instance_seg/monuseg_test/cpsam_test/results.json", common_cpsam]
    return out


def _extract_entry(path: Path, split: str, allow_non_test: bool) -> Optional[Entry]:
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)

    found_split = None
    metrics = None
    if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
        found_split = str(data.get("split", "unknown"))
        metrics = data["metrics"]
    elif isinstance(data, dict) and isinstance(data.get(split), dict):
        found_split = split
        metrics = data[split]
    elif allow_non_test and isinstance(data, dict):
        for key in ("test", "val", "train"):
            if isinstance(data.get(key), dict):
                found_split = key
                metrics = data[key]
                break

    if metrics is None or found_split is None:
        return None
    if found_split != split and not allow_non_test:
        return None
    return Entry(split=found_split, metrics=metrics, source=str(path))


def _first_valid(paths: Iterable[str], root: Path, split: str, allow_non_test: bool) -> Optional[Entry]:
    for raw in paths:
        entry = _extract_entry(root / raw, split=split, allow_non_test=allow_non_test)
        if entry is not None:
            return entry
    return None


def _fmt(entry: Optional[Entry]) -> str:
    if entry is None:
        return "NA"
    aji = entry.metrics.get("AJI")
    bpq = entry.metrics.get("bPQ")
    if aji is None:
        return "NA"
    if bpq is None:
        return f"{aji:.3f}"
    return f"{aji:.3f}/{bpq:.3f}"


def _verdict(bio: Optional[Entry], zs: Optional[Entry], ft: Optional[Entry]) -> str:
    if bio is None:
        return "NA"
    target = ft or zs
    if target is None:
        return "NA"
    b = bio.metrics.get("AJI")
    t = target.metrics.get("AJI")
    if b is None or t is None:
        return "NA"
    name = "cpsam-ft" if ft is not None else "cpsam-zs"
    return ("WIN" if b >= t else "lose") + f" vs {name}"


def _markdown_table(rows: List[Dict]) -> str:
    headers = ["dataset", "frozen bio", "generic DINOv3", "cpsam zero-shot", "cpsam fine-tuned", "verdict"]
    lines = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            "| {dataset} | {frozen_bio} | {generic_dinov3} | {cpsam_zero_shot} | "
            "{cpsam_finetuned} | {verdict} |".format(**row)
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Summarize Line-2 official split results.")
    p.add_argument("--root", default=".", help="Repository/output root.")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--split", default="test")
    p.add_argument("--allow-non-test", action="store_true",
                   help="Allow val/train fallback. Default refuses non-test rows.")
    p.add_argument("--out-json", default=None)
    p.add_argument("--show-sources", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    summary = {"split": args.split, "allow_non_test": bool(args.allow_non_test), "datasets": {}}
    rows = []
    for dataset in args.datasets:
        entries = {
            model: _first_valid(paths, root, args.split, args.allow_non_test)
            for model, paths in _candidates(dataset).items()
        }
        row = {
            "dataset": dataset,
            "frozen_bio": _fmt(entries["frozen_bio"]),
            "generic_dinov3": _fmt(entries["generic_dinov3"]),
            "cpsam_zero_shot": _fmt(entries["cpsam_zero_shot"]),
            "cpsam_finetuned": _fmt(entries["cpsam_finetuned"]),
            "verdict": _verdict(entries["frozen_bio"], entries["cpsam_zero_shot"], entries["cpsam_finetuned"]),
        }
        rows.append(row)
        summary["datasets"][dataset] = {
            model: (
                {"split": e.split, "metrics": e.metrics, "source": e.source}
                if e is not None else None
            )
            for model, e in entries.items()
        }
        summary["datasets"][dataset]["verdict"] = row["verdict"]

    print(_markdown_table(rows))
    if args.show_sources:
        print("\nSources:")
        for dataset, model_map in summary["datasets"].items():
            for model in MODELS:
                entry = model_map.get(model)
                source = entry["source"] if entry else "MISSING"
                split = entry["split"] if entry else "-"
                print(f"{dataset:10s} {model:17s} split={split:5s} {source}")

    if args.out_json:
        out = root / args.out_json
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
