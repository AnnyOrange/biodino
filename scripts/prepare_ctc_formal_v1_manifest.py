#!/usr/bin/env python3
"""Create the fixed five-fold CTC sequence-and-domain-held-out manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = Path("/mnt/huawei_deepcad/benchmark/external_benchmarks_20260901/CellTrackingChallenge/training_zips")
OUTPUT = ROOT / "outputs/02_eval_inputs/formal_v3/ctc"
PROTOCOL = "ctc-labelled-training-sequence-domain-heldout-v1"


def archive_signature(path: Path) -> str:
    """Hash ZIP member names/sizes/CRCs without rereading hundreds of GB."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            digest.update(f"{info.filename}\t{info.file_size}\t{info.CRC}\n".encode())
    return digest.hexdigest()


def inventory(path: Path) -> dict:
    domain = path.stem
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    def count(prefix: str, marker: str) -> int:
        return sum(name.startswith(prefix) and marker in Path(name).name for name in names)
    sequences = {}
    for sequence in ("01", "02"):
        raw_prefix = f"{domain}/{sequence}/"
        tra_prefix = f"{domain}/{sequence}_GT/TRA/"
        seg_prefix = f"{domain}/{sequence}_GT/SEG/"
        sequences[sequence] = {
            "raw_frames": count(raw_prefix, "t"),
            "tra_masks": count(tra_prefix, "man_track"),
            "seg_masks": count(seg_prefix, "man_seg"),
            "track_table": f"{domain}/{sequence}_GT/TRA/man_track.txt",
        }
    return {"domain": domain, "archive": str(path), "archive_central_directory_sha256": archive_signature(path),
            "archive_bytes": path.stat().st_size, "sequences": sequences}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, default=ARCHIVES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archives = sorted(args.archives.glob("*.zip"))
    if len(archives) != 20:
        raise RuntimeError(f"expected 20 CTC archives, found {len(archives)}")
    inventories = [inventory(path) for path in archives]
    rows = []
    domains = [item["domain"] for item in inventories]
    by_domain = {item["domain"]: item for item in inventories}
    for fold in range(5):
        heldout = set(domains[fold::5])
        if len(heldout) != 4:
            raise RuntimeError(f"fold {fold} does not hold out exactly four domains")
        for domain in domains:
            base = by_domain[domain]
            if domain in heldout:
                rows.append({"fold": fold, "role": "test", "domain": domain, "sequence": "02", **base})
            else:
                rows.append({"fold": fold, "role": "head_train", "domain": domain, "sequence": "01", **base})
    manifest = args.output / "split_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    evaluator_smoke = args.output / "native_evaluator_smoke.json"
    evaluator_ready = False
    if evaluator_smoke.exists():
        try:
            evaluator_ready = json.loads(evaluator_smoke.read_text()).get("status") == "PASS"
        except Exception:
            evaluator_ready = False
    report = {"status": "MANIFEST_AND_EVALUATOR_READY_HEAD_LINKER_PENDING" if evaluator_ready else "MANIFEST_READY_EVALUATOR_PENDING", "protocol_id": PROTOCOL,
              "domains": len(domains), "folds": 5, "heldout_domains_per_fold": 4,
              "test_sequences": 20, "head_train_domain_sequences": 80,
              "manifest": str(manifest), "manifest_sha256": digest,
              "native_evaluator_smoke": str(evaluator_smoke), "native_evaluator_ready": evaluator_ready,
              "split_rule": "fold k holds out sorted domains[k::5]; train only sequence 01 of other domains; test sequence 02 of held-out domains"}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
