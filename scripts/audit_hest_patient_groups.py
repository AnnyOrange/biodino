"""Bind official released anonymized patient labels to HEST preflight folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


OFFICIAL_GIT_BLOB = "c4348a7787c030e403bebd73639892305b61970d"
OFFICIAL_SOURCE_URL = (
    "https://github.com/mahmoodlab/HEST/blob/"
    "3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/assets/HEST_v1_1_0.csv"
)
OFFICIAL_PAPER_URL = (
    "https://papers.nips.cc/paper_files/paper/2024/file/"
    "60a899cc31f763be0bde781a75e04458-Paper-Datasets_and_Benchmarks_Track.pdf"
)


def verify_metadata(payload: bytes) -> dict[str, dict[str, str]]:
    blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    if blob != OFFICIAL_GIT_BLOB:
        raise ValueError("HEST metadata does not match the pinned official Git blob")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    if not rows or not {"id", "patient", "dataset_title"} <= rows[0].keys():
        raise ValueError("HEST metadata missing required identity columns")
    identities = [row["id"] for row in rows]
    if any(not value for value in identities) or len(set(identities)) != len(identities):
        raise ValueError("HEST metadata has duplicate or missing sample IDs")
    return {row["id"]: row for row in rows}


def audit_patient_groups(preflight: dict, payload: bytes) -> dict:
    metadata = verify_metadata(payload)
    tasks = preflight.get("tasks", {})
    if not tasks:
        raise ValueError("HEST preflight has no tasks")
    records = []
    mapping = {}
    failures = []
    for tissue, task in sorted(tasks.items()):
        folds = task.get("folds", {})
        if not folds:
            raise ValueError(f"HEST tissue has no folds: {tissue}")
        for fold, partitions in sorted(folds.items()):
            ids = {}
            patients = {}
            unknown = {}
            for split in ("train", "test"):
                ids[split] = partitions.get(split, {}).get("sample_ids", [])
                if not ids[split] or len(set(ids[split])) != len(ids[split]):
                    raise ValueError(f"Missing/duplicate fold sample IDs: {tissue}/{fold}/{split}")
                patients[split] = set()
                unknown[split] = []
                for sid in ids[split]:
                    if sid not in metadata:
                        raise ValueError(f"Sample missing from official metadata: {sid}")
                    row = metadata[sid]
                    # Generic Patient1 labels are not globally unique across cohorts.
                    label = "".join(row["patient"].casefold().split())
                    key = f"{tissue}:{label}" if label else None
                    mapping[(tissue, sid)] = {
                        "tissue": tissue, "sample_id": sid,
                        "released_patient_label": row["patient"], "patient_key": key,
                        "dataset_title": row["dataset_title"],
                        "study_link": row.get("study_link", ""),
                        "download_page_link1": row.get("download_page_link1", ""),
                    }
                    if key:
                        patients[split].add(key)
                    else:
                        unknown[split].append(sid)
            overlap = sorted(patients["train"] & patients["test"])
            sample_overlap = sorted(set(ids["train"]) & set(ids["test"]))
            if overlap or sample_overlap:
                failures.append({"tissue": tissue, "fold": fold,
                                 "patient_overlap": overlap, "sample_overlap": sample_overlap})
            records.append({
                "tissue": tissue, "fold": fold,
                "train_sample_ids": ids["train"], "test_sample_ids": ids["test"],
                "train_patient_keys": sorted(patients["train"]),
                "test_patient_keys": sorted(patients["test"]),
                "unknown_train_ids": unknown["train"], "unknown_test_ids": unknown["test"],
                "known_patient_overlap": overlap, "sample_overlap": sample_overlap,
                "known_patient_disjoint": not overlap and not sample_overlap,
                "fully_patient_verified": not overlap and not sample_overlap and not any(unknown.values()),
            })
    unknown_ids = sorted({row["sample_id"] for row in mapping.values() if row["patient_key"] is None})
    if failures:
        status = "FAIL_KNOWN_PATIENT_OVERLAP"
    elif unknown_ids == ["TENX111"]:
        status = "PASS_KNOWN_PATIENTS_ONE_UNRESOLVED"
    elif unknown_ids:
        status = "FAIL_UNRESOLVED_PATIENT_IDENTITIES"
    else:
        status = "PASS_ALL_RELEASED_PATIENTS"
    return {
        "status": status, "metadata_sha256": hashlib.sha256(payload).hexdigest(),
        "metadata_filename": "patient_metadata.csv", "official_source_url": OFFICIAL_SOURCE_URL,
        "official_git_blob_sha1": OFFICIAL_GIT_BLOB,
        "slides": len(mapping), "known_slides": sum(bool(row["patient_key"]) for row in mapping.values()),
        "unknown_ids": unknown_ids, "all_fold_records": records,
        "sample_patient_mapping": sorted(mapping.values(), key=lambda row: (row["tissue"], row["sample_id"])),
        "failures": failures,
        "patient_disjoint_fully_verified": not failures and not unknown_ids,
        "patient_key_scope": "Released anonymized label casefold/whitespace-normalized within benchmark tissue/cohort; never joined globally",
        "official_paper_assertion": {
            "source_url": OFFICIAL_PAPER_URL,
            "section": "5.1; Appendix C.3 and Table A11",
            "assertion": "Authors describe patient-stratified folds to avoid patient leakage; this assertion is distinct from our explicit released-label audit",
        },
        "limitations": [
            "TENX111 COAD patient field is blank; no fabricated label or inferred personal identity",
            "Only public released anonymized grouping labels are used, no patient reidentification",
            "Companion table v1.1.0 pinned to official published Git blob; newer HF table may be gated",
        ],
    }


def write_checked_preflight(preflight_path: Path, metadata_path: Path) -> dict:
    original = json.loads(preflight_path.read_text())
    payload = metadata_path.read_bytes()
    grouping = audit_patient_groups(original, payload)
    destination = preflight_path.parent
    (destination / "patient_metadata.csv").write_bytes(payload)
    (destination / "patient_grouping.json").write_text(json.dumps(grouping, indent=2) + "\n")
    checked = {**original, "patient_grouping": grouping}
    (destination / "preflight_patient_checked.json").write_text(json.dumps(checked, indent=2) + "\n")
    return grouping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    grouping = write_checked_preflight(args.preflight, args.metadata)
    print(json.dumps({key: grouping[key] for key in ("status", "slides", "known_slides", "unknown_ids", "metadata_sha256")}))
    if not grouping["status"].startswith("PASS_"):
        raise SystemExit("HEST patient grouping failed; checked preflight is scientifically blocked")


if __name__ == "__main__":
    main()
