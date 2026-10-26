#!/usr/bin/env python3
"""Record the existing formal v4 OOD sample identities and ID bank/test split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927/provenance/ood_selected_records.json"
ADMISSION = ROOT / "outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927/FULL_V4_ADMISSION.json"
OUTPUT = ROOT / "outputs/02_eval_inputs/formal_v4/fm18_ood_selection_20261021.json"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def main() -> None:
    source = json.loads(SOURCE.read_text())
    admission = json.loads(ADMISSION.read_text())["ood"]
    if len(source["xray"]) != 992 or len(source["cryo"]) != 80000:
        raise RuntimeError("Formal X-ray/Cryo population count changed")
    if [(part["source"], len(part["indices"])) for part in source["id"]] != [
        ("bloodmnist", 1000), ("bbbc048", 1000), ("cyclops", 1000)
    ]:
        raise RuntimeError("Formal ID reference selection changed")
    for key, records in (("xray", source["xray"]), ("cryo", source["cryo"]),
                         ("id", source["id"])):
        expected = admission[key]["selection_sha256" if key == "id" else "records_sha256"]
        if sha256_bytes(canonical(records)) != expected:
            raise RuntimeError(f"{key} selection differs from the admitted record hash")

    identity = [
        {"position": position, "source": item[0], "source_index": item[1]}
        for position, item in enumerate(
            (part["source"], int(index))
            for part in source["id"]
            for index in part["indices"]
        )
    ]
    if len(identity) != 3000 or len({(row["source"], row["source_index"]) for row in identity}) != 3000:
        raise RuntimeError("ID selection contains missing or duplicate identities")
    order = np.arange(len(identity))
    np.random.default_rng(0).shuffle(order)
    bank = [identity[int(index)] for index in order[:2100]]
    test = [identity[int(index)] for index in order[2100:]]
    payload = {
        "protocol": "fixed-common-ID3000-OOD-native-records-v1",
        "status": "LOCKED_BEFORE_FM18_EVALUATION",
        "source_records": str(SOURCE),
        "source_records_sha256": sha256_bytes(SOURCE.read_bytes()),
        "source_admission": str(ADMISSION),
        "source_admission_sha256": sha256_bytes(ADMISSION.read_bytes()),
        "id_split_algorithm": "numpy.default_rng(0).shuffle(arange(3000)); first 2100 bank, final 900 test",
        "id_bank": bank,
        "id_test": test,
        "id_bank_sha256": sha256_bytes(canonical(bank)),
        "id_test_sha256": sha256_bytes(canonical(test)),
        "xray_records": {"count": 992, "sha256": admission["xray"]["records_sha256"]},
        "cryo_records": {"count": 80000, "sha256": admission["cryo"]["records_sha256"]},
        "settings": admission["settings"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text() != content:
        raise RuntimeError(f"Existing locked selection differs: {OUTPUT}")
    OUTPUT.write_text(content)
    print(str(OUTPUT))
    print(f"bank={len(bank)} test={len(test)} xray=992 cryo=80000")
    print(f"bank_sha256={payload['id_bank_sha256']} test_sha256={payload['id_test_sha256']}")


if __name__ == "__main__":
    main()
