#!/usr/bin/env python3
"""Export reviewed dataset blockers; missing selections/scores never become zero."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"HEST_Benchmark", "OpenCell", "Transloc", "AllenCell_Morphology", "CytoImageNet",
           "ACROBAT", "AllenCell", "CIMA", "CLEM_Reg", "ANHIR"}
FIELDS = ["dataset", "official protocol", "grouping", "DINOv3 task", "searched features/hparams",
          "selected protocol", "1TB", "5TB", "20TB", "protocol status", "code commit", "machine", "blocker"]


def inventory(root=ROOT):
    registry = json.loads((root / "Evaluation Rules/unprotocolized_protocols.json").read_text())
    entries = {}
    for filename in registry["audit_files"]:
        audit = json.loads((root / "Evaluation Rules" / filename).read_text())
        for entry in audit["datasets"]:
            if not {"dataset", "task", "classification", "official_split", "grouping", "metric",
                    "status", "blocker", "sources", "local_evidence"}.issubset(entry):
                raise ValueError(f"Incomplete dataset audit entry in {filename}")
            if entry["classification"] not in {"OFFICIAL", "ESTABLISHED_CONVENTION", "PROPOSED_BY_US"}:
                raise ValueError(f"Undeclared decision label: {entry['dataset']}")
            if entry["dataset"] not in entries:
                entries[entry["dataset"]] = {**entry, "audit_files": [filename]}
            else:
                entries[entry["dataset"]]["audit_files"].append(filename)
    if TARGETS - entries.keys():
        raise ValueError(f"Requested datasets missing from audit: {sorted(TARGETS - entries.keys())}")
    return sorted(entries.values(), key=lambda entry: entry["dataset"].casefold())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    entries = inventory()
    commit = subprocess.check_output(["git", "-c", "safe.directory=" + str(ROOT), "-C", str(ROOT),
                                      "rev-parse", "HEAD"], text=True).strip()
    with (output / "dataset_protocol_audit.json").open("x") as handle:
        json.dump({"status": "REVIEWED_NOT_FULL_BENCHMARK_COMPLETE", "code_commit": commit,
                   "dataset_count": len(entries), "datasets": entries}, handle, indent=2)
        handle.write("\n")
    with (output / "dataset_protocol_summary.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow({"dataset": entry["dataset"],
                             "official protocol": entry["classification"] + ": " + entry["official_split"],
                             "grouping": entry["grouping"], "DINOv3 task": entry["task"] + "; " + entry["metric"],
                             "searched features/hparams": "NOT_RUN; candidates only in registry/audit",
                             "selected protocol": "NOT_SELECTED", "1TB": "NOT_RUN", "5TB": "NOT_RUN", "20TB": "NOT_RUN",
                             "protocol status": entry["status"], "code commit": commit,
                             "machine": "local audit; no model evaluation", "blocker": entry["blocker"]})
    print(json.dumps({"datasets": len(entries), "code_commit": commit, "output": str(output)}))


if __name__ == "__main__":
    main()
