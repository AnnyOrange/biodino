#!/usr/bin/env python3
"""Hard preflight for the approved biological evaluation protocol v3."""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from validate_bio_eval_formal_v1 import _sha256, _validate_conic, _validate_pannuke, _validate_registry
from validate_bio_eval_formal_v2 import _validate_dataset_policy, _validate_livecell, _validate_manifest_v2


PROTOCOL_PATH = REPO / "Evaluation Rules" / "protocol_v3.json"


def _validate_v3_additions(protocol: dict) -> dict:
    tier_b_retrieval = set(protocol["tier_b"]["retrieval"])
    if "rxrx3-core" not in tier_b_retrieval:
        raise AssertionError("v3 Tier B retrieval must include rxrx3-core")
    if set(protocol["tier_b"].get("cell_tracking", [])) != {"ctc"}:
        raise AssertionError("v3 Tier B cell_tracking must be exactly ctc")
    if "midogpp" not in set(protocol["forbidden_formal_datasets"]):
        raise AssertionError("MIDOG++ must be forbidden from formal v3 manifests")
    if "detection/midogpp" not in set(protocol["forbidden_formal_task_datasets"]):
        raise AssertionError("detection/midogpp must be an excluded formal pair")

    rxrx3 = protocol["retrieval_splits"]["rxrx3-core"]
    ctc = protocol["cell_tracking_splits"]["ctc"]
    pending = []
    if rxrx3.get("manifest_sha256") == "PENDING_IMPLEMENTATION":
        pending.append("rxrx3-core fixed eligible-gene manifest/hash")
    if ctc.get("manifest_sha256") == "PENDING_IMPLEMENTATION":
        pending.append("CTC sequence/domain-heldout manifest/hash")
    if ctc.get("formal_status") != "READY_NATIVE_EVALUATOR":
        pending.append("CTC fixed frozen instance head/linker implementation")
    if pending:
        raise AssertionError("formal v3 is approved but not launch-ready: " + "; ".join(pending))
    return {"rxrx3-core": rxrx3, "ctc": ctc, "midogpp": "EXCLUDED"}


def _validate_manifest_v3(path: Path, protocol: dict) -> dict:
    rows = json.loads(path.read_text())
    tracking = [row for row in rows if str(row.get("task")) == "cell_tracking"]
    if {str(row.get("dataset")) for row in tracking} != {"ctc"}:
        raise AssertionError("formal v3 manifest must contain exactly cell_tracking/ctc")
    for row in tracking:
        cmd = [str(x) for x in row.get("cmd", [])]
        if "--batch-size" not in cmd or cmd[cmd.index("--batch-size") + 1] != str(protocol["batch_sizes"]["cell_tracking"]):
            raise AssertionError("cell_tracking/ctc must use the v3 tracking batch size")
    return _validate_manifest_v2(path, protocol)


def build_report(protocol: dict, benchmark_root: Path, command_manifest: Path | None) -> dict:
    checks = {}
    validators = {
        "dataset_policy": lambda: _validate_dataset_policy(protocol),
        "v3_additions": lambda: _validate_v3_additions(protocol),
        "registry": lambda: _validate_registry(protocol),
        "conic": lambda: _validate_conic(protocol, benchmark_root),
        "livecell": lambda: _validate_livecell(protocol, benchmark_root),
        "pannuke": lambda: _validate_pannuke(protocol, benchmark_root),
    }
    for name, validate in validators.items():
        try:
            checks[name] = {"status": "PASS", "evidence": validate()}
        except Exception as error:
            checks[name] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    if command_manifest is None:
        checks["commands"] = {"status": "FAIL", "error": "No command manifest; launch matrix is unverified"}
    else:
        try:
            checks["commands"] = {"status": "PASS", "evidence": _validate_manifest_v3(command_manifest, protocol)}
        except Exception as error:
            checks["commands"] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    return {
        "status": "PASS" if all(row["status"] == "PASS" for row in checks.values()) else "FAIL",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "checks": checks,
        "failed_checks": [name for name, row in checks.items() if row["status"] == "FAIL"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--command-manifest", default=None)
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL_PATH.read_text())
    report = build_report(protocol, Path(args.benchmark_root),
                          Path(args.command_manifest) if args.command_manifest else None)
    report["git_commit"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    report["git_status_porcelain"] = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    git_ok = bool(args.expected_commit and args.expected_commit == report["git_commit"]
                  and not report["git_status_porcelain"].strip())
    report["checks"]["git"] = {
        "status": "PASS" if git_ok else "FAIL",
        "expected_commit": args.expected_commit,
        "error": None if git_ok else "Expected commit must be explicit and evaluation checkout must be clean",
    }
    report["failed_checks"] = [name for name, row in report["checks"].items() if row["status"] == "FAIL"]
    report["status"] = "FAIL" if report["failed_checks"] else "PASS"
    report["scope"] = "Protocol/split/command checks only; checkpoint and resource launch gates are also required"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"PRECHECK FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
