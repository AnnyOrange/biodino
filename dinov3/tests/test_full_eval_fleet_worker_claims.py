from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_hs6_l_6m_full_eval_fleet_worker.py"
SPEC = importlib.util.spec_from_file_location("full_eval_fleet_worker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


def test_fresh_heartbeat_protects_old_cross_host_claim(tmp_path: Path) -> None:
    claim = tmp_path / "job.lock"
    claim.mkdir()
    owner = {
        "claimed_at_unix": time.time() - 10_000,
        "heartbeat_at_unix": time.time(),
        "host": "another-host",
        "pid": 123,
    }
    (claim / "owner.json").write_text(json.dumps(owner))

    assert not worker.clear_stale_claim(claim, stale_seconds=60)
    assert claim.is_dir()


def test_refresh_claim_lease_preserves_origin_and_adds_heartbeat(tmp_path: Path) -> None:
    claim = tmp_path / "job.lock"
    claim.mkdir()
    owner = {
        "claimed_at_unix": 123.0,
        "claimed_at_utc": "origin",
        "host": "worker-host",
        "pid": 456,
    }

    worker.refresh_claim_lease(claim, owner)

    refreshed = json.loads((claim / "owner.json").read_text())
    assert refreshed["claimed_at_unix"] == 123.0
    assert refreshed["claimed_at_utc"] == "origin"
    assert refreshed["heartbeat_at_unix"] > 123.0
    assert refreshed["heartbeat_at_utc"]
