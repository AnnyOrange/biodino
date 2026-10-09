"""Create the exact FM-v3 tail campaign from unvalidated parent cells."""
from __future__ import annotations

import json
import time
from pathlib import Path


PARENT = Path("/mnt/huawei_deepcad/benchmark_model/benchmark_runs/retest_20260918")
OUTPUT = Path("/mnt/huawei_deepcad/benchmark_model/benchmark_runs/fm_v3_completion")


def main() -> None:
    if (OUTPUT / "campaign_manifest.json").exists():
        raise RuntimeError(f"Refusing to replace existing campaign: {OUTPUT}")
    manifest = json.loads((PARENT / "campaign_manifest.json").read_text())
    done = {path.stem for path in (PARENT / "_state/done").glob("*.json")}
    tasks = [
        task
        for task in manifest["tasks"]
        if task["asset"]["arm"].startswith("fm_") and task["key"] not in done
    ]
    expected = {
        "fm_cytoimagenet_ckpretrained__classification__chestmnist",
        *{
            f"fm_{model}_ckpretrained__segmentation__{dataset}__primary-last__formal-static-v1"
            for model in (
                "dinov2", "mae", "siglip2", "pe", "bioclip", "cytoself",
                "jump_cp", "cytoimagenet", "uni", "conch", "phikon2",
                "virchow2", "gigapath", "hoptimus0",
            )
            for dataset in ("livecell", "multimodal_cellseg")
        },
    }
    actual = {task["key"] for task in tasks}
    if actual != expected:
        raise RuntimeError(f"FM tail changed: missing={expected-actual}, extra={actual-expected}")
    manifest["protocol_id"] = "bio-eval-formal-v3-fm-completion"
    manifest["authorization"] = "Explicit user authorization 20260923: fill every GPU below 70%"
    manifest["tasks"] = tasks
    manifest["checkpoint_assets"] = [
        asset for asset in manifest["checkpoint_assets"] if asset["arm"].startswith("fm_")
    ]
    manifest["parent_campaign"] = str(PARENT)
    manifest["created_unix"] = time.time()
    manifest["deadline_unix"] = time.time() + 5 * 3600
    manifest["full_v3_aggregate_allowed"] = False
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PREPARED {len(tasks)} FM tasks at {OUTPUT}")


if __name__ == "__main__":
    main()
