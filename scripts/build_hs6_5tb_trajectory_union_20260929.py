#!/usr/bin/env python3
"""Index the L 5TB continuation and the separate H+ 5TB trajectory."""

import csv
import json
import shlex
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/00_reports/hs6_5tb_trajectory_union_20260929"
TRAIN = REPO / "outputs/01_training_runs"
EVAL = REPO / "outputs/02_eval_runs"
L_RUN = TRAIN / "HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_5tb_mix1m03_5tv107_8x5090zxr_20260907"
G_RUN = TRAIN / "HS6_L5_ck12687_official_gram_a12687_b32_gb1024_noac_4xdeepcad_u2440_contract_v2_20260915"
HOST = "5090-hxw-xzj"

REMOTE_INVENTORY = r'''
import json
from pathlib import Path

l_root = Path('/data/hs6_l_5tb_nogram_eval_20260921')
h_root = Path('/data/hs6_hplus_5tb_eval_20260921')
rows = []
def valid_components(root, step):
    count = 0
    for report in (root / 'v3/cells').glob(f'point_{step}__*/validation_report.json'):
        try:
            count += json.loads(report.read_text()).get('status') == 'VALID_COMPLETE'
        except (OSError, ValueError):
            pass
    return count

for directory in sorted((l_root / 'source/eval').glob('training_*')):
    checkpoint = directory / 'teacher_checkpoint.pth'
    step = int(directory.name[9:])
    status_path = l_root / 'results/_online_status' / f'ckpt_{step}.status.json'
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    rows.append(dict(model='hs6_l_5tb_no_gram', step=int(directory.name[9:]),
                     segment='hxw_continuation_from_ck23911',
                     checkpoint=str(checkpoint), checkpoint_available=checkpoint.is_file(),
                     evaluation_root=str(l_root / 'results' / ('point_' + directory.name[9:])),
                     formal_root=str(l_root / 'v3/cells'),
                     remote_reported_done_lanes=len(status.get('done_lanes', [])),
                     remote_reported_expected_lanes=status.get('expected_lanes', ''),
                     formal_validated_components=valid_components(l_root, step)))
for adapter in sorted(h_root.glob('adapters/[0-9]*')):
    checkpoint = adapter / 'checkpoint.pth'
    if adapter.is_dir():
        step = int(adapter.name)
        rows.append(dict(model='hs6_hplus_5tb', step=int(adapter.name),
                         segment='hplus_5tb', checkpoint=str(checkpoint.resolve()),
                         checkpoint_available=checkpoint.is_file(),
                         evaluation_root=str(h_root / 'old' / ('point_' + adapter.name)),
                         formal_root=str(h_root / 'v3/cells'),
                         remote_reported_done_lanes='', remote_reported_expected_lanes='',
                         formal_validated_components=valid_components(h_root, step)))
print(json.dumps(rows))
'''


def local_rows():
    rows = []
    formal_counts = {}
    coverage = EVAL / "old_v3_protocol_union/coverage.csv"
    with coverage.open(newline="") as stream:
        for record in csv.DictReader(stream):
            if (record["model"] in ("5tb_no_gram", "5tb_gram12687")
                    and record["v3_state"] == "VALID_COMPLETE"):
                key = (record["model"], int(record["checkpoint"]))
                formal_counts[key] = formal_counts.get(key, 0) + 1
    for model, run, segment, old_root in (
        ("hs6_l_5tb_no_gram", L_RUN, "original_and_ck23911_continuation",
         EVAL / "hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908"),
        ("hs6_l_5tb_gram12687", G_RUN, "gram_from_no_gram_ck12687",
         EVAL / "hs6_l5_ck12687_official_gram_curve_fullregistry_3090fleet_20260915"),
    ):
        for directory in sorted((run / "eval").glob("training_*")):
            checkpoint = directory / "teacher_checkpoint.pth"
            step = int(directory.name[9:])
            rows.append(dict(model=model, step=step, segment=segment,
                             checkpoint=str(checkpoint),
                             checkpoint_available=checkpoint.is_file(),
                             evaluation_root=str(old_root / f"point_{step}"),
                             formal_root=str(EVAL / "old_v3_protocol_union/items" /
                                             ("5tb_no_gram" if model.endswith("no_gram")
                                              else "5tb_gram12687") / f"ck{step}"),
                             remote_reported_done_lanes='', remote_reported_expected_lanes='',
                             formal_validated_components=formal_counts.get(
                                 ("5tb_no_gram" if model.endswith("no_gram")
                                  else "5tb_gram12687", step), 0)))
    anchor = L_RUN / "eval/training_12687/teacher_checkpoint.pth"
    if anchor.is_file():
        rows.append(dict(model="hs6_l_5tb_gram12687", step=12687,
                         segment="shared_no_gram_branch_point", checkpoint=str(anchor),
                         checkpoint_available=True,
                         evaluation_root=str(EVAL / "hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908/point_12687"),
                         formal_root=str(EVAL / "old_v3_protocol_union/items/5tb_no_gram/ck12687"),
                         remote_reported_done_lanes='', remote_reported_expected_lanes='',
                         formal_validated_components=formal_counts.get(("5tb_no_gram", 12687), 0)))
    return rows


def main():
    remote = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                             HOST, "python3 -c " + shlex.quote(REMOTE_INVENTORY)],
                            check=True, capture_output=True, text=True)
    rows = local_rows() + json.loads(remote.stdout)
    inventory_by_model = {
        "hs6_l_5tb_no_gram": EVAL / "hs6_5tb_v4_checkpoint_task_inventory_20260924/l_checkpoint_task_inventory.json",
        "hs6_hplus_5tb": EVAL / "hs6_5tb_v4_checkpoint_task_inventory_20260929/hplus_checkpoint_task_inventory.json",
    }
    inventories = {model: json.loads(path.read_text()) for model, path in inventory_by_model.items()}
    selected_v4 = json.loads((EVAL / "hs6_l5_selective_retention_v4_20260923/V4_INVENTORY.json").read_text())
    rows.sort(key=lambda row: (row["model"], row["step"], row["segment"]))
    seen = set()
    for row in rows:
        identity = (row["model"], row["step"])
        if identity in seen:
            raise ValueError(f"Duplicate checkpoint identity: {identity}")
        seen.add(identity)
        row["evidence_host"] = HOST if row["segment"].startswith(("hxw", "hplus")) else "shared"
        row["evaluation_scope"] = "historical_and_component_evidence; see protocol inventory"
        inventory = inventories.get(row["model"], {})
        point = inventory.get("checkpoints", {}).get(str(row["step"]))
        row["v4_id_execution_present"] = ""
        row["v4_id_execution_total"] = ""
        row["v4_inventory"] = ""
        if point is not None:
            id_families = ("classification", "regression", "retrieval", "clustering",
                           "segmentation", "detection_proxy")
            cells = [cell for family in id_families
                     for cell in point.get(family, {}).values()]
            row["v4_id_execution_present"] = sum(
                cell.get("status") in ("RESULT_PRESENT", "VALID_COMPLETE") for cell in cells)
            row["v4_id_execution_total"] = len(cells)
            row["v4_inventory"] = str(inventory_by_model[row["model"]])
        selected_arm = ("N" if row["model"] == "hs6_l_5tb_no_gram" else "G") + str(row["step"])
        row["selected_v4_done_cells"] = selected_v4.get("counts", {}).get(selected_arm, {}).get("DONE", "")

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ("model", "step", "segment", "evidence_host", "checkpoint", "checkpoint_available",
              "evaluation_root", "formal_root", "remote_reported_done_lanes",
              "remote_reported_expected_lanes", "formal_validated_components",
              "v4_id_execution_present", "v4_id_execution_total", "v4_inventory",
              "selected_v4_done_cells", "evaluation_scope")
    with (OUT / "checkpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        summary[model] = dict(count=len(subset), first_step=subset[0]["step"],
                              last_step=subset[-1]["step"],
                              weights_available=sum(row["checkpoint_available"] for row in subset),
                              segments=sorted({row["segment"] for row in subset}))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
