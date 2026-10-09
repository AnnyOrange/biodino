#!/usr/bin/env python3
"""Score every already-evaluated representation arm with the capability-regret metric.

Reference (S*_i, sigma_i, tau_i) is frozen from the vanilla 5TB no-Gram 60-checkpoint
curve.  Arms come from the weight-space campaign's paired/ files (which also carry the
coexistence feature-fusion arms).  A metric enters the cross-campaign aggregate only if
the arm campaign reproduces the curve campaign at E/M/L within 1 sigma; otherwise it is
excluded with an explicit reason and never silently imputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

REPO = Path("/mnt/huawei_deepcad/dinov3")
PAIRED = REPO / "outputs/02_eval_runs/hs6_l5_weightspace_baseline_v4_20260922/paired"
REFDIR = REPO / "outputs/00_reports/hs6_l5_capability_trajectory_20260923"
ARMS = ["E", "M", "L", "WA025", "WA050", "WA075", "AVG3", "E+L", "M+L", "PCA(E+L)->d"]
ANCHOR = {"E": 12687, "M": 20007, "L": 29279}


def load_arm_values() -> dict[str, dict[str, float]]:
    """metric_key -> arm -> value, using the curve campaign's metric naming."""
    out: dict[str, dict[str, float]] = {}
    for path in sorted(PAIRED.glob("*.json")):
        j = json.loads(path.read_text())
        fam_file = path.stem.split("_", 1)[0]
        ds = j.get("dataset", "")
        res = j.get("results")
        if isinstance(res, dict) and res:                      # frozen non-dense
            for arm, vals in res.items():
                if not isinstance(vals, dict):
                    continue
                for met in ("balanced_accuracy", "macro_auc", "r2", "recall_at_1"):
                    if met in vals and isinstance(vals[met], (int, float)):
                        out.setdefault(f"{fam_file}:{ds}:{met}", {})[arm] = float(vals[met])
        elif j.get("family") == "detection_proxy":
            for arm, vals in (j.get("arms") or {}).items():
                v = vals.get("test_patch_f1")
                if isinstance(v, (int, float)):
                    out.setdefault(f"detection:{ds}:test_patch_f1", {})[arm] = float(v)
        elif j.get("family") == "segmentation":
            # average mDice over seeds at the E20 budget (matches the curve budget)
            acc: dict[str, list[float]] = {}
            for cell in j.get("cells", []):
                if cell.get("budget") != 20:
                    continue
                for arm, vals in (cell.get("arms") or {}).items():
                    v = vals.get("test_mDice")
                    if isinstance(v, (int, float)):
                        acc.setdefault(arm, []).append(float(v))
            for arm, vs in acc.items():
                out.setdefault(f"segmentation:{ds}:mDice", {})[arm] = float(np.mean(vs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tol-sigma", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path, default=REFDIR)
    args = ap.parse_args()

    ref = json.loads((REFDIR / "capability_reference_frozen_20260923.json").read_text())
    traj: dict[str, dict[int, float]] = {}
    for rec in csv.DictReader((REFDIR / "trajectory_long.csv").open()):
        traj.setdefault(rec["metric_key"], {})[int(rec["checkpoint"])] = float(rec["value"])

    arm_vals = load_arm_values()
    rows, excluded = [], []
    for key, refi in sorted(ref.items()):
        if key.startswith("ood:"):
            continue
        sig, star = refi["sigma"], refi["S_star"]
        if not math.isfinite(sig) or sig <= 0:
            excluded.append((key, "NO_NOISE_SCALE"))
            continue
        vals = arm_vals.get(key)
        if not vals:
            excluded.append((key, "NOT_EVALUATED_IN_ARM_CAMPAIGN"))
            continue
        worst, bad = 0.0, None
        for arm, ck in ANCHOR.items():
            if arm not in vals or ck not in traj.get(key, {}):
                bad = f"MISSING_ANCHOR_{arm}"
                break
            d = abs(vals[arm] - traj[key][ck]) / sig
            if d > worst:
                worst, = (d,)
        if bad:
            excluded.append((key, bad))
            continue
        if worst > args.tol_sigma:
            excluded.append((key, f"CROSS_CAMPAIGN_OFFSET_{worst:.2f}sigma"))
            continue
        row = {"metric_key": key, "family": key.split(":")[0], "sigma": sig, "S_star": star,
               "star_ck": refi["star_ck"], "tau_ck": refi["tau_ck"], "class": refi["class"],
               "anchor_offset_sigma": worst}
        for arm in ARMS:
            v = vals.get(arm)
            row[arm] = v
            row[f"r_{arm}"] = (max(0.0, star - v) / sig) if isinstance(v, (int, float)) else None
        rows.append(row)

    print(f"included {len(rows)} capabilities; excluded {len(excluded)}")
    for k, why in excluded:
        print(f"  EXCLUDED {k}: {why}")

    # aggregate over the arms that cover every included capability
    full_arms = [a for a in ARMS if all(r[f"r_{a}"] is not None for r in rows)]
    part_arms = [a for a in ARMS if a not in full_arms]
    print(f"\narms with complete coverage of the {len(rows)} capabilities: {full_arms}")
    print(f"arms with partial coverage (scored on their own subset, not comparable): {part_arms}")

    agg = {}
    for a in ARMS:
        rr = [r[f"r_{a}"] for r in rows if r[f"r_{a}"] is not None]
        if not rr:
            continue
        agg[a] = {"n": len(rr), "MNR": float(np.mean(rr)), "maxNR": float(np.max(rr)),
                  "medNR": float(np.median(rr)),
                  "n_within_2sigma": int(sum(1 for x in rr if x < 2.0)),
                  "complete": a in full_arms}

    per_ck = json.loads((REFDIR / "regret_summary.json").read_text())
    print(f"\n{'arm':12s} {'n':>3s} {'MNR':>7s} {'maxNR':>7s} {'medNR':>7s} {'<2sig':>7s}  cov")
    for a, d in sorted(agg.items(), key=lambda kv: kv[1]["MNR"]):
        print(f"{a:12s} {d['n']:3d} {d['MNR']:7.3f} {d['maxNR']:7.2f} {d['medNR']:7.3f} "
              f"{d['n_within_2sigma']:3d}/{d['n']:<3d} {'full' if d['complete'] else 'PARTIAL'}")
    print(f"\n(full-41-metric trajectory reference: endpoint MNR={per_ck['endpoint']['MNR']:.3f}, "
          f"oracle single-ckpt floor MNR={per_ck['oracle_single_checkpoint_floor']['MNR']:.3f} "
          f"at ck{per_ck['oracle_single_checkpoint_floor']['checkpoint']})")

    # restricted oracle floor on exactly the included metric subset
    keys = [r["metric_key"] for r in rows]
    cks = sorted(traj[keys[0]])
    sub = []
    for ck in cks:
        rr = [max(0.0, ref[k]["S_star"] - traj[k][ck]) / ref[k]["sigma"] for k in keys]
        sub.append({"checkpoint": ck, "MNR": float(np.mean(rr)), "maxNR": float(np.max(rr))})
    fl = min(sub, key=lambda d: d["MNR"])
    print(f"oracle single-ckpt floor on THIS {len(keys)}-metric subset: "
          f"MNR={fl['MNR']:.3f} at ck{fl['checkpoint']}")

    out = {"included": len(rows), "excluded": [{"metric_key": k, "reason": w} for k, w in excluded],
           "aggregate": agg, "subset_oracle_floor": fl, "subset_per_checkpoint": sub,
           "full_reference_summary": {"endpoint": per_ck["endpoint"],
                                      "oracle_floor": per_ck["oracle_single_checkpoint_floor"]}}
    (args.out_dir / "arm_regret_scores.json").write_text(json.dumps(out, indent=1))
    with (args.out_dir / "arm_regret_scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
