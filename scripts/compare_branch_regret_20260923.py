#!/usr/bin/env python3
"""Matched branch comparison under the capability-regret metric.

Both arms resume the same checkpoint and consume the same stream for the same number
of updates; only the regularisation differs.  At every matched update count we score
each arm with the frozen reference (S*_i, sigma_i) taken from the full vanilla
trajectory, and report MNR / maxNR plus the retention and plasticity split that the
objective requires:

  retention set   capabilities whose reference peak tau_i is at or before the branch
                  point (already matured when the branch started)
  plasticity set  capabilities whose reference peak is after the branch point
                  (still growing, so the branch must not stall them)

A method must lower MNR on the retention set AND not raise it on the plasticity set.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np

REPO = Path("/mnt/huawei_deepcad/dinov3")
REF = REPO / "outputs/00_reports/hs6_l5_capability_trajectory_20260923"


def read(p: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for r in csv.DictReader(p.open()):
        out.setdefault(r["metric_key"], {})[int(r["checkpoint"])] = float(r["value"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=REF / "trajectory_long.csv")
    ap.add_argument("--branch", type=Path,
                    default=REPO / "outputs/00_reports/hs6_l5_gram_branch_trajectory_20260923/trajectory_long.csv")
    ap.add_argument("--branch-point", type=int, default=12687)
    ap.add_argument("--branch-name", default="official-Gram")
    ap.add_argument("--out", type=Path, default=REF / "branch_regret_gram_vs_vanilla.json")
    a = ap.parse_args()

    ref = json.loads((REF / "capability_reference_frozen_20260923.json").read_text())
    base, br = read(a.baseline), read(a.branch)

    # capabilities usable in both arms at every matched checkpoint
    cks = sorted(set(br[next(iter(br))].keys()) if br else [])
    keys = []
    for k, r in ref.items():
        if k.startswith("ood:") or not np.isfinite(r["sigma"]) or r["sigma"] <= 0:
            continue
        if k not in br or k not in base:
            continue
        keys.append(k)
    matched = sorted(set.intersection(*[set(br[k]) for k in keys]) & set(base[keys[0]]))
    matched = [c for c in matched if c >= a.branch_point]
    keys = [k for k in keys if all(c in br[k] and c in base[k] for c in matched)]
    print(f"{len(keys)} capabilities, {len(matched)} matched checkpoints "
          f"{matched[0]}..{matched[-1]}")

    retention = [k for k in keys if ref[k]["tau_ck"] <= a.branch_point]
    plasticity = [k for k in keys if ref[k]["tau_ck"] > a.branch_point]
    print(f"retention set (already matured at ck{a.branch_point}): {len(retention)}")
    print(f"plasticity set (peaks later):                        {len(plasticity)}")

    def mnr(arm, ck, ks):
        return float(np.mean([max(0.0, ref[k]["S_star"] - arm[k][ck]) / ref[k]["sigma"] for k in ks]))

    def mx(arm, ck, ks):
        return float(np.max([max(0.0, ref[k]["S_star"] - arm[k][ck]) / ref[k]["sigma"] for k in ks]))

    rows = []
    for ck in matched:
        rows.append({"checkpoint": ck,
                     "vanilla_MNR": mnr(base, ck, keys), "branch_MNR": mnr(br, ck, keys),
                     "vanilla_maxNR": mx(base, ck, keys), "branch_maxNR": mx(br, ck, keys),
                     "vanilla_MNR_retention": mnr(base, ck, retention),
                     "branch_MNR_retention": mnr(br, ck, retention),
                     "vanilla_MNR_plasticity": mnr(base, ck, plasticity),
                     "branch_MNR_plasticity": mnr(br, ck, plasticity)})

    print(f"\n{'ck':>7s} | {'MNR van':>8s} {'MNR '+a.branch_name[:6]:>8s} {'delta':>7s} | "
          f"{'ret van':>8s} {'ret br':>8s} {'delta':>7s} | {'pla van':>8s} {'pla br':>8s} {'delta':>7s}")
    for r in rows:
        print(f"{r['checkpoint']:7d} | {r['vanilla_MNR']:8.3f} {r['branch_MNR']:8.3f} "
              f"{r['branch_MNR']-r['vanilla_MNR']:+7.3f} | "
              f"{r['vanilla_MNR_retention']:8.3f} {r['branch_MNR_retention']:8.3f} "
              f"{r['branch_MNR_retention']-r['vanilla_MNR_retention']:+7.3f} | "
              f"{r['vanilla_MNR_plasticity']:8.3f} {r['branch_MNR_plasticity']:8.3f} "
              f"{r['branch_MNR_plasticity']-r['vanilla_MNR_plasticity']:+7.3f}")

    end = rows[-1]
    verdict = ("RETENTION+PLASTICITY" if end["branch_MNR_retention"] < end["vanilla_MNR_retention"]
               and end["branch_MNR_plasticity"] <= end["vanilla_MNR_plasticity"] + 0.05
               else "RETENTION_ONLY" if end["branch_MNR_retention"] < end["vanilla_MNR_retention"]
               else "PLASTICITY_ONLY" if end["branch_MNR_plasticity"] < end["vanilla_MNR_plasticity"]
               else "NO_IMPROVEMENT")
    print(f"\nverdict at ck{end['checkpoint']}: {verdict}")

    # per-capability detail at the matched endpoint
    det = []
    for k in keys:
        s, sg = ref[k]["S_star"], ref[k]["sigma"]
        rv = max(0.0, s - base[k][end["checkpoint"]]) / sg
        rb = max(0.0, s - br[k][end["checkpoint"]]) / sg
        det.append({"metric_key": k, "set": "retention" if k in retention else "plasticity",
                    "tau_ck": ref[k]["tau_ck"], "class": ref[k]["class"],
                    "vanilla": base[k][end["checkpoint"]], "branch": br[k][end["checkpoint"]],
                    "r_vanilla": rv, "r_branch": rb, "delta_r": rb - rv})
    det.sort(key=lambda d: d["delta_r"])
    print(f"\nlargest regret reductions from {a.branch_name} at ck{end['checkpoint']}:")
    for d in det[:8]:
        print(f"  {d['metric_key'][:46]:46s} {d['set'][:4]:4s} r {d['r_vanilla']:6.2f} -> {d['r_branch']:6.2f}"
              f"  ({d['delta_r']:+.2f})")
    print("largest regret increases:")
    for d in det[-8:]:
        print(f"  {d['metric_key'][:46]:46s} {d['set'][:4]:4s} r {d['r_vanilla']:6.2f} -> {d['r_branch']:6.2f}"
              f"  ({d['delta_r']:+.2f})")

    a.out.write_text(json.dumps({"branch_name": a.branch_name, "branch_point": a.branch_point,
                                 "n_capabilities": len(keys), "retention_set": retention,
                                 "plasticity_set": plasticity, "curve": rows,
                                 "endpoint_detail": det, "verdict": verdict}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
