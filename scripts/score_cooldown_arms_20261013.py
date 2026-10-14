#!/usr/bin/env python3
"""Early scorer for the 2026-10-13 cooldown / distillation arms (non-seg families).

Cooldown results are the fleet-worker layout  results/point_<ck>/<lane>/bio_<family>/<ds>/<ck>/last_result.json
(mirrored JSON-only under /tmp/cooldown_scores/{hxw,lyx}).  References:
  * 5TB trunk: the same fleet worker's results for the no-GRAM continuation (hxw /data/hs6_l_5tb_nogram_eval_20260921),
    points 39527..41479, mirrored under /tmp/cooldown_scores/ref5tb.
  * 20TB trunk: paired_component_scores.csv (arm noGRAM20tb) from the route2 all-checkpoint v4 campaign, points <= 74663.
Metric per dataset follows the union-v4 convention (balanced_accuracy; chestmnist macro_auc; regression r2,
bbbc013 compound_mean_r2 when present; retrieval recall_at_1; clustering nmi from the retrieval lane).
Family means are over datasets present in BOTH the arm point and the reference; deltas are in percentage points.
"""
import csv, glob, json, os, statistics as st, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
S = Path("/tmp/cooldown_scores")
OUT = ROOT / "outputs/00_reports/scaling_plateau_cooldown_20261013"
CLS_METRIC = defaultdict(lambda: "balanced_accuracy", {"chestmnist": "macro_auc"})
REG_METRIC = defaultdict(lambda: "r2", {"bbbc013": "compound_mean_r2"})

def parse_fleet(root):
    """-> {ck: {(family, ds): value}}"""
    out = defaultdict(dict)
    for p in glob.glob(str(root / "point_*/*/bio_*/*/*/last_result.json")):
        parts = Path(p).parts
        ck = int(parts[-6].split("_")[1]); lane = parts[-5]; ds = parts[-3]
        d = json.load(open(p))
        if lane.startswith("classification"):
            m = CLS_METRIC[ds]
            if m in d: out[ck][("classification", ds)] = float(d[m])
        elif lane == "regression":
            m = REG_METRIC[ds] if REG_METRIC[ds] in d else "r2"
            if m in d: out[ck][("regression", ds)] = float(d[m])
        elif lane == "retrieval":
            if "recall_at_1" in d: out[ck][("retrieval", ds)] = float(d["recall_at_1"])
            if "nmi" in d: out[ck][("clustering", ds)] = float(d["nmi"])
    # segmentation lanes: results/point_<ck>/segmentation_*/bio_segmentation/<protocol>/<ds>/<ck>/results.json (test mDice;
    # the three PanNuke rotations are averaged), detection lane: .../bio_detection/<ds>/<ck>/results_bio_detection.json
    seg = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(str(root / "point_*/segmentation_*/bio_segmentation/*/*/*/results.json")):
        parts = Path(p).parts; ck = int(parts[-7].split("_")[1]); ds = parts[-3]
        try: seg[ck][ds].append(float(json.load(open(p))["test"]["mDice"]))
        except (KeyError, TypeError, ValueError): pass
    for ck, dss in seg.items():
        for ds, vals in dss.items(): out[ck][("segmentation", ds)] = st.mean(vals)
    for p in glob.glob(str(root / "point_*/detection/bio_detection/*/*/results_bio_detection.json")):
        parts = Path(p).parts; ck = int(parts[-6].split("_")[1]); ds = parts[-3]
        d = json.load(open(p))
        if "test_patch_f1" in d: out[ck][("detection", ds)] = float(d["test_patch_f1"]) / 100.0
    return out

def parse_20tb_csv():
    out = defaultdict(dict)
    f = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/20tb_route2_all_checkpoints_v4_20261012/paired_component_scores.csv"
    for r in csv.DictReader(open(f)):
        if r["arm"] != "noGRAM20tb": continue
        ck = int(r["checkpoint"]); fam, ds, m = r["family"], r["dataset"], r["metric"]
        try: v = float(r["value"])
        except ValueError: continue
        if fam == "classification" and m == CLS_METRIC[ds]: out[ck][("classification", ds)] = v
        elif fam == "regression" and m == "r2": out[ck][("regression", ds)] = v
        elif fam == "retrieval" and m == "recall_at_1": out[ck][("retrieval", ds)] = v
        elif fam == "retrieval" and m == "nmi": out[ck][("clustering", ds)] = v
    return out

def ref_mean(ref, cks):
    acc = defaultdict(list)
    for ck in cks:
        for k, v in ref.get(ck, {}).items(): acc[k].append(v)
    return {k: st.mean(v) for k, v in acc.items()}, {k: (st.pstdev(v) if len(v) > 1 else 0.0) for k, v in acc.items()}

def report(name, arm, refm, refsd, ref_cks, rows):
    print(f"\n=== {name}: cooldown points vs trunk reference mean of {ref_cks}")
    fams = ["classification", "regression", "retrieval", "clustering", "segmentation", "detection"]
    print("ckpt      " + "".join(f"{f[:12]:>22s}" for f in fams) + "   family-mean Δ")
    for ck in sorted(arm):
        cells = arm[ck]; line = f"{ck:<10d}"; fam_deltas = []
        for f in fams:
            common = [k for k in cells if k[0] == f and k in refm]
            if not common: line += f"{'—':>22s}"; continue
            d = 100 * st.mean(cells[k] - refm[k] for k in common)
            wins = sum(cells[k] > refm[k] for k in common)
            line += f"{d:+7.2f}pp ({wins:2d}/{len(common):2d} n)".rjust(22); fam_deltas.append(d)
            for k in common: rows.append(dict(arm=name, checkpoint=ck, family=f, dataset=k[1], value=cells[k], ref=refm[k], ref_sd=refsd.get(k, 0.0), delta_pp=100 * (cells[k] - refm[k])))
        line += f"   {st.mean(fam_deltas):+.2f}pp over {len(fam_deltas)} fam" if fam_deltas else ""
        print(line)
    # reference self-noise: sd across the reference points, averaged per family
    noise = {f: 100 * st.mean(sd for k, sd in refsd.items() if k[0] == f) for f in fams if any(k[0] == f for k in refsd)}
    print("reference point-to-point sd (pp): " + ", ".join(f"{f[:5]} {v:.2f}" for f, v in noise.items()))

def main():
    rows = []
    arm5 = parse_fleet(S / "hxw"); ref5 = parse_fleet(S / "ref5tb")
    cks5 = sorted(ref5); m5, sd5 = ref_mean(ref5, cks5)
    if arm5: report("5tb_cooldown", arm5, m5, sd5, cks5, rows)
    # 20TB reference: prefer the trunk points re-evaluated with the SAME fleet worker (local eval root
    # 20tb_trunk_ref, mirrored under S/ref20tb); fall back to the campaign CSV for cells it does not cover yet.
    ref20_csv = parse_20tb_csv(); cks_csv = [ck for ck in sorted(ref20_csv) if ck >= 72711]
    m_csv, sd_csv = ref_mean(ref20_csv, cks_csv)
    ref20_fleet = parse_fleet(S / "ref20tb"); cks_fleet = sorted(ref20_fleet)
    m_fleet, sd_fleet = ref_mean(ref20_fleet, cks_fleet)
    m20 = dict(m_csv); m20.update(m_fleet); sd20 = dict(sd_csv); sd20.update(sd_fleet)
    print(f"\n20TB reference: {len(m_fleet)} cells from same-worker trunk re-eval {cks_fleet}, "
          f"{len(m20) - len(m_fleet)} cells from campaign CSV {cks_csv}")
    cks20 = cks_fleet or cks_csv
    for name, sub in (("20tb_cooldown", "lyx"), ("20tb_distill_lw2 [local]", "local_distill_lw2"),
                      ("20tb_distill_lw5 [hxw]", "hxw_distill_lw5"), ("20tb_distill_lw2_rebal [hxw]", "hxw_distill_lw2_rebal"),
                      ("lyx_distill", "lyx_distill")):
        a = parse_fleet(S / sub)
        if a: report(name, a, m20, sd20, cks20, rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "early_scores_nonseg.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "checkpoint", "family", "dataset", "value", "ref", "ref_sd", "delta_pp"]); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT/'early_scores_nonseg.csv'} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
