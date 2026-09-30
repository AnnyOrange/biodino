#!/usr/bin/env python3
"""Capability regret of Adaptive vs original 5TB no-GRAM / GRAM under ONE (v4) protocol.

The frozen 20260923 reference was built on the fullregistry_20260908 campaign, whose
segmentation/detection (and, after the sklearn rerun, classification) values differ from
the v4 cells that Adaptive was evaluated on. Scoring v4 arms against it would read protocol
differences as regret. So the reference (S*_i, sigma_i, tau_i) is rebuilt here from the
matched-scope v4 no-GRAM trajectory 487..29279 in SELECTED_CELLS.csv with the identical
formula (capability_regret_20260923.analyse, window=5), and all three arms are scored on it.
CPU only. No test-side selection: checkpoints are every registered Adaptive checkpoint.
"""
from __future__ import annotations
import csv, json, math, importlib.util, collections
from pathlib import Path
import numpy as np

REPO = Path("/mnt/huawei_deepcad/dinov3")
UNION = REPO / "outputs/00_reports/deepcad_method_20260927/adaptive_original_union_20260929/SELECTED_CELLS.csv"
OUT = REPO / "outputs/00_reports/deepcad_method_20260927/regret_v4_20260929"
BRANCH_POINT, REF_LAST, WINDOW, MIN_COV = 12687, 29279, 5, 55

spec = importlib.util.spec_from_file_location("cr", REPO / "scripts/capability_regret_20260923.py")
cr = importlib.util.module_from_spec(spec); spec.loader.exec_module(cr)

FAM = {"detection_proxy": "detection"}
def key(r):
    fam = FAM.get(r["family"], r["family"]); m = r["metric"]
    if fam == "segmentation": m = "mDice"          # E20 for every arm (budget column == 20)
    if fam == "detection": m = "test_patch_f1"
    return f"{fam}:{r['dataset']}:{m}"

arms = collections.defaultdict(lambda: collections.defaultdict(dict))  # arm -> key -> ck -> value
for r in csv.DictReader(UNION.open()):
    if r["scope"] != "matched": continue
    if r["family"] == "segmentation" and r["budget"] != "20": continue
    if r["metric"] == "compound_mean_r2": continue
    arms[r["method"]][key(r)][int(r["checkpoint"])] = float(r["value"])

# ---- reference from v4 no-GRAM 487..29279 -------------------------------------
ref, skipped = {}, []
for k, series in sorted(arms["no-GRAM"].items()):
    cks = sorted(c for c in series if c <= REF_LAST)
    if len(cks) < MIN_COV:
        skipped.append({"metric_key": k, "n": len(cks)}); continue
    row = cr.analyse(k, cks, np.array([series[c] for c in cks]), WINDOW)
    row["capability_class"] = cr.classify(row)
    if not (math.isfinite(row["sigma"]) and row["sigma"] > 0): skipped.append({"metric_key": k, "n": len(cks), "reason": "no sigma"}); continue
    ref[k] = row
retention = sorted(k for k in ref if ref[k]["tau_ck"] <= BRANCH_POINT)
plasticity = sorted(k for k in ref if ref[k]["tau_ck"] > BRANCH_POINT)

def nr(arm, k, ck):
    return max(0.0, ref[k]["smoothed_best"] - arms[arm][k][ck]) / ref[k]["sigma"]

ad_cks = sorted({c for k in arms["Adaptive"] for c in arms["Adaptive"][k]})
curve = []
for ck in ad_cks:
    keys = [k for k in ref if all(ck in arms[a].get(k, {}) for a in ("no-GRAM", "Adaptive"))]
    gram_keys = [k for k in keys if ck in arms["GRAM"].get(k, {})]
    rec = {"checkpoint": ck, "n_keys": len(keys), "n_keys_with_gram": len(gram_keys),
           "n_retention": sum(k in retention for k in keys), "n_plasticity": sum(k in plasticity for k in keys)}
    for arm, ks in (("no-GRAM", keys), ("Adaptive", keys), ("GRAM", gram_keys)):
        if not ks: continue
        r_all = [nr(arm, k, ck) for k in ks]
        r_ret = [nr(arm, k, ck) for k in ks if k in retention]
        r_pla = [nr(arm, k, ck) for k in ks if k in plasticity]
        rec[f"{arm}_MNR"] = float(np.mean(r_all)); rec[f"{arm}_maxNR"] = float(np.max(r_all))
        rec[f"{arm}_MNR_retention"] = float(np.mean(r_ret)) if r_ret else float("nan")
        rec[f"{arm}_MNR_plasticity"] = float(np.mean(r_pla)) if r_pla else float("nan")
    # GRAM on its own key subset is not comparable to no-GRAM on the full set; also give no-GRAM on gram subset
    if gram_keys:
        rec["no-GRAM_MNR_on_gram_keys"] = float(np.mean([nr("no-GRAM", k, ck) for k in gram_keys]))
        rec["Adaptive_MNR_on_gram_keys"] = float(np.mean([nr("Adaptive", k, ck) for k in gram_keys]))
    curve.append(rec)

# fixed common key set across the fully-evaluated Adaptive checkpoints, for a like-for-like curve
full_cks = [r["checkpoint"] for r in curve if r["n_keys"] >= 0.9 * max(x["n_keys"] for x in curve)]
common = sorted(k for k in ref if all(ck in arms[a].get(k, {}) for a in ("no-GRAM", "Adaptive") for ck in full_cks))
fixed = []
for ck in full_cks:
    rec = {"checkpoint": ck}
    for arm in ("no-GRAM", "Adaptive"):
        rec[f"{arm}_MNR"] = float(np.mean([nr(arm, k, ck) for k in common]))
        rec[f"{arm}_MNR_retention"] = float(np.mean([nr(arm, k, ck) for k in common if k in retention]))
        rec[f"{arm}_MNR_plasticity"] = float(np.mean([nr(arm, k, ck) for k in common if k in plasticity]))
        rec[f"{arm}_maxNR"] = float(np.max([nr(arm, k, ck) for k in common]))
    fixed.append(rec)

# per-capability at the last full checkpoint
last = full_cks[-1]
percap = []
for k in common:
    rv, ra = nr("no-GRAM", k, last), nr("Adaptive", k, last)
    percap.append({"metric_key": k, "set": "retention" if k in retention else "plasticity",
                   "class": ref[k]["capability_class"], "tau_ck": ref[k]["tau_ck"], "star_ck": ref[k]["smoothed_best_ck"],
                   "S_star": ref[k]["smoothed_best"], "sigma": ref[k]["sigma"],
                   "no-GRAM": arms["no-GRAM"][k][last], "Adaptive": arms["Adaptive"][k][last],
                   "r_no-GRAM": rv, "r_Adaptive": ra, "delta_r": ra - rv,
                   "delta_raw_sigma": (arms["Adaptive"][k][last] - arms["no-GRAM"][k][last]) / ref[k]["sigma"]})
percap.sort(key=lambda d: d["delta_r"])

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "reference_v4_from_union_noGRAM.json").write_text(json.dumps(
    {k: {f: v[f] for f in ("smoothed_best", "smoothed_best_ck", "sigma", "tau_ck", "tau_frac", "capability_class", "norm_regret")} for k, v in ref.items()}, indent=1))
(OUT / "regret_curve.json").write_text(json.dumps({"branch_point": BRANCH_POINT, "reference_span": [487, REF_LAST], "window": WINDOW,
    "n_reference_capabilities": len(ref), "skipped": skipped, "retention_set": retention, "plasticity_set": plasticity,
    "per_checkpoint_available_keys": curve, "fixed_common_keys": common, "fixed_common_curve": fixed,
    "per_capability_at_last_full": {"checkpoint": last, "rows": percap}}, indent=1))
with (OUT / f"per_capability_ck{last}.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(percap[0].keys())); w.writeheader(); w.writerows(percap)

print(f"reference: {len(ref)} capabilities from v4 no-GRAM 487..{REF_LAST}; skipped {[(s['metric_key'], s['n']) for s in skipped]}")
print(f"retention (tau<=12687): {len(retention)}   plasticity: {len(plasticity)}")
print("class counts:", collections.Counter(ref[k]["capability_class"] for k in ref))
print(f"\nfixed common key set: {len(common)} keys ({sum(k in retention for k in common)} ret / {sum(k in plasticity for k in common)} pla) over ck {full_cks}")
print(f"{'ck':>6s} | {'MNR nG':>7s} {'MNR Ad':>7s} {'d':>6s} | {'ret nG':>7s} {'ret Ad':>7s} {'d':>6s} | {'pla nG':>7s} {'pla Ad':>7s} {'d':>6s} | {'max nG':>6s} {'max Ad':>6s}")
for r in fixed:
    print(f"{r['checkpoint']:6d} | {r['no-GRAM_MNR']:7.3f} {r['Adaptive_MNR']:7.3f} {r['Adaptive_MNR']-r['no-GRAM_MNR']:+6.3f} | "
          f"{r['no-GRAM_MNR_retention']:7.3f} {r['Adaptive_MNR_retention']:7.3f} {r['Adaptive_MNR_retention']-r['no-GRAM_MNR_retention']:+6.3f} | "
          f"{r['no-GRAM_MNR_plasticity']:7.3f} {r['Adaptive_MNR_plasticity']:7.3f} {r['Adaptive_MNR_plasticity']-r['no-GRAM_MNR_plasticity']:+6.3f} | "
          f"{r['no-GRAM_maxNR']:6.2f} {r['Adaptive_maxNR']:6.2f}")
print("\nper-checkpoint on all available keys (incl. GRAM where present):")
print(f"{'ck':>6s} {'n':>3s} | {'nG':>6s} {'Ad':>6s} {'GRAM':>6s} (GRAM on {'nGk':>3s} keys: nG {'':>5s} Ad {'':>5s}) | ret nG/Ad | pla nG/Ad")
for r in curve:
    g = f"{r['GRAM_MNR']:6.3f}" if "GRAM_MNR" in r else "   -  "
    gsub = f"{r['n_keys_with_gram']:3d}      {r.get('no-GRAM_MNR_on_gram_keys', float('nan')):6.3f}   {r.get('Adaptive_MNR_on_gram_keys', float('nan')):6.3f}" if "GRAM_MNR" in r else ""
    print(f"{r['checkpoint']:6d} {r['n_keys']:3d} | {r['no-GRAM_MNR']:6.3f} {r['Adaptive_MNR']:6.3f} {g} ({gsub}) | "
          f"{r['no-GRAM_MNR_retention']:5.2f}/{r['Adaptive_MNR_retention']:5.2f} | {r['no-GRAM_MNR_plasticity']:5.2f}/{r['Adaptive_MNR_plasticity']:5.2f}")
print(f"\nper-capability at ck{last} (delta_r = r_Adaptive - r_noGRAM; negative = Adaptive retains more):")
for d in percap:
    print(f"  {d['metric_key'][:46]:46s} {d['set'][:3]:3s} {d['class'][:22]:22s} tau{d['tau_ck']:6d} r {d['r_no-GRAM']:6.2f} -> {d['r_Adaptive']:6.2f}  d{d['delta_r']:+6.2f}")
