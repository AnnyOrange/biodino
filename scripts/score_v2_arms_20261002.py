#!/usr/bin/env python3
"""Score the Adaptive-v2 arms (global / global_local, FSDP fork of the original 5TB no-GRAM full state at 29279)
against the continuous no-GRAM continuation at the SAME optimizer steps (29767 .. 35135), under the union-v4 protocol.

Inputs (all CPU, no test-side selection):
  * union SELECTED_CELLS.csv            -> no-GRAM trajectory 487..41479 (reference + same-step baseline)
  * outputs/02_eval_runs/hs6_l5_v2_recovery_eval_20260930/<arm>/   (mirror of hxw /data/hs6_l5_v2_recovery_eval_20260930)
        results/point_<ck>/<lane>/bio_<family>/<ds>/<ck>/last_result.json      non-seg lanes (same fleet recipe as the baseline)
        v4/detection_b8/point_<ck>/<ds>/results_bio_detection.json             v4 detection proxy b8/224/5ep
        v3/cells/point_<ck>__<ds>__<split>/results/**/budget20/seed*/**/results.json   v3 segmentation cells (E20, 3 seeds)
        monuseg30/cells/v2_<arm>_ck<ck>__segmentation__monuseg__primary-last__formal-static-v1/results/**/budget20/seed*/**/results.json
                                                                                  MoNuSeg 30/7/14 (0929 amendment) site campaign
Outputs: outputs/00_reports/deepcad_method_20260927/v2_results_20261002/{REPORT.md, per_point_deltas.csv, summary.json, CURVES.png}

Scoring:
  1. same-step delta vs no-GRAM per capability (raw and in reference-sigma units), per family and per point;
  2. capability regret split on a reference rebuilt from the FULL no-GRAM run 487..41479 (capability_regret_20260923.analyse,
     window 5): retention set = tau <= 29279 (matured before the fork), plasticity set = tau > 29279.
"""
from __future__ import annotations
import csv, json, math, glob, collections, importlib.util, statistics
from pathlib import Path
import numpy as np

REPO = Path("/mnt/huawei_deepcad/dinov3")
UNION = REPO / "outputs/00_reports/deepcad_method_20260927/adaptive_original_union_20260929/SELECTED_CELLS.csv"
MIRROR = REPO / "outputs/02_eval_runs/hs6_l5_v2_recovery_eval_20260930"
OUT = REPO / "outputs/00_reports/deepcad_method_20260927/v2_results_20261002"
FORK, REF_LAST, WINDOW, MIN_COV = 29279, 41479, 5, 55
ARMS = ("global", "global_local")
POINTS = [29767 + 488 * i for i in range(12)]
PANNUKE_FOLDS = ("pannuke-fold1-train-fold2-val-fold3-test", "pannuke-fold2-train-fold1-val-fold3-test",
                 "pannuke-fold3-train-fold2-val-fold1-test")

spec = importlib.util.spec_from_file_location("cr", REPO / "scripts/capability_regret_20260923.py")
cr = importlib.util.module_from_spec(spec); spec.loader.exec_module(cr)

FAM = {"detection_proxy": "detection"}


def ukey(r):
    fam = FAM.get(r["family"], r["family"]); m = r["metric"]
    if fam == "segmentation": m = "mDice"
    if fam == "detection": m = "test_patch_f1"
    return f"{fam}:{r['dataset']}:{m}"


# ---------------- baseline (union no-GRAM) ----------------
base = collections.defaultdict(dict)   # key -> ck -> value
seen_scope = {}
for r in csv.DictReader(UNION.open()):
    if r["method"] != "no-GRAM": continue
    if r["family"] == "segmentation" and r["budget"] != "20": continue
    k, ck = ukey(r), int(r["checkpoint"])
    if (k, ck) in seen_scope and seen_scope[(k, ck)] == "matched": continue
    base[k][ck] = float(r["value"]); seen_scope[(k, ck)] = r["scope"]

# ---------------- v2 arms ----------------
def load_rows(p):
    o = json.loads(Path(p).read_text()); return o.get("rows", [o])


def seg_mean(cell_glob):
    vals = []
    for res in glob.glob(cell_glob, recursive=True):
        if "/budget20/" not in res: continue
        try: vals.append(float(json.loads(Path(res).read_text())["test"]["mDice"]))
        except Exception: pass
    return (statistics.mean(vals), len(vals)) if vals else (None, 0)


arms = {a: collections.defaultdict(dict) for a in ARMS}
coverage = {a: collections.Counter() for a in ARMS}
for a in ARMS:
    root = MIRROR / a
    for p in glob.glob(str(root / "results/point_*/*/bio_*/*/*/last_result.json")):
        parts = Path(p).parts; ck = int(parts[-2]); ds = parts[-3]; fam = parts[-4][4:]
        for row in load_rows(p):
            for k in list(base):
                kf, kd, km = k.split(":")
                if kd != ds: continue
                if kf == fam and row.get(km) is not None:
                    arms[a][k][ck] = float(row[km])
                elif kf == "clustering" and fam == "retrieval" and row.get(km) is not None:
                    arms[a][k][ck] = float(row[km])
    for p in glob.glob(str(root / "v4/detection_b8/point_*/*/results_bio_detection.json")):
        r = json.loads(Path(p).read_text())
        if r.get("test_patch_f1") is None or r.get("batch_size") != 8: continue
        v = float(r["test_patch_f1"]); v = v / 100.0 if v > 1.5 else v   # center_probe logs percent; the union stores fractions
        arms[a][f"detection:{r['dataset']}:test_patch_f1"][int(r["checkpoint"])] = v
    for cell in glob.glob(str(root / "v3/cells/point_*__*")):
        name = Path(cell).name; ck = int(name.split("__")[0][6:]); ds = name.split("__")[1]
        if not (Path(cell) / "validation_report.json").exists(): continue
        m, n = seg_mean(f"{cell}/results/**/results.json")
        if m is None: continue
        if ds == "pannuke":
            arms[a].setdefault("_pannuke_folds", collections.defaultdict(list))[ck].append(m)
        elif ds == "monuseg":
            arms[a]["segmentation:monuseg-official24:mDice"][ck] = m   # 24/6/14 official split, NOT the union key
        else:
            arms[a][f"segmentation:{ds}:mDice"][ck] = m
    folds = arms[a].pop("_pannuke_folds", {})
    for ck, vals in folds.items():
        if len(vals) == 3: arms[a]["segmentation:pannuke:mDice"][ck] = statistics.mean(vals)
    for cell in glob.glob(str(MIRROR / f"monuseg30_campaign_v2/cells/v2_{a}_ck*__segmentation__monuseg__*")):
        ck = int(Path(cell).name.split("__")[0].split("_ck")[1])
        m, n = seg_mean(f"{cell}/results/**/results.json")
        if m is not None and n == 3: arms[a]["segmentation:monuseg:mDice"][ck] = m
    for k in arms[a]:
        coverage[a][k.split(":")[0]] += len(arms[a][k])

# ---------------- reference from the full no-GRAM run ----------------
ref, skipped = {}, []
for k, series in sorted(base.items()):
    cks = sorted(c for c in series if c <= REF_LAST)
    if len(cks) < MIN_COV: skipped.append((k, len(cks))); continue
    row = cr.analyse(k, cks, np.array([series[c] for c in cks]), WINDOW)
    row["capability_class"] = cr.classify(row)
    if not (math.isfinite(row["sigma"]) and row["sigma"] > 0): skipped.append((k, len(cks), "no sigma")); continue
    ref[k] = row
retention = sorted(k for k in ref if ref[k]["tau_ck"] <= FORK)
plasticity = sorted(k for k in ref if ref[k]["tau_ck"] > FORK)


def nr(series, k, ck):
    return max(0.0, ref[k]["smoothed_best"] - series[k][ck]) / ref[k]["sigma"]


# ---------------- per-point deltas ----------------
rows = []
for ck in POINTS:
    for a in ARMS:
        for k in sorted(base):
            if ck not in base[k] or ck not in arms[a].get(k, {}): continue
            d = arms[a][k][ck] - base[k][ck]
            sig = ref[k]["sigma"] if k in ref else float("nan")
            rows.append({"checkpoint": ck, "arm": a, "key": k, "family": k.split(":")[0], "dataset": k.split(":")[1],
                         "noGRAM": base[k][ck], "v2": arms[a][k][ck], "delta": d, "delta_pp": 100 * d,
                         "delta_sigma": d / sig if sig and sig > 0 else float("nan"),
                         "set": ("retention" if k in retention else "plasticity" if k in plasticity else "n/a"),
                         "class": ref[k]["capability_class"] if k in ref else "n/a"})
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "per_point_deltas.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def summarize(sel):
    if not sel: return None
    d = [r["delta_pp"] for r in sel]; s = [r["delta_sigma"] for r in sel if math.isfinite(r["delta_sigma"])]
    return {"n": len(sel), "mean_pp": statistics.mean(d), "median_pp": statistics.median(d),
            "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d),
            "mean_sigma": statistics.mean(s) if s else float("nan"),
            "n_gt_1sigma": sum(x > 1 for x in s), "n_lt_m1sigma": sum(x < -1 for x in s)}


families = sorted({r["family"] for r in rows})
summary = {"per_arm_point": {}, "per_arm_family_allpoints": {}, "per_arm_set_allpoints": {}, "endpoint": {}}
for a in ARMS:
    for ck in POINTS:
        sel = [r for r in rows if r["arm"] == a and r["checkpoint"] == ck]
        if sel: summary["per_arm_point"][f"{a}@{ck}"] = summarize(sel)
    for f in families:
        summary["per_arm_family_allpoints"][f"{a}:{f}"] = summarize([r for r in rows if r["arm"] == a and r["family"] == f])
    for s in ("retention", "plasticity"):
        summary["per_arm_set_allpoints"][f"{a}:{s}"] = summarize([r for r in rows if r["arm"] == a and r["set"] == s])

# ---------------- regret curves on common keys ----------------
curve = []
for ck in POINTS:
    keys = [k for k in ref if ck in base[k] and all(ck in arms[a].get(k, {}) for a in ARMS)]
    if not keys: continue
    rec = {"checkpoint": ck, "n_keys": len(keys), "n_ret": sum(k in retention for k in keys), "n_pla": sum(k in plasticity for k in keys)}
    for name, series in (("noGRAM", base),) + tuple((a, arms[a]) for a in ARMS):
        r_all = [nr(series, k, ck) for k in keys]
        rec[f"{name}_MNR"] = float(np.mean(r_all)); rec[f"{name}_maxNR"] = float(np.max(r_all))
        for s, ks in (("ret", retention), ("pla", plasticity)):
            v = [nr(series, k, ck) for k in keys if k in ks]
            rec[f"{name}_MNR_{s}"] = float(np.mean(v)) if v else float("nan")
    curve.append(rec)

SIGMA_FLOOR = 0.002   # keys whose reference noise is below 0.2pp are ceiling/degenerate and dominate MNR
robust_keys = sorted(k for k in ref if ref[k]["sigma"] >= SIGMA_FLOOR)
curve_robust = []
for ck in [28303, 28791, 29279] + POINTS:
    keys = [k for k in robust_keys if ck in base[k] and (ck <= FORK or all(ck in arms[a].get(k, {}) for a in ARMS))]
    if not keys: continue
    rec = {"checkpoint": ck, "n_keys": len(keys), "n_ret": sum(k in retention for k in keys), "n_pla": sum(k in plasticity for k in keys)}
    series_list = (("noGRAM", base),) + (tuple((a, arms[a]) for a in ARMS) if ck > FORK else ())
    for name, series in series_list:
        r_all = [nr(series, k, ck) for k in keys]
        rec[f"{name}_MNR"] = float(np.mean(r_all)); rec[f"{name}_medNR"] = float(np.median(r_all))
        for s_, ks in (("ret", retention), ("pla", plasticity)):
            v = [nr(series, k, ck) for k in keys if k in ks]
            rec[f"{name}_MNR_{s_}"] = float(np.mean(v)) if v else float("nan")
    curve_robust.append(rec)

# endpoint per-capability table (last point with full coverage)
last = max((r["checkpoint"] for r in curve), default=None)
percap = []
if last is not None:
    for k in sorted(ref):
        if last not in base[k] or any(last not in arms[a].get(k, {}) for a in ARMS): continue
        rec = {"key": k, "set": "retention" if k in retention else "plasticity", "class": ref[k]["capability_class"],
               "tau_ck": ref[k]["tau_ck"], "S_star": ref[k]["smoothed_best"], "sigma": ref[k]["sigma"],
               "noGRAM": base[k][last], "r_noGRAM": nr(base, k, last)}
        for a in ARMS:
            rec[a] = arms[a][k][last]; rec[f"r_{a}"] = nr(arms[a], k, last)
            rec[f"d_{a}_sigma"] = (arms[a][k][last] - base[k][last]) / ref[k]["sigma"]
        percap.append(rec)
    percap.sort(key=lambda d: d["d_global_local_sigma"])
summary["endpoint"] = {"checkpoint": last, "rows": percap}
summary["reference"] = {"span": [487, REF_LAST], "n": len(ref), "retention": len(retention), "plasticity": len(plasticity),
                        "skipped": skipped, "class_counts": dict(collections.Counter(ref[k]["capability_class"] for k in ref))}
summary["coverage"] = {a: dict(coverage[a]) for a in ARMS}
summary["curve"] = curve
summary["curve_robust"] = {"sigma_floor": SIGMA_FLOOR, "n_keys": len(robust_keys), "rows": curve_robust}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

# ---------------- figure ----------------
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cols = {"noGRAM": "#555555", "global": "#1f77b4", "global_local": "#d62728"}
    for ax, what, title in ((axes[0, 0], "MNR", "MNR (all common keys)"), (axes[0, 1], "MNR_ret", "MNR retention (tau<=29279)"),
                            (axes[1, 0], "MNR_pla", "MNR plasticity (tau>29279)")):
        for name in cols:
            ax.plot([c["checkpoint"] for c in curve], [c[f"{name}_{what}"] for c in curve], "-o", ms=4, color=cols[name], label=name)
        ax.set_title(title); ax.set_xlabel("optimizer update"); ax.grid(alpha=.3); ax.legend()
    ax = axes[1, 1]
    for a in ARMS:
        for f in families:
            ys = []
            for ck in POINTS:
                sel = [r["delta_pp"] for r in rows if r["arm"] == a and r["checkpoint"] == ck and r["family"] == f]
                ys.append(statistics.mean(sel) if sel else float("nan"))
            ax.plot(POINTS, ys, "-" if a == "global" else "--", label=f"{a}:{f}")
    ax.axhline(0, color="k", lw=.8); ax.set_title("mean same-step delta vs no-GRAM (pp) per family"); ax.set_xlabel("optimizer update"); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "CURVES.png", dpi=130)
except Exception as e:
    print("figure skipped:", e)

# ---------------- report ----------------
L = []
L.append("# Adaptive-v2 (fixed-weight anchor recoverability, restart-free fork @29279) vs continuous no-GRAM — union-v4 protocol\n")
L.append(f"Arms: `global` (CLS+patch-mean stream only) and `global_local` (+16-patch local stream), both FSDP 4x64xacc4 from the original 5TB "
         f"no-GRAM full state at 29279 (in-run anchor = EMA teacher@29279). Baseline = the lyx continuation of the same run evaluated at the same steps. "
         f"Non-seg lanes use the identical fleet recipe/env/host as the baseline numbers; dense cells use the same pinned snapshots.\n")
L.append(f"Reference for regret: no-GRAM 487..{REF_LAST} (window {WINDOW}); {len(ref)} capabilities, retention (tau<={FORK}) {len(retention)}, plasticity {len(plasticity)}; "
         f"classes {summary['reference']['class_counts']}; skipped {skipped}.\n")
L.append("Coverage (capability-points per family): " + "; ".join(f"{a}: {dict(coverage[a])}" for a in ARMS) + "\n")
L.append("## Same-step delta vs no-GRAM, all evaluated points pooled (pp = percentage points of the metric; sigma = reference noise)\n")
L.append("| arm | family | n | mean pp | median pp | wins/losses | mean Δ/σ | >+1σ | <−1σ |\n|---|---|---|---|---|---|---|---|---|")
for a in ARMS:
    for f in families:
        s = summary["per_arm_family_allpoints"][f"{a}:{f}"]
        if s: L.append(f"| {a} | {f} | {s['n']} | {s['mean_pp']:+.2f} | {s['median_pp']:+.2f} | {s['wins']}/{s['losses']} | {s['mean_sigma']:+.2f} | {s['n_gt_1sigma']} | {s['n_lt_m1sigma']} |")
    for st in ("retention", "plasticity"):
        s = summary["per_arm_set_allpoints"][f"{a}:{st}"]
        if s: L.append(f"| {a} | **{st} set** | {s['n']} | {s['mean_pp']:+.2f} | {s['median_pp']:+.2f} | {s['wins']}/{s['losses']} | {s['mean_sigma']:+.2f} | {s['n_gt_1sigma']} | {s['n_lt_m1sigma']} |")
L.append("\n## Per point (all keys available at that point)\n")
L.append("| ck | arm | n | mean pp | wins/losses | mean Δ/σ |\n|---|---|---|---|---|---|")
for ck in POINTS:
    for a in ARMS:
        s = summary["per_arm_point"].get(f"{a}@{ck}")
        if s: L.append(f"| {ck} | {a} | {s['n']} | {s['mean_pp']:+.2f} | {s['wins']}/{s['losses']} | {s['mean_sigma']:+.2f} |")
L.append("\n## Capability regret (common keys at each point; lower is better)\n")
L.append("| ck | n (ret/pla) | MNR noGRAM | MNR global | MNR global_local | ret noGRAM | ret global | ret gl | pla noGRAM | pla global | pla gl |\n|---|---|---|---|---|---|---|---|---|---|---|")
for c in curve:
    L.append(f"| {c['checkpoint']} | {c['n_keys']} ({c['n_ret']}/{c['n_pla']}) | {c['noGRAM_MNR']:.3f} | {c['global_MNR']:.3f} | {c['global_local_MNR']:.3f} | "
             f"{c['noGRAM_MNR_ret']:.3f} | {c['global_MNR_ret']:.3f} | {c['global_local_MNR_ret']:.3f} | {c['noGRAM_MNR_pla']:.3f} | {c['global_MNR_pla']:.3f} | {c['global_local_MNR_pla']:.3f} |")
L.append(f"\n## Capability regret, robust key set (reference sigma >= {SIGMA_FLOOR:.3f}; {len(robust_keys)} keys; pre-fork rows give the no-GRAM context)\n")
L.append("| ck | n (ret/pla) | MNR noGRAM | MNR global | MNR global_local | medNR noGRAM | medNR global | medNR gl | ret noGRAM/global/gl | pla noGRAM/global/gl |\n|---|---|---|---|---|---|---|---|---|---|")
for c in curve_robust:
    g = lambda name, f: (f"{c[f'{name}_{f}']:.3f}" if f"{name}_{f}" in c and math.isfinite(c[f"{name}_{f}"]) else "–")
    L.append(f"| {c['checkpoint']} | {c['n_keys']} ({c['n_ret']}/{c['n_pla']}) | {g('noGRAM','MNR')} | {g('global','MNR')} | {g('global_local','MNR')} | {g('noGRAM','medNR')} | {g('global','medNR')} | {g('global_local','medNR')} | "
             f"{g('noGRAM','MNR_ret')}/{g('global','MNR_ret')}/{g('global_local','MNR_ret')} | {g('noGRAM','MNR_pla')}/{g('global','MNR_pla')}/{g('global_local','MNR_pla')} |")
if percap:
    L.append(f"\n## Endpoint {last}: per capability (sorted by global_local delta in sigma units)\n")
    L.append("| key | set | class | noGRAM | global | Δ/σ | global_local | Δ/σ |\n|---|---|---|---|---|---|---|---|")
    for r in percap:
        L.append(f"| {r['key']} | {r['set'][:3]} | {r['class']} | {r['noGRAM']:.4f} | {r['global']:.4f} | {r['d_global_sigma']:+.2f} | {r['global_local']:.4f} | {r['d_global_local_sigma']:+.2f} |")
L.append("\nNotes: `segmentation:monuseg:mDice` is the 30/7/14 (0929 amendment) split from the hxw site campaign; the official 24/6/14 v3 cell is reported separately as "
         "`segmentation:monuseg-official24:mDice` and is not in the union reference. rxrx3-core (retrieval/clustering) is not evaluated for v2 yet. "
         "Streams are not sample-paired with the baseline (loader RNG not in the checkpoint).")
(OUT / "REPORT.md").write_text("\n".join(L) + "\n")
print(f"rows {len(rows)}; ref {len(ref)} (ret {len(retention)} / pla {len(plasticity)}); coverage {dict((a, dict(coverage[a])) for a in ARMS)}")
for a in ARMS:
    for f in families:
        s = summary["per_arm_family_allpoints"][f"{a}:{f}"]
        if s: print(f"{a:13s} {f:14s} n={s['n']:3d} mean {s['mean_pp']:+.2f}pp  wins/losses {s['wins']}/{s['losses']}  mean d/sigma {s['mean_sigma']:+.2f}")
    for st in ("retention", "plasticity"):
        s = summary["per_arm_set_allpoints"][f"{a}:{st}"]
        if s: print(f"{a:13s} {st:14s} n={s['n']:3d} mean {s['mean_pp']:+.2f}pp  wins/losses {s['wins']}/{s['losses']}  mean d/sigma {s['mean_sigma']:+.2f}")
for c in curve_robust:
    g = lambda name, f: (f"{c[f'{name}_{f}']:.3f}" if f"{name}_{f}" in c else "–")
    print(f"robust ck {c['checkpoint']} n={c['n_keys']} MNR {g('noGRAM','MNR')}/{g('global','MNR')}/{g('global_local','MNR')} medNR {g('noGRAM','medNR')}/{g('global','medNR')}/{g('global_local','medNR')} ret {g('noGRAM','MNR_ret')}/{g('global','MNR_ret')}/{g('global_local','MNR_ret')} pla {g('noGRAM','MNR_pla')}/{g('global','MNR_pla')}/{g('global_local','MNR_pla')}")
for c in curve:
    print(f"ck {c['checkpoint']} n={c['n_keys']} MNR noGRAM {c['noGRAM_MNR']:.3f} global {c['global_MNR']:.3f} gl {c['global_local_MNR']:.3f} | ret {c['noGRAM_MNR_ret']:.3f}/{c['global_MNR_ret']:.3f}/{c['global_local_MNR_ret']:.3f} | pla {c['noGRAM_MNR_pla']:.3f}/{c['global_MNR_pla']:.3f}/{c['global_local_MNR_pla']:.3f}")
print("wrote", OUT)
