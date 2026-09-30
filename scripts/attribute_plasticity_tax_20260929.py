#!/usr/bin/env python3
"""Attribute Adaptive's plasticity tax using existing paired-branch results and training logs. CPU only.

(A) Capability regret of the 2026-09-23 PAIRED branch from ck12687 (same start, same optimizer reset,
    2 ranks x 128 x acc4 = 1024, same stream): vanilla_formal / fixed_formal (gate==1) / adaptive_formal /
    gram_formal at 13175..15127, plus single-rank ck_c (frozen budget) / ck_k (bounded gain) at 12931..13663.
    Scored on the v4 reference rebuilt from the original no-GRAM (regret_v4_20260929). Retention/plasticity
    split by tau <= 12687. Segmentation (E50 in RESULTS.json vs E20 reference) is excluded.
(B) Recovery-controller logs: monitored error / floor / gate per optimizer update for adaptive_formal,
    fixed_formal, the long Adaptive continuation, and ck_c / ck_k. Quantifies whether the running-min floor
    is driven below the error by monitor noise (ratchet) relative to the 0.02 tolerance.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path("/mnt/huawei_deepcad/dinov3")
RES = json.load((REPO / "outputs/00_reports/deepcad_method_20260927/results_20260929/RESULTS.json").open())
REF = json.load((REPO / "outputs/00_reports/deepcad_method_20260927/regret_v4_20260929/reference_v4_from_union_noGRAM.json").open())
OUT = REPO / "outputs/00_reports/deepcad_method_20260927/plasticity_tax_20260929"; OUT.mkdir(parents=True, exist_ok=True)
BP = 12687
METRIC_MAP = {"bbbc005 R2": "regression:bbbc005:r2", "bbbc013 R2": "regression:bbbc013:r2",
              "conic-cell-count R2": "regression:conic-cell-count:r2", "livecell-cell-count R2": "regression:livecell-cell-count:r2",
              "hpa-subcellular R@1 (%)": ("retrieval:hpa-subcellular:recall_at_1", .01), "rxrx1-cross R@1 (%)": ("retrieval:rxrx1-cross:recall_at_1", .01),
              "bbbc038 patch F1 (%)": ("detection:bbbc038:test_patch_f1", .01)}

def arm_values(arm):
    m = RES["models"][arm]; out = {}
    for ds, v in m.get("classification", {}).items():
        k = f"classification:{ds}:{'macro_auc' if ds == 'chestmnist' else 'balanced_accuracy'}"
        if k in REF: out[k] = v / 100
    for name, v in m.get("metrics", {}).items():
        if name in METRIC_MAP:
            k, s = (METRIC_MAP[name], 1) if isinstance(METRIC_MAP[name], str) else METRIC_MAP[name]
            if k in REF: out[k] = v * s
    return out

def nr(k, v): return max(0.0, REF[k]["smoothed_best"] - v) / REF[k]["sigma"]

groups = {"paired_2rank": {"vanilla": "vanilla_formal_ck{}", "fixed(gate=1)": "fixed_formal_ck{}", "adaptive": "adaptive_formal_ck{}", "gram": "gram_formal_ck{}"},
          "single_rank_CK": {"C(frozen budget)": "ck_c_e12687_formal_ck{}", "K(bounded gain)": "ck_k_e12687_formal_ck{}"}}
report = {"reference": "regret_v4_20260929/reference_v4_from_union_noGRAM.json", "branch_point": BP, "groups": {}}
lines = []
for gname, arms in groups.items():
    cks = sorted({int(a.rsplit("_ck", 1)[1]) for a in RES["models"] if any(a.startswith(p.split("{}")[0]) for p in arms.values())})
    rows = []
    for ck in cks:
        vals = {lab: arm_values(p.format(ck)) for lab, p in arms.items() if p.format(ck) in RES["models"]}
        if len(vals) < 2: continue
        common = sorted(set.intersection(*[set(v) for v in vals.values()]))
        ret = [k for k in common if REF[k]["tau_ck"] <= BP]; pla = [k for k in common if REF[k]["tau_ck"] > BP]
        rec = {"checkpoint": ck, "n_common": len(common), "n_ret": len(ret), "n_pla": len(pla), "arms": {}}
        for lab, v in vals.items():
            rec["arms"][lab] = {"MNR": float(np.mean([nr(k, v[k]) for k in common])), "maxNR": float(np.max([nr(k, v[k]) for k in common])),
                                "MNR_ret": float(np.mean([nr(k, v[k]) for k in ret])) if ret else float("nan"),
                                "MNR_pla": float(np.mean([nr(k, v[k]) for k in pla])) if pla else float("nan"),
                                "per_key": {k: {"value": v[k], "r": nr(k, v[k])} for k in common}}
        rows.append(rec)
    report["groups"][gname] = rows
    lines.append(f"\n## {gname}  (common keys per checkpoint; retention = tau<=12687)\n")
    labs = list(arms.keys())
    lines.append("|ck|n (ret/pla)|" + "|".join(f"{l} MNR|{l} ret|{l} pla" for l in labs) + "|")
    lines.append("|---|---|" + "|".join("---:|---:|---:" for _ in labs) + "|")
    for r in rows:
        cells = []
        for l in labs:
            a = r["arms"].get(l)
            cells.append(f"{a['MNR']:.2f}|{a['MNR_ret']:.2f}|{a['MNR_pla']:.2f}" if a else "–|–|–")
        lines.append(f"|{r['checkpoint']}|{r['n_common']} ({r['n_ret']}/{r['n_pla']})|" + "|".join(cells) + "|")
# per-capability deltas vs vanilla at the last paired checkpoint
last = report["groups"]["paired_2rank"][-1]
van = last["arms"]["vanilla"]["per_key"]
lines.append(f"\n## paired branch ck{last['checkpoint']}: Δr vs vanilla (negative = arm retains more), raw delta in native units\n")
ARMS = [l for l in ("adaptive", "fixed(gate=1)", "gram") if l in last["arms"]]
lines.append("|capability|set|tau|vanilla|" + "|".join(f"{l}−vanilla" for l in ARMS) + "|" + "|".join(f"Δr {l}" for l in ARMS) + "|")
lines.append("|---|---|---:|---:|" + "|".join("---:" for _ in ARMS) + "|" + "|".join("---:" for _ in ARMS) + "|")
percap = []
for k in sorted(van, key=lambda k: last["arms"]["adaptive"]["per_key"][k]["r"] - van[k]["r"]):
    d = {"key": k, "set": "ret" if REF[k]["tau_ck"] <= BP else "pla", "tau": REF[k]["tau_ck"], "vanilla": van[k]["value"]}
    for l in ARMS:
        a = last["arms"][l]["per_key"][k]; d[f"{l}_raw"] = a["value"] - van[k]["value"]; d[f"{l}_dr"] = a["r"] - van[k]["r"]
    percap.append(d)
    u = 100 if not k.startswith("regression") else 1
    lines.append(f"|{k}|{d['set']}|{d['tau']}|{van[k]['value']*u:.2f}|" + "|".join(f"{d[f'{l}_raw']*u:+.2f}" for l in ARMS) + "|" + "|".join(f"{d[f'{l}_dr']:+.2f}" for l in ARMS) + "|")
report["per_capability_last_paired"] = percap

# ---- (B) controller logs -------------------------------------------------------------
LOGS = {"adaptive_formal (2 rank, 12688-15127)": REPO / "outputs/01_training_runs/hs6_l5_selective_retention_20260923/adaptive_formal/raw_loss_metrics.jsonl",
        "fixed_formal (gate=1)": REPO / "outputs/01_training_runs/hs6_l5_selective_retention_20260923/fixed_formal/raw_loss_metrics.jsonl",
        "adaptive_continue_v2_gpu2 (15127-16103)": REPO / "outputs/01_training_runs/hs6_l5_deepcad_method_20260927/adaptive_continue_v2_gpu2/raw_loss_metrics.jsonl",
        "adaptive_continue_resume16103 (16103-17689)": REPO / "outputs/01_training_runs/hs6_l5_deepcad_method_20260927/adaptive_continue_resume16103_gpu3_formal/raw_loss_metrics.jsonl",
        "ck_c frozen budget (12688-15127)": REPO / "outputs/01_training_runs/hs6_l5_deepcad_method_20260927/ck_c_e12687_formal/raw_loss_metrics.jsonl",
        "ck_k bounded gain (12688-15127)": REPO / "outputs/01_training_runs/hs6_l5_deepcad_method_20260927/ck_k_e12687_formal/raw_loss_metrics.jsonl"}
series = {}
for name, p in LOGS.items():
    if not p.exists(): continue
    recs = [json.loads(l) for l in p.open() if l.strip()]
    recs = [r for r in recs if "recovery_global_error" in r]
    recs.sort(key=lambda r: r["optimizer_update"])
    s = {"update": np.array([r["optimizer_update"] for r in recs])}
    for st in ("global", "local"):
        for f in ("error", "floor", "gate", "budget"):
            k = f"recovery_{st}_{f}"
            if k in recs[0]: s[f"{st}_{f}"] = np.array([r[k] for r in recs], dtype=float)
    series[name] = s
ctrl = {}
lines.append("\n## controller logs: is the floor driven below the error by noise? (global stream)\n")
lines.append("|run|updates|error first→last|floor first→last|gate first→last|update gate>0.9|noise σ of error (50-step resid)|tol|(error−floor) median|frac steps error>floor+tol|")
lines.append("|---|---:|---|---|---|---:|---:|---:|---:|---:|")
for name, s in series.items():
    if "global_error" not in s: continue
    e, u = s["global_error"], s["update"]; g = s.get("global_gate"); fl = s.get("global_floor", s.get("global_budget"))
    resid = e - np.array([np.median(e[max(0, i - 25): i + 25]) for i in range(len(e))])
    sig = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    cross = int(u[np.argmax(g > 0.9)]) if g is not None and (g > 0.9).any() else None
    gap = e - fl if fl is not None else None
    tol = 0.02 if "ck_c" not in name else float(np.median(s["global_budget"] - s["global_floor"])) if "global_budget" in s else 0.02
    ctrl[name] = {"n": int(len(e)), "error": [float(e[:50].mean()), float(e[-50:].mean())], "floor": [float(fl[:50].mean()), float(fl[-50:].mean())] if fl is not None else None,
                  "gate": [float(g[:50].mean()), float(g[-50:].mean())] if g is not None else None, "gate_gt_0.9_at": cross, "noise_sigma": sig,
                  "tolerance": tol, "gap_median": float(np.median(gap)) if gap is not None else None,
                  "frac_over_tol": float(np.mean(gap > tol)) if gap is not None else None}
    c = ctrl[name]
    lines.append(f"|{name}|{c['n']}|{c['error'][0]:.3f}→{c['error'][1]:.3f}|{c['floor'][0]:.3f}→{c['floor'][1]:.3f}|{c['gate'][0]:.2f}→{c['gate'][1]:.2f}|{c['gate_gt_0.9_at'] or '–'}|{sig:.4f}|{tol:.3f}|{c['gap_median']:.4f}|{c['frac_over_tol']:.2f}|")
report["controller"] = ctrl
# figure
fig, axes = plt.subplots(2, 3, figsize=(17, 8)); axes = axes.flatten()
for ax, (name, s) in zip(axes, series.items()):
    if "global_error" not in s: continue
    ax.plot(s["update"], s["global_error"], lw=.7, label="monitored error (global)")
    if "global_floor" in s: ax.plot(s["update"], s["global_floor"], lw=.9, label="floor (running min)" if "ck_c" not in name else "frozen reference")
    if "global_budget" in s: ax.plot(s["update"], s["global_budget"], lw=.9, ls="--", label="budget (ref+band)")
    ax2 = ax.twinx(); ax2.plot(s["update"], s["global_gate"], color="crimson", lw=1, label="gate"); ax2.set_ylim(0, 1.05); ax2.set_ylabel("gate", color="crimson")
    ax.set_title(name, fontsize=9); ax.set_xlabel("optimizer update"); ax.set_ylabel("normalised recovery error"); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper left")
fig.suptitle("Recovery controller: monitored error vs floor vs gate (global stream). Tolerance = 0.02 for Adaptive/K; C uses frozen reference + noise band.", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "CONTROLLER_LOGS.png", dpi=150)
(OUT / "report.json").write_text(json.dumps(report, indent=1))
(OUT / "TABLES.md").write_text("\n".join(lines))
print("\n".join(lines))
