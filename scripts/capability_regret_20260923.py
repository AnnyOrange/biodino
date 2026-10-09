#!/usr/bin/env python3
"""Capability-regret metric for asynchronous maturation during prolonged SSL.

Defines, for every downstream capability i measured along a reference training
trajectory S_i(t_1..t_T):

  smoothed curve      S~_i(t)   centered moving average, width w (odd)
  noise scale         sigma_i   1.4826 * MAD(S_i - S~_i)      (robust, curve-derived)
  reference best      S*_i      max_t S~_i(t)                 (smoothed best; the
                                frozen reference all later methods are scored on)
  maturation step     tau_i     first t with S~_i(t) >= S*_i - sigma_i
  regret              R_i(T)    S*_i - S_i(T)
  normalised regret   r_i(T)    max(0, R_i(T)) / sigma_i
  surplus             u_i(T)    max(0, -R_i(T)) / sigma_i

Aggregates (the objective every future method must optimise):
  MNR   = mean_i r_i        primary, minimise
  maxNR = max_i  r_i        worst-case guard, minimise
  MNS   = mean_i u_i        genuine new gain, report

MNR cannot be gamed by freezing an early checkpoint (late-growing capabilities
then carry large r_i) nor by plain continuation (early-matured capabilities do).
The oracle single-checkpoint floor min_t MNR(t) is the bar a real method beats.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/mnt/huawei_deepcad/dinov3")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectory", type=Path,
                   default=REPO / "outputs/00_reports/hs6_l5_capability_trajectory_20260923/trajectory_long.csv")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "outputs/00_reports/hs6_l5_capability_trajectory_20260923")
    p.add_argument("--window", type=int, default=5, help="smoothing width in checkpoints (odd)")
    p.add_argument("--min-coverage", type=int, default=60)
    p.add_argument("--exclude-families", default="ood",
                   help="comma-separated families kept out of the ID aggregate")
    return p.parse_args()


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average with shrinking window at the edges."""
    n = len(y)
    h = w // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        a, b = max(0, i - h), min(n, i + h + 1)
        out[i] = y[a:b].mean()
    return out


def mad_sigma(resid: np.ndarray) -> float:
    med = np.median(resid)
    return float(1.4826 * np.median(np.abs(resid - med)))


def analyse(metric: str, cks: list[int], y: np.ndarray, w: int) -> dict:
    ys = smooth(y, w)
    resid = y - ys
    sigma = mad_sigma(resid)
    if sigma <= 0 or not math.isfinite(sigma):
        # degenerate (constant / ceiling) metric: fall back to the smallest
        # non-zero absolute step seen on the curve, else mark unusable.
        steps = np.abs(np.diff(y))
        steps = steps[steps > 0]
        sigma = float(steps.min()) if steps.size else float("nan")
    star = float(ys.max())
    star_idx = int(ys.argmax())
    raw_star = float(y.max())
    win3 = np.array([y[max(0, i - 1): i + 2].mean() for i in range(len(y))])
    end = float(y[-1])
    R = star - end
    r = max(0.0, R) / sigma if math.isfinite(sigma) and sigma > 0 else float("nan")
    u = max(0.0, -R) / sigma if math.isfinite(sigma) and sigma > 0 else float("nan")
    thr = star - sigma
    tau_idx = int(np.argmax(ys >= thr)) if (ys >= thr).any() else len(ys) - 1
    start = float(ys[0])
    headroom = star - start
    growth_sig = (end - start) / sigma if sigma > 0 else float("nan")
    return dict(
        metric_key=metric,
        family=metric.split(":")[0],
        dataset=metric.split(":")[1] if metric.count(":") >= 2 else "",
        n_points=len(y),
        sigma=sigma,
        start_value=start,
        end_value=end,
        smoothed_best=star,
        smoothed_best_ck=cks[star_idx],
        raw_best=raw_star,
        raw_best_ck=cks[int(y.argmax())],
        window3_best=float(win3.max()),
        window3_best_ck=cks[int(win3.argmax())],
        regret=R,
        norm_regret=r,
        norm_surplus=u,
        headroom=headroom,
        frac_of_gain_lost=(R / headroom if headroom > 2 * sigma else float("nan")),
        tau_ck=cks[tau_idx],
        tau_frac=tau_idx / (len(cks) - 1),
        end_minus_start_sigma=growth_sig,
    )


def classify(row: dict) -> str:
    r, tf, g = row["norm_regret"], row["tau_frac"], row["end_minus_start_sigma"]
    if not math.isfinite(r):
        return "UNUSABLE_NO_NOISE_SCALE"
    eroded = r >= 2.0
    late = (g >= 2.0) and tf > 0.60
    early = tf <= 0.40
    if eroded and early:
        return "EARLY_MATURED_ERODED"
    if eroded:
        return "MID_MATURED_ERODED"
    if late:
        return "LATE_GROWING"
    if early:
        return "EARLY_MATURED_STABLE"
    return "GRADUAL_OR_FLAT"


def main() -> int:
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    excl = {x.strip() for x in a.exclude_families.split(",") if x.strip()}

    traj: dict[str, dict[int, float]] = defaultdict(dict)
    for rec in csv.DictReader(a.trajectory.open()):
        traj[rec["metric_key"]][int(rec["checkpoint"])] = float(rec["value"])

    rows, skipped = [], []
    for key, series in sorted(traj.items()):
        cks = sorted(series)
        if len(cks) < a.min_coverage:
            skipped.append({"metric_key": key, "n_points": len(cks), "reason": "PARTIAL_COVERAGE"})
            continue
        y = np.array([series[c] for c in cks], dtype=np.float64)
        rows.append(analyse(key, cks, y, a.window))
    for row in rows:
        row["capability_class"] = classify(row)

    id_rows = [r for r in rows if r["family"] not in excl and math.isfinite(r["norm_regret"])]
    dropped = [r["metric_key"] for r in rows if r["family"] not in excl and not math.isfinite(r["norm_regret"])]

    # ---- oracle single-checkpoint floor -------------------------------------
    keys = [r["metric_key"] for r in id_rows]
    all_cks = sorted(traj[keys[0]])
    star = {r["metric_key"]: r["smoothed_best"] for r in id_rows}
    sig = {r["metric_key"]: r["sigma"] for r in id_rows}
    per_ck = []
    for ck in all_cks:
        rr = [max(0.0, star[k] - traj[k][ck]) / sig[k] for k in keys]
        per_ck.append({"checkpoint": ck, "MNR": float(np.mean(rr)), "maxNR": float(np.max(rr)),
                       "medNR": float(np.median(rr))})
    floor = min(per_ck, key=lambda d: d["MNR"])
    endpoint = per_ck[-1]

    summary = {
        "reference_run": "HS6_L_robust_biosafe256_..._5tb_mix1m03_5tv107_8x5090zxr_20260907 (no-Gram)",
        "reference_curve_campaign": "outputs/02_eval_runs/hs6_l_5t_full_every_05m_3090qi_v2_fullregistry_20260908",
        "frozen_on": "2026-09-23",
        "smoothing_window_ckpts": a.window,
        "n_metrics_scored": len(id_rows),
        "excluded_families": sorted(excl),
        "excluded_partial_coverage": skipped,
        "dropped_no_noise_scale": dropped,
        "endpoint": endpoint,
        "oracle_single_checkpoint_floor": floor,
        "per_checkpoint": per_ck,
        "class_counts": {c: sum(1 for r in id_rows if r["capability_class"] == c)
                         for c in sorted({r["capability_class"] for r in id_rows})},
    }

    ref = {r["metric_key"]: {"S_star": r["smoothed_best"], "sigma": r["sigma"],
                             "star_ck": r["smoothed_best_ck"], "tau_ck": r["tau_ck"],
                             "class": r["capability_class"]} for r in rows}
    (a.out_dir / "capability_reference_frozen_20260923.json").write_text(json.dumps(ref, indent=1))
    (a.out_dir / "regret_summary.json").write_text(json.dumps(summary, indent=1))

    cols = list(rows[0].keys())
    with (a.out_dir / "capability_regret.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda d: -d["norm_regret"] if math.isfinite(d["norm_regret"]) else 1e9):
            w.writerow(r)

    print(f"scored {len(id_rows)} in-domain capabilities ({len(rows)} total, "
          f"{len(skipped)} partial-coverage skipped)")
    print(f"endpoint  ck{endpoint['checkpoint']}: MNR={endpoint['MNR']:.3f}  maxNR={endpoint['maxNR']:.2f}")
    print(f"oracle floor ck{floor['checkpoint']}: MNR={floor['MNR']:.3f}  maxNR={floor['maxNR']:.2f}")
    print("classes:", summary["class_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
