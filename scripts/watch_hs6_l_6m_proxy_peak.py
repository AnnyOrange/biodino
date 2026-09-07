#!/usr/bin/env python3
"""Rank completed Proxy-9 points and preserve a sustained-decline peak teacher."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_symlink(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, path)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def update_curve(repo: Path, eval_root: Path, state_root: Path) -> list[dict[str, str]]:
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/compare_splus_objective_proxy.py"),
            "--run",
            f"trajectory={eval_root}",
            "--output-dir",
            str(state_root / "ranking"),
            "--expected-datasets",
            "9",
        ],
        check=True,
        cwd=repo,
        stdout=subprocess.DEVNULL,
    )
    return [
        row
        for row in load_csv(state_root / "ranking" / "summary.csv")
        if row["complete"] == "1"
    ]


def decline_evidence(
    details: list[dict[str, str]], best_checkpoint: int, later_checkpoints: list[int]
) -> list[dict[str, int]]:
    values = {
        (int(row["checkpoint"]), row["task"], row["dataset"]): float(row["value"])
        for row in details
    }
    best = {
        (task, dataset): value
        for (checkpoint, task, dataset), value in values.items()
        if checkpoint == best_checkpoint
    }
    evidence = []
    for checkpoint in later_checkpoints:
        wins = ties = losses = 0
        for key, best_value in best.items():
            value = values.get((checkpoint, *key))
            if value is None:
                continue
            if value > best_value:
                wins += 1
            elif value < best_value:
                losses += 1
            else:
                ties += 1
        evidence.append(
            {"checkpoint": checkpoint, "wins_vs_peak": wins, "ties_vs_peak": ties, "losses_vs_peak": losses}
        )
    return evidence


def parse_args() -> argparse.Namespace:
    repo = Path("/mnt/huawei_deepcad/dinov3")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--train-root",
        type=Path,
        default=repo
        / "outputs/01_training_runs/HS6_L_robust_biosafe256_gb1024_lr1e4_wu3_tw30_nosig_e15_6m_mix1m03_10tv107_8x5090zxr_20260826",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=repo / "outputs/02_eval_runs/hs6_l_6m_mix_proxy1m_3090qi_20260826",
    )
    parser.add_argument("--expected-points", type=int, default=90)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--decline-window", type=int, default=3)
    parser.add_argument("--losses-required", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.train_root = args.train_root.resolve()
    args.eval_root = args.eval_root.resolve()
    state_root = args.eval_root / "_peak_monitor"
    state_root.mkdir(parents=True, exist_ok=True)
    previous_done = -1

    while True:
        done = len(list((args.eval_root / "_online_status").glob("ckpt_*.done")))
        failed = len(list((args.eval_root / "_online_status").glob("ckpt_*.failed")))
        if done != previous_done and done > 0:
            summaries = update_curve(args.repo, args.eval_root, state_root)
            if summaries:
                summaries.sort(key=lambda row: (float(row["mean_rank"]), -int(row["wins"]), int(row["checkpoint"])))
                best = summaries[0]
                best_checkpoint = int(best["checkpoint"])
                source = args.train_root / f"eval/training_{best_checkpoint}/teacher_checkpoint.pth"
                payload: dict[str, object] = {
                    "checkpoint": best_checkpoint,
                    "complete_proxy_points": len(summaries),
                    "image_visits": (best_checkpoint + 1) * 1024,
                    "mean_rank": float(best["mean_rank"]),
                    "source": str(source),
                    "updated_at_utc": utc_now(),
                    "wins": int(best["wins"]),
                }
                atomic_json(state_root / "provisional_peak.json", payload)
                atomic_symlink(state_root / "provisional_peak_teacher_checkpoint.pth", source)

                ordered = sorted(int(row["checkpoint"]) for row in summaries)
                later = [checkpoint for checkpoint in ordered if checkpoint > best_checkpoint]
                window = later[-args.decline_window :]
                if len(window) == args.decline_window:
                    details = load_csv(state_root / "ranking" / "details.csv")
                    evidence = decline_evidence(details, best_checkpoint, window)
                    sustained = all(row["losses_vs_peak"] >= args.losses_required for row in evidence)
                    if sustained:
                        decline = {
                            **payload,
                            "criterion": (
                                f"last {args.decline_window} later Proxy-9 points each lose on at least "
                                f"{args.losses_required}/9 datasets versus peak"
                            ),
                            "detected_at_utc": utc_now(),
                            "evidence": evidence,
                        }
                        atomic_json(state_root / "decline_detected.json", decline)
                        atomic_symlink(state_root / "selected_peak_teacher_checkpoint.pth", source)

            atomic_json(
                state_root / "status.json",
                {
                    "done_points": done,
                    "expected_points": args.expected_points,
                    "failed_points": failed,
                    "updated_at_utc": utc_now(),
                },
            )
            previous_done = done

        if done + failed >= args.expected_points:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
