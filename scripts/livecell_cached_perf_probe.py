#!/usr/bin/env python3
"""One-off LIVECell cached performance probe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PYTHON = "/home/inspur/anaconda3/envs/dinov3/bin/python"
CHECKPOINT = ROOT / "outputs" / "01_training_runs" / "bio_continue_rgb3_vith16plus" / "ckpt" / "14349"
HEAD = ROOT / "outputs" / "instance_seg_tuning" / "bio_continue_rgb3_vith16plus_seven_dataset" / "livecell" / "livecell_validated" / "best_head.pth"
SCRIPT = ROOT / "scripts" / "run_biodino_test_optimization_cached.py"


def children(pid: int):
    path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        text = path.read_text().strip()
    except Exception:
        return []
    if not text:
        return []
    out = []
    for token in text.split():
        try:
            out.append(int(token))
        except ValueError:
            pass
    return out


def tree_pids(root_pid: int):
    seen = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for child in children(pid):
            if child not in seen:
                stack.append(child)
    return sorted(seen)


def rss_kib(pid: int) -> int:
    path = Path("/proc") / str(pid) / "status"
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return 0
    for line in lines:
        if line.startswith("VmRSS:"):
            parts = line.split()
            try:
                return int(parts[1])
            except Exception:
                return 0
    return 0


def read_bytes(pid: int) -> int:
    path = Path("/proc") / str(pid) / "io"
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return 0
    for line in lines:
        if line.startswith("read_bytes:"):
            parts = line.split()
            try:
                return int(parts[1])
            except Exception:
                return 0
    return 0


def sample_tree(root_pid: int):
    rss = 0
    reads = 0
    pids = 0
    for pid in tree_pids(root_pid):
        if not (Path("/proc") / str(pid)).exists():
            continue
        pids += 1
        rss += rss_kib(pid)
        reads += read_bytes(pid)
    return {"rss_kib": rss, "read_bytes": reads, "pids": pids}


def monitor(proc: subprocess.Popen, interval: float = 0.25):
    start = time.time()
    baseline = None
    peak_rss = 0
    peak_read = 0
    peak_pids = 0
    while True:
        if proc.poll() is not None:
            break
        snap = sample_tree(proc.pid)
        if baseline is None:
            baseline = snap
        delta_read = snap["read_bytes"] - baseline["read_bytes"] if baseline else 0
        peak_rss = max(peak_rss, snap["rss_kib"])
        peak_read = max(peak_read, delta_read)
        peak_pids = max(peak_pids, snap["pids"])
        time.sleep(interval)
    snap = sample_tree(proc.pid)
    if baseline is None:
        baseline = snap
    delta_read = snap["read_bytes"] - baseline["read_bytes"] if baseline else 0
    peak_rss = max(peak_rss, snap["rss_kib"])
    peak_read = max(peak_read, delta_read)
    peak_pids = max(peak_pids, snap["pids"])
    rc = proc.wait()
    return {
        "rc": rc,
        "wall_seconds": time.time() - start,
        "peak_rss_mib": peak_rss / 1024.0,
        "peak_read_mib": peak_read / (1024.0 * 1024.0),
        "peak_pid_count": peak_pids,
    }


def run(cmd, log_path: Path, interval: float = 0.25):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        stats = monitor(proc, interval=interval)
    stats["log_path"] = str(log_path)
    return stats


def selected_candidates():
    return [
        ("fg_0.180_energy_0.470_min_20_sobel_15", (0.18, 0.47, 20, 15)),
        ("fg_0.180_energy_0.470_min_20_sobel_21", (0.18, 0.47, 20, 21)),
        ("fg_0.180_energy_0.470_min_20_sobel_31", (0.18, 0.47, 20, 31)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    out_root = Path(args.output_root).resolve()
    cache_dir = out_root / "cache" / "livecell"
    bench_dir = out_root / "bench" / "livecell"
    log_dir = out_root / "logs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": str(CHECKPOINT),
        "head": str(HEAD),
        "cache_dir": str(cache_dir),
        "bench_dir": str(bench_dir),
        "selected": [{"tag": tag, "params": list(params)} for tag, params in selected_candidates()],
        "workers": [1, 4, 8],
    }

    prep_cmd = [
        PYTHON,
        "-u",
        str(SCRIPT),
        "prepare-cache",
        "--dataset",
        "livecell",
        "--head",
        str(HEAD),
        "--checkpoint",
        str(CHECKPOINT),
        "--cache-dir",
        str(cache_dir),
        "--force",
    ]
    if (cache_dir / "manifest.json").exists():
        prep_stats = {
            "rc": 0,
            "wall_seconds": 0.0,
            "peak_rss_mib": 0.0,
            "peak_read_mib": 0.0,
            "peak_pid_count": 0,
            "log_path": str(log_dir / "prepare-cache.log"),
            "skipped_existing_cache": True,
        }
        print(json.dumps({"stage": "prepare-cache-skip-existing", "cache_dir": str(cache_dir)}, ensure_ascii=False), flush=True)
    else:
        print(json.dumps({"stage": "prepare-cache-start", "cache_dir": str(cache_dir)}, ensure_ascii=False), flush=True)
        prep_stats = run(prep_cmd, log_dir / "prepare-cache.log", interval=0.5)
    summary["prepare"] = prep_stats
    print(json.dumps({"stage": "prepare-cache-done", **prep_stats}, ensure_ascii=False, sort_keys=True), flush=True)
    if prep_stats["rc"] != 0:
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return int(prep_stats["rc"])

    import scripts.instseg_optimization_workflow as w
    import scripts.run_biodino_test_optimization_cached as c
    from scripts.run_biodino_test_optimization import cfg as legacy_cfg

    manifest = c.load_manifest(cache_dir)
    cached = [c.load_sample(cache_dir, rec) for rec in manifest["samples"]]
    n = len(cached)
    num_types = w.DATASET_NUM_TYPES.get("livecell", 0)

    benchmark_rows = []
    for tag, params in selected_candidates():
        for workers in [1, 4, 8]:
            out_dir = bench_dir / tag / f"workers_{workers}"
            out_dir.mkdir(parents=True, exist_ok=True)
            bench_log = log_dir / f"{tag}_workers_{workers}.log"
            inline = "\n".join(
                [
                    "from pathlib import Path",
                    "import scripts.run_biodino_test_optimization_cached as c",
                    f"cache_dir = Path({repr(str(cache_dir))})",
                    f"out_dir = Path({repr(str(out_dir))})",
                    f"params = {repr(tuple(params))}",
                    f"c.evaluate_one(cache_dir, out_dir, {repr(tag)}, params, {workers})",
                ]
            )
            cmd = [PYTHON, "-u", "-c", inline]
            print(json.dumps({"stage": "benchmark-start", "tag": tag, "workers": workers, "output_dir": str(out_dir)}, ensure_ascii=False), flush=True)
            stats = run(cmd, bench_log, interval=0.25)
            result_path = out_dir / "metrics.json"
            result = json.loads(result_path.read_text())
            benchmark_rows.append(
                {
                    "tag": tag,
                    "workers": workers,
                    "params": list(params),
                    "stats": stats,
                    "result": {
                        "primary_metric": result["primary_metric"],
                        "primary_value": result["metrics"].get(result["primary_metric"]),
                        "post_seconds": result.get("post_seconds"),
                        "n": result.get("n"),
                        "checkpoint_sha256": result.get("checkpoint_sha256"),
                        "head_sha256": result.get("head_sha256"),
                        "metrics_path": str(result_path),
                        "log_path": str(bench_log),
                    },
                }
            )
            print(json.dumps({"stage": "benchmark-done", "tag": tag, "workers": workers, **stats}, ensure_ascii=False, sort_keys=True), flush=True)
            if stats["rc"] != 0:
                (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                return int(stats["rc"])

    verification_rows = []
    for tag, params in selected_candidates():
        old_cfg = legacy_cfg("livecell", Path("fixture"), tag, *params)
        for workers in [1, 4, 8]:
            out_dir = bench_dir / tag / f"workers_{workers}"
            new = json.loads((out_dir / "metrics.json").read_text())
            old = w._eval_row_from_cached_outputs(
                old_cfg,
                "eval_only",
                new["primary_metric"],
                num_types,
                n,
                cached,
                "cached",
                0,
                0,
                0.0,
                0.0,
            )
            keys = sorted(set(old["metrics"]) | set(new["metrics"]))
            diffs = {k: abs(float(old["metrics"].get(k, float("nan"))) - float(new["metrics"].get(k, float("nan")))) for k in keys}
            max_diff = max(diffs.values()) if diffs else 0.0
            verification_rows.append(
                {
                    "tag": tag,
                    "workers": workers,
                    "max_abs_diff": max_diff,
                    "pass": max_diff < 1e-10,
                    "AJI": new["metrics"].get("AJI"),
                    "bPQ": new["metrics"].get("bPQ"),
                    "primary": new["metrics"].get(new["primary_metric"]),
                }
            )

    summary["benchmarks"] = benchmark_rows
    summary["verification"] = verification_rows
    summary["livecell_sample_count"] = n
    summary["cache_bytes"] = sum((cache_dir / rec["path"]).stat().st_size for rec in manifest["samples"])
    summary["cache_build_seconds"] = prep_stats["wall_seconds"]
    summary["remaining_candidates"] = 685

    worker_summary = {}
    for workers in [1, 4, 8]:
        rows = [row for row in benchmark_rows if row["workers"] == workers]
        mean_post = sum(float(row["result"]["post_seconds"]) for row in rows) / len(rows)
        mean_wall = sum(float(row["stats"]["wall_seconds"]) for row in rows) / len(rows)
        peak_rss = max(float(row["stats"]["peak_rss_mib"]) for row in rows)
        peak_read = max(float(row["stats"]["peak_read_mib"]) for row in rows)
        worker_summary[workers] = {
            "mean_post_seconds": mean_post,
            "mean_wall_seconds": mean_wall,
            "peak_rss_mib": peak_rss,
            "peak_read_mib": peak_read,
            "eta_remaining_seconds": prep_stats["wall_seconds"] + 685 * mean_post,
        }
    worker_summary[4]["speedup_vs_w1"] = worker_summary[1]["mean_post_seconds"] / worker_summary[4]["mean_post_seconds"] if worker_summary[4]["mean_post_seconds"] else 0.0
    worker_summary[8]["speedup_vs_w1"] = worker_summary[1]["mean_post_seconds"] / worker_summary[8]["mean_post_seconds"] if worker_summary[8]["mean_post_seconds"] else 0.0
    summary["worker_summary"] = worker_summary

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "summary-written", "summary_path": str(summary_path)}, ensure_ascii=False, sort_keys=True), flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
