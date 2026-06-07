from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dinov3.eval.eval_ood.dinov3_runner import RUN_SPECS


@dataclass(frozen=True)
class Protocol:
    name: str
    n_last_blocks: int
    avgpool: bool
    xray_input_mode: str
    cryo_invert: bool
    percentile_low: float = 0.5
    percentile_high: float = 99.5


PROTOCOLS: dict[str, list[Protocol]] = {
    "default": [
        Protocol("nlb1_avg_25d_raw", 1, True, "three_slices", False),
    ],
    "selection": [
        Protocol("nlb4_cls_25d_raw", 4, False, "three_slices", False),
    ],
    "core": [
        Protocol("nlb1_avg_25d_raw", 1, True, "three_slices", False),
        Protocol("nlb1_cls_25d_raw", 1, False, "three_slices", False),
        Protocol("nlb4_avg_25d_raw", 4, True, "three_slices", False),
        Protocol("nlb4_cls_25d_raw", 4, False, "three_slices", False),
        Protocol("nlb1_avg_slice_raw", 1, True, "slice", False),
        Protocol("nlb1_avg_25d_inv", 1, True, "three_slices", True),
    ],
    "full": [
        Protocol(f"nlb{n}_{'avg' if avg else 'cls'}_{mode}_{'inv' if inv else 'raw'}", n, avg, mode, inv)
        for n in (1, 4)
        for avg in (True, False)
        for mode in ("slice", "three_slices")
        for inv in (False, True)
    ],
}


def _available_iters(ckpt_root: Path) -> list[str]:
    return sorted([p.name for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda x: int(x))


def _select_iters(mode: str, spec_name: str, explicit: list[str] | None) -> list[str]:
    spec = RUN_SPECS[spec_name]
    if explicit:
        return explicit
    if mode == "suggested":
        return [x for x in spec.suggested_iters if (spec.ckpt_root / x).is_dir()]
    numeric = _available_iters(spec.ckpt_root)
    if mode == "latest":
        return [numeric[-1]]
    if mode == "all":
        return numeric
    raise ValueError(f"Unknown ckpt mode: {mode}")


def _command(args, run_name: str, ckpt_iter: str, protocol: Protocol) -> list[str]:
    spec = RUN_SPECS[run_name]
    model_name = f"{run_name}_{protocol.name}"
    cmd = [
        sys.executable,
        "-m",
        "dinov3.eval.eval_ood.dinov3_runner",
        "--model-name",
        model_name,
        "--ckpt-root",
        str(spec.ckpt_root),
        "--ckpt-iter",
        str(ckpt_iter),
        "--train-config",
        str(spec.train_config),
        "--output-dir",
        str(args.output_dir),
        "--tasks",
        *args.tasks,
        "--device",
        "cuda:0",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--n-last-blocks",
        str(protocol.n_last_blocks),
        "--xray-input-mode",
        protocol.xray_input_mode,
        "--xray-slices-per-volume",
        str(args.xray_slices_per_volume),
        "--cryo-max-particles-per-project",
        str(args.cryo_max_particles_per_project),
        "--id-max-samples",
        str(args.id_max_samples),
        "--id-datasets",
        *args.id_datasets,
        "--percentile-low",
        str(protocol.percentile_low),
        "--percentile-high",
        str(protocol.percentile_high),
        "--seed",
        str(args.seed),
    ]
    if not protocol.avgpool:
        cmd.append("--no-avgpool")
    if protocol.cryo_invert:
        cmd.append("--cryo-invert")
    if args.xray_max_volumes:
        cmd.extend(["--xray-max-volumes", str(args.xray_max_volumes)])
    if args.cryo_max_projects:
        cmd.extend(["--cryo-max-projects", str(args.cryo_max_projects)])
    if args.cryo_max_per_class:
        cmd.extend(["--cryo-max-per-class", str(args.cryo_max_per_class)])
    if args.overwrite_features:
        cmd.append("--overwrite-features")
    return cmd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Launch DINOv3 OOD protocol sweeps on selected GPUs.")
    parser.add_argument("--runs", nargs="+", default=list(RUN_SPECS), choices=sorted(RUN_SPECS))
    parser.add_argument("--ckpt-mode", default="suggested", choices=["suggested", "latest", "all"])
    parser.add_argument("--ckpt-iters", nargs="+")
    parser.add_argument("--protocol-grid", default="default", choices=sorted(PROTOCOLS))
    parser.add_argument("--tasks", nargs="+", default=["xray", "cryo"], choices=["xray", "cryo"])
    parser.add_argument("--gpus", nargs="+", default=["6", "7"])
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmark_runs/eval_ood")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--xray-slices-per-volume", type=int, default=8)
    parser.add_argument("--xray-max-volumes", type=int)
    parser.add_argument("--cryo-max-projects", type=int)
    parser.add_argument("--cryo-max-particles-per-project", type=int, default=20000)
    parser.add_argument("--cryo-max-per-class", type=int)
    parser.add_argument("--id-max-samples", type=int, default=3000)
    parser.add_argument("--id-datasets", nargs="+", default=["bloodmnist", "bbbc048", "cyclops"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.xray_slices_per_volume = min(args.xray_slices_per_volume, 3)
        args.xray_max_volumes = args.xray_max_volumes or 16
        args.cryo_max_projects = args.cryo_max_projects or 2
        args.cryo_max_particles_per_project = min(args.cryo_max_particles_per_project, 1000)
        args.id_max_samples = min(args.id_max_samples, 600)
        args.batch_size = min(args.batch_size, 32)
    jobs: list[list[str]] = []
    for run_name in args.runs:
        for ckpt_iter in _select_iters(args.ckpt_mode, run_name, args.ckpt_iters):
            for protocol in PROTOCOLS[args.protocol_grid]:
                jobs.append(_command(args, run_name, ckpt_iter, protocol))

    print(f"[sweep] jobs={len(jobs)} gpus={args.gpus} max_parallel={args.max_parallel or len(args.gpus)}", flush=True)
    for cmd in jobs:
        print("[cmd]", shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0

    max_parallel = args.max_parallel or len(args.gpus)
    running: list[tuple[subprocess.Popen, str, Path]] = []
    logs_dir = Path(args.output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    next_gpu = 0

    while jobs or running:
        while jobs and len(running) < max_parallel:
            cmd = jobs.pop(0)
            gpu = args.gpus[next_gpu % len(args.gpus)]
            next_gpu += 1
            stamp = f"{int(time.time())}_{len(jobs)}"
            log_path = logs_dir / f"job_{stamp}_gpu{gpu}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            with log_path.open("w") as log:
                log.write("$ " + shlex.join(cmd) + "\n")
                log.flush()
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            print(f"[launch] gpu={gpu} pid={proc.pid} log={log_path}", flush=True)
            running.append((proc, gpu, log_path))

        still: list[tuple[subprocess.Popen, str, Path]] = []
        for proc, gpu, log_path in running:
            code = proc.poll()
            if code is None:
                still.append((proc, gpu, log_path))
            elif code != 0:
                failures += 1
                print(f"[failed] gpu={gpu} code={code} log={log_path}", flush=True)
            else:
                print(f"[done] gpu={gpu} log={log_path}", flush=True)
        running = still
        if running or jobs:
            time.sleep(15)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
