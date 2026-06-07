from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dinov3.eval.eval_ood.dinov3_runner import RUN_SPECS
from dinov3.eval.eval_ood.sweep_dinov3 import PROTOCOLS, Protocol, _select_iters


DINOV3_ROOT = Path("/mnt/huawei_deepcad/dinov3")
DEFAULT_OOD_ROOT = Path("/mnt/huawei_deepcad/benchmark/ood")
DEFAULT_BENCHMARK_ROOT = Path("/mnt/huawei_deepcad/benchmark")
DEFAULT_ENV_NAME = "dinov3"

def _default_hosts_from_env() -> tuple[str, ...]:
    """Worker SSH aliases to launch on, read from the environment.

    Set ``EVAL_OOD_HOSTS`` to a comma- or whitespace-separated list, e.g.
    ``EVAL_OOD_HOSTS="node1,node2,node3"``. Aliases are resolved through your
    local ``~/.ssh/config``. Nothing is hardcoded; if it is unset, pass the
    hosts explicitly with ``--hosts``.
    """
    raw = os.environ.get("EVAL_OOD_HOSTS", "")
    return tuple(h for h in raw.replace(",", " ").split() if h)


DEFAULT_HOSTS = _default_hosts_from_env()

def _default_conda_prefixes_from_env() -> tuple[str, ...]:
    """Candidate conda install prefixes to probe on each remote host.

    Override with ``EVAL_OOD_CONDA_PREFIXES`` (comma/space separated). The token
    ``$HOME`` is expanded on the *remote* host (it is set at login independently
    of ``~/.bashrc``), so the generic defaults below work for any login user
    without hardcoding a username.
    """
    raw = os.environ.get("EVAL_OOD_CONDA_PREFIXES", "")
    items = tuple(p for p in raw.replace(",", " ").split() if p)
    if items:
        return items
    return (
        "$HOME/anaconda3",
        "$HOME/miniconda3",
        "$HOME/mambaforge",
        "$HOME/miniforge3",
        "/opt/conda",
    )


CONDA_PREFIX_CANDIDATES = _default_conda_prefixes_from_env()

DEFAULT_BATCH_SIZE_BY_RUN = {
    "base": 128,
    "vitl_oep1025": 64,
    "channelvit_s6_fixed": 64,
    "hplus_rgb3": 64,
}


@dataclass(frozen=True)
class HostInfo:
    alias: str
    ok: bool
    reason: str
    hostname: str = ""
    gpu_name: str = ""
    gpu_free_mb: int = -1
    gpu_used_mb: int = -1
    conda_prefix: str = ""


@dataclass(frozen=True)
class EvalJob:
    job_id: str
    run_name: str
    ckpt_iter: str
    protocol: str
    model_name: str
    result_path: str
    command: list[str]


def _now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _ssh(alias: str, remote_cmd: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", alias, remote_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def _parse_key_value_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def probe_host(alias: str, *, env_name: str, min_free_mb: int, timeout: int = 30) -> HostInfo:
    # Double-quote (not shlex single-quote) so $HOME expands on the remote host.
    prefixes = " ".join('"' + p.replace('"', '\\"') + '"' for p in CONDA_PREFIX_CANDIDATES)
    remote = f"""
set -u
echo HOSTNAME=$(hostname)
if [ -d {shlex.quote(str(DINOV3_ROOT))} ]; then echo REPO=ok; else echo REPO=missing; fi
if [ -d {shlex.quote(str(DEFAULT_OOD_ROOT))} ]; then echo OOD=ok; else echo OOD=missing; fi
CONDA_PREFIX_FOUND=
for p in {prefixes}; do
  if [ -x "$p/bin/conda" ] && [ -d "$p/envs/{shlex.quote(env_name)}" ]; then
    CONDA_PREFIX_FOUND="$p"
    break
  fi
done
echo CONDA_PREFIX=$CONDA_PREFIX_FOUND
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader,nounits 2>&1 | head -1 | awk -F, '{{gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3); print "GPU_NAME="$1; print "GPU_FREE_MB="$2; print "GPU_USED_MB="$3}}'
else
  echo GPU_NAME=no_nvidia_smi
  echo GPU_FREE_MB=-1
  echo GPU_USED_MB=-1
fi
""".strip()
    try:
        proc = _ssh(alias, remote, timeout=timeout)
    except subprocess.TimeoutExpired:
        return HostInfo(alias=alias, ok=False, reason="ssh_timeout")
    if proc.returncode != 0:
        reason = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"ssh_rc_{proc.returncode}"
        return HostInfo(alias=alias, ok=False, reason=reason)
    kv = _parse_key_value_lines(proc.stdout)
    reason_parts: list[str] = []
    if kv.get("REPO") != "ok":
        reason_parts.append("repo_missing")
    if kv.get("OOD") != "ok":
        reason_parts.append("ood_missing")
    if not kv.get("CONDA_PREFIX"):
        reason_parts.append(f"env_{env_name}_missing")
    try:
        free_mb = int(float(kv.get("GPU_FREE_MB", "-1")))
    except ValueError:
        free_mb = -1
    try:
        used_mb = int(float(kv.get("GPU_USED_MB", "-1")))
    except ValueError:
        used_mb = -1
    gpu_name = kv.get("GPU_NAME", "")
    if "RTX 3090" not in gpu_name:
        reason_parts.append(f"gpu_not_3090:{gpu_name or 'unknown'}")
    if free_mb < min_free_mb:
        reason_parts.append(f"free_mb_{free_mb}_lt_{min_free_mb}")
    return HostInfo(
        alias=alias,
        ok=not reason_parts,
        reason="ok" if not reason_parts else ",".join(reason_parts),
        hostname=kv.get("HOSTNAME", ""),
        gpu_name=gpu_name,
        gpu_free_mb=free_mb,
        gpu_used_mb=used_mb,
        conda_prefix=kv.get("CONDA_PREFIX", ""),
    )


def _parse_batch_overrides(items: list[str] | None) -> dict[str, int]:
    if not items:
        return {}
    out: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected RUN=BATCH for --batch-size-override, got {item!r}")
        run_name, value = item.split("=", 1)
        if run_name not in RUN_SPECS:
            raise ValueError(f"Unknown run in --batch-size-override: {run_name}")
        out[run_name] = int(value)
    return out


def _parse_ckpt_map(items: list[str] | None) -> dict[str, str]:
    if not items:
        return {}
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected RUN=ITER for --ckpt-map, got {item!r}")
        run_name, value = item.split("=", 1)
        if run_name not in RUN_SPECS:
            raise ValueError(f"Unknown run in --ckpt-map: {run_name}")
        out[run_name] = value
    return out


def _batch_size_for_run(args: argparse.Namespace, run_name: str) -> int:
    if args.batch_size is not None:
        return int(args.batch_size)
    overrides = _parse_batch_overrides(args.batch_size_override)
    if run_name in overrides:
        return overrides[run_name]
    return DEFAULT_BATCH_SIZE_BY_RUN[run_name]


def _iters_for_run(args: argparse.Namespace, run_name: str) -> list[str]:
    ckpt_map = _parse_ckpt_map(args.ckpt_map)
    if ckpt_map:
        if run_name not in ckpt_map:
            raise ValueError(f"Missing --ckpt-map entry for run {run_name}")
        return [ckpt_map[run_name]]
    return _select_iters(args.ckpt_mode, run_name, args.ckpt_iters)


def _protocols_for_args(args: argparse.Namespace) -> list[Protocol]:
    protocols = PROTOCOLS[args.protocol_grid]
    if not args.protocol_names:
        return protocols
    requested = set(args.protocol_names)
    selected = [p for p in protocols if p.name in requested]
    missing = sorted(requested - {p.name for p in selected})
    if missing:
        raise ValueError(f"Unknown protocol names for grid {args.protocol_grid}: {missing}")
    return selected


def _append_if_present(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _command_for_job(
    args: argparse.Namespace,
    *,
    run_name: str,
    ckpt_iter: str,
    protocol: Protocol,
    results_dir: Path,
) -> list[str]:
    spec = RUN_SPECS[run_name]
    cmd = [
        "<PYTHON>",
        "-m",
        "dinov3.eval.eval_ood.dinov3_runner",
        "--model-name",
        f"{run_name}_{protocol.name}",
        "--ckpt-root",
        str(spec.ckpt_root),
        "--ckpt-iter",
        str(ckpt_iter),
        "--train-config",
        str(spec.train_config),
        "--output-dir",
        str(results_dir),
        "--ood-root",
        str(args.ood_root),
        "--benchmark-root",
        str(args.benchmark_root),
        "--tasks",
        *args.tasks,
        "--device",
        "cuda:0",
        "--batch-size",
        str(_batch_size_for_run(args, run_name)),
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
        "--autocast-dtype",
        str(args.autocast_dtype),
        "--seed",
        str(args.seed),
    ]
    if not protocol.avgpool:
        cmd.append("--no-avgpool")
    if protocol.cryo_invert:
        cmd.append("--cryo-invert")
    _append_if_present(cmd, "--xray-max-volumes", args.xray_max_volumes)
    _append_if_present(cmd, "--cryo-max-projects", args.cryo_max_projects)
    _append_if_present(cmd, "--cryo-max-per-class", args.cryo_max_per_class)
    if args.overwrite_features:
        cmd.append("--overwrite-features")
    return cmd


def build_jobs(args: argparse.Namespace, run_dir: Path) -> list[EvalJob]:
    if args.smoke:
        args.xray_slices_per_volume = min(args.xray_slices_per_volume, 3)
        args.xray_max_volumes = args.xray_max_volumes or 16
        args.cryo_max_projects = args.cryo_max_projects or 2
        args.cryo_max_particles_per_project = min(args.cryo_max_particles_per_project, 1000)
        args.id_max_samples = min(args.id_max_samples, 600)
        args.batch_size = min(args.batch_size or 32, 32)

    results_dir = run_dir / "results"
    jobs: list[EvalJob] = []
    protocols = _protocols_for_args(args)
    for run_name in args.runs:
        for ckpt_iter in _iters_for_run(args, run_name):
            for protocol in protocols:
                model_name = f"{run_name}_{protocol.name}"
                result_path = results_dir / model_name / str(ckpt_iter) / "last_result.json"
                if result_path.exists() and not args.overwrite_jobs:
                    continue
                job_id = f"{len(jobs):04d}_{run_name}_{ckpt_iter}_{protocol.name}"
                jobs.append(
                    EvalJob(
                        job_id=job_id,
                        run_name=run_name,
                        ckpt_iter=str(ckpt_iter),
                        protocol=protocol.name,
                        model_name=model_name,
                        result_path=str(result_path),
                        command=_command_for_job(
                            args,
                            run_name=run_name,
                            ckpt_iter=str(ckpt_iter),
                            protocol=protocol,
                            results_dir=results_dir,
                        ),
                    )
                )
    return jobs


def write_manifest(path: Path, jobs: list[EvalJob], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for job in jobs:
            f.write(json.dumps(asdict(job), sort_keys=True) + "\n")
    _atomic_write_json(path.with_suffix(".meta.json"), meta)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def _state_dirs(manifest: Path) -> dict[str, Path]:
    root = manifest.parent / "state"
    dirs = {
        "root": root,
        "claims": root / "claims",
        "done": root / "done",
        "failed": root / "failed",
        "workers": root / "workers",
        "logs": manifest.parent / "logs",
        "worker_logs": manifest.parent / "worker_logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _claim_next_job(manifest: Path, host_alias: str) -> dict[str, Any] | None:
    dirs = _state_dirs(manifest)
    jobs = read_manifest(manifest)
    lock_path = dirs["root"] / "queue.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for job in jobs:
            job_id = job["job_id"]
            if (dirs["done"] / f"{job_id}.json").exists():
                continue
            if (dirs["failed"] / f"{job_id}.json").exists():
                continue
            if (dirs["claims"] / f"{job_id}.json").exists():
                continue
            claim = {
                "job_id": job_id,
                "host_alias": host_alias,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "claimed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _atomic_write_json(dirs["claims"] / f"{job_id}.json", claim)
            return job
    return None


def worker_main(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest)
    dirs = _state_dirs(manifest)
    worker_info = {
        "host_alias": args.host_alias,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": args.gpu,
        "python": sys.executable,
    }
    _atomic_write_json(dirs["workers"] / f"{args.host_alias}.json", worker_info)
    failures = 0
    while True:
        job = _claim_next_job(manifest, args.host_alias)
        if job is None:
            break

        job_id = job["job_id"]
        log_path = dirs["logs"] / f"{job_id}.{args.host_alias}.log"
        command = [sys.executable if token == "<PYTHON>" else token for token in job["command"]]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMP_NUM_THREADS", "4")

        started = time.time()
        with log_path.open("w") as log:
            log.write(f"# host_alias={args.host_alias} hostname={socket.gethostname()} pid={os.getpid()}\n")
            log.write("# cwd=" + str(DINOV3_ROOT) + "\n")
            log.write("$ " + shlex.join(command) + "\n")
            log.flush()
            try:
                proc = subprocess.run(
                    command,
                    cwd=DINOV3_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=None if args.job_timeout <= 0 else args.job_timeout,
                )
                rc = int(proc.returncode)
            except subprocess.TimeoutExpired:
                rc = 124
                log.write(f"\n# TIMEOUT after {args.job_timeout} seconds\n")
        payload = {
            "job_id": job_id,
            "host_alias": args.host_alias,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "returncode": rc,
            "seconds": round(time.time() - started, 3),
            "log_path": str(log_path),
            "result_path": job["result_path"],
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if rc == 0:
            _atomic_write_json(dirs["done"] / f"{job_id}.json", payload)
        else:
            failures += 1
            _atomic_write_json(dirs["failed"] / f"{job_id}.json", payload)
    worker_info["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    worker_info["failures"] = failures
    _atomic_write_json(dirs["workers"] / f"{args.host_alias}.json", worker_info)
    return 1 if failures else 0


def launch_worker(host: HostInfo, manifest: Path, args: argparse.Namespace) -> int:
    dirs = _state_dirs(manifest)
    worker_log = dirs["worker_logs"] / f"{host.alias}.log"
    remote_args = [
        str(Path(host.conda_prefix) / "bin" / "conda"),
        "run",
        "--no-capture-output",
        "-n",
        args.env_name,
        "python",
        "-m",
        "dinov3.eval.eval_ood.remote_launcher",
        "worker",
        "--manifest",
        str(manifest),
        "--host-alias",
        host.alias,
        "--gpu",
        str(args.gpu),
        "--job-timeout",
        str(args.job_timeout),
    ]
    remote = (
        f"mkdir -p {shlex.quote(str(worker_log.parent))} && "
        f"cd {shlex.quote(str(DINOV3_ROOT))} && "
        f"setsid -f {shlex.join(remote_args)} > {shlex.quote(str(worker_log))} 2>&1 < /dev/null && "
        f"echo launched"
    )
    if args.dry_run:
        print(f"[dry-run launch-worker] {host.alias}: {remote}")
        return 0
    proc = _ssh(host.alias, remote, timeout=20)
    if proc.returncode != 0:
        msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else proc.stdout.strip()
        print(f"[worker failed to launch] {host.alias}: {msg}", flush=True)
        return 1
    pid_text = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    payload = {
        "host_alias": host.alias,
        "hostname": host.hostname,
        "gpu_name": host.gpu_name,
        "gpu_free_mb": host.gpu_free_mb,
        "conda_prefix": host.conda_prefix,
        "launcher_pid": pid_text,
        "worker_log": str(worker_log),
        "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_write_json(dirs["workers"] / f"{host.alias}.launch.json", payload)
    print(f"[worker] {host.alias} launch={pid_text} free_mb={host.gpu_free_mb} log={worker_log}", flush=True)
    return 0


def launch_main(args: argparse.Namespace) -> int:
    run_dir = Path(args.output_dir) if args.output_dir else Path("benchmark_runs/eval_ood_remote") / _now_stamp()
    if not run_dir.is_absolute():
        run_dir = DINOV3_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args, run_dir)
    manifest = run_dir / "remote_jobs.jsonl"
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "created_on": socket.gethostname(),
        "runs": args.runs,
        "ckpt_mode": args.ckpt_mode,
        "ckpt_iters": args.ckpt_iters,
        "protocol_grid": args.protocol_grid,
        "tasks": args.tasks,
        "autocast_dtype": args.autocast_dtype,
        "output_dir": str(run_dir),
        "results_dir": str(run_dir / "results"),
        "job_count": len(jobs),
    }
    write_manifest(manifest, jobs, meta)

    print(f"[plan] run_dir={run_dir}", flush=True)
    print(f"[plan] manifest={manifest}", flush=True)
    print(f"[plan] jobs={len(jobs)} runs={args.runs} ckpt_mode={args.ckpt_mode} protocol_grid={args.protocol_grid}", flush=True)
    for job in jobs[: min(8, len(jobs))]:
        print("[job]", job.job_id, shlex.join(job.command), flush=True)
    if len(jobs) > 8:
        print(f"[job] ... {len(jobs) - 8} more", flush=True)
    if args.plan_only:
        return 0

    if not args.hosts:
        print(
            "[error] no hosts specified; set EVAL_OOD_HOSTS (comma/space separated) "
            "or pass --hosts host1 host2 ...",
            flush=True,
        )
        return 2

    print(f"[probe] hosts={args.hosts} min_free_mb={args.min_free_mb}", flush=True)
    probed = [probe_host(h, env_name=args.env_name, min_free_mb=args.min_free_mb, timeout=args.ssh_timeout) for h in args.hosts]
    good = [h for h in probed if h.ok]
    if args.max_hosts:
        good = good[: args.max_hosts]
    for host in probed:
        status = "ok" if host.ok else "skip"
        print(
            f"[probe] {status} {host.alias} host={host.hostname} gpu={host.gpu_name} "
            f"free={host.gpu_free_mb} conda={host.conda_prefix or '-'} reason={host.reason}",
            flush=True,
        )
    if not good:
        print("[error] no usable 3090 hosts after probing", flush=True)
        return 2

    failures = 0
    for host in good:
        failures += launch_worker(host, manifest, args)
    print(f"[launch] workers={len(good)} failures={failures}", flush=True)
    print(f"[status] python -m dinov3.eval.eval_ood.remote_launcher status --run-dir {shlex.quote(str(run_dir))}", flush=True)
    return 1 if failures else 0


def status_main(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    manifest = run_dir / "remote_jobs.jsonl"
    if not manifest.exists():
        print(f"[error] missing manifest: {manifest}", flush=True)
        return 2
    dirs = _state_dirs(manifest)
    jobs = read_manifest(manifest)
    total = len(jobs)
    done = {p.stem for p in dirs["done"].glob("*.json")}
    failed = {p.stem for p in dirs["failed"].glob("*.json")}
    claims = {p.stem for p in dirs["claims"].glob("*.json")}
    active = sorted(claims - done - failed)
    pending = total - len(done) - len(failed) - len(active)
    print(f"[status] run_dir={run_dir}", flush=True)
    print(f"[status] total={total} done={len(done)} failed={len(failed)} active={len(active)} pending={pending}", flush=True)

    worker_files = sorted(dirs["workers"].glob("*.launch.json"))
    for path in worker_files:
        info = _read_json(path)
        print(
            f"[worker] {info.get('host_alias')} pid={info.get('launcher_pid')} "
            f"free_mb={info.get('gpu_free_mb')} log={info.get('worker_log')}",
            flush=True,
        )

    if active:
        print("[active]", " ".join(active[:20]) + (" ..." if len(active) > 20 else ""), flush=True)
    failed_files = sorted(dirs["failed"].glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in failed_files[: args.show_failures]:
        info = _read_json(path)
        print(f"[failed] {path.stem} host={info.get('host_alias')} rc={info.get('returncode')} log={info.get('log_path')}", flush=True)
    done_files = sorted(dirs["done"].glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in done_files[: args.show_done]:
        info = _read_json(path)
        print(f"[done] {path.stem} host={info.get('host_alias')} seconds={info.get('seconds')} result={info.get('result_path')}", flush=True)
    return 1 if failed else 0


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _infer_run_name(row: dict[str, str]) -> str:
    train_config = row.get("train_config", "")
    for run_name, spec in RUN_SPECS.items():
        if str(spec.output_dir) in train_config or spec.output_dir.name in train_config:
            return run_name
    model = row.get("model", "")
    for run_name in sorted(RUN_SPECS, key=len, reverse=True):
        if model.startswith(run_name + "_"):
            return run_name
    return "unknown"


def _infer_ckpt_iter(row: dict[str, str]) -> str:
    checkpoint = row.get("checkpoint", "").rstrip("/")
    if checkpoint.endswith("/checkpoint.pth"):
        return Path(checkpoint).parent.name
    if checkpoint:
        return Path(checkpoint).name
    model = row.get("model", "")
    return model.rsplit("-", 1)[-1] if "-" in model else ""


def rank_main(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    summary = Path(args.summary) if args.summary else run_dir / "results" / "summary.csv"
    if not summary.exists():
        print(f"[error] missing summary: {summary}", flush=True)
        return 2

    metric_names = [
        "xray_pair_recall_at_1",
        "xray_dose_r2",
        "cryo_class_accuracy",
        "cryo_quality_auroc",
        "cryo_retrieval_map_at_10",
    ]
    if args.include_ood:
        metric_names = ["xray_ood_auroc", "cryo_ood_auroc"] + metric_names

    ranked: list[dict[str, Any]] = []
    with summary.open(newline="") as f:
        for row in csv.DictReader(f):
            metrics = {name: _float_or_nan(row.get(name)) for name in metric_names}
            vals = [v for v in metrics.values() if v == v]
            score = sum(vals) / len(vals) if vals else float("nan")
            ranked.append(
                {
                    "score": score,
                    "run": _infer_run_name(row),
                    "ckpt_iter": _infer_ckpt_iter(row),
                    "model": row.get("model", ""),
                    "xray_ood_auroc": _float_or_nan(row.get("xray_ood_auroc")),
                    "xray_pair_recall_at_1": _float_or_nan(row.get("xray_pair_recall_at_1")),
                    "xray_dose_r2": _float_or_nan(row.get("xray_dose_r2")),
                    "cryo_class_accuracy": _float_or_nan(row.get("cryo_class_accuracy")),
                    "cryo_quality_auroc": _float_or_nan(row.get("cryo_quality_auroc")),
                    "cryo_retrieval_map_at_10": _float_or_nan(row.get("cryo_retrieval_map_at_10")),
                    "checkpoint": row.get("checkpoint", ""),
                }
            )
    ranked.sort(key=lambda r: r["score"], reverse=True)

    output = Path(args.output) if args.output else summary.parent / "ranking_by_composite.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "score",
        "run",
        "ckpt_iter",
        "model",
        "xray_ood_auroc",
        "xray_pair_recall_at_1",
        "xray_dose_r2",
        "cryo_class_accuracy",
        "cryo_quality_auroc",
        "cryo_retrieval_map_at_10",
        "checkpoint",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)

    print(f"[rank] summary={summary}", flush=True)
    print(f"[rank] output={output}", flush=True)
    print("[rank] top configurations", flush=True)
    print("score\trun\tckpt\tmodel\txood\txR1\tdoseR2\tcryoAcc\tqAUROC\tmap10", flush=True)
    for row in ranked[: args.top_k]:
        print(
            f"{row['score']:.4f}\t{row['run']}\t{row['ckpt_iter']}\t{row['model']}\t"
            f"{row['xray_ood_auroc']:.3f}\t{row['xray_pair_recall_at_1']:.3f}\t"
            f"{row['xray_dose_r2']:.3f}\t{row['cryo_class_accuracy']:.3f}\t"
            f"{row['cryo_quality_auroc']:.3f}\t{row['cryo_retrieval_map_at_10']:.4f}",
            flush=True,
        )

    if args.best_per_run:
        print("[rank] best per run", flush=True)
        seen: set[str] = set()
        for row in ranked:
            if row["run"] in seen:
                continue
            seen.add(row["run"])
            print(
                f"{row['run']}\t{row['score']:.4f}\tckpt={row['ckpt_iter']}\t{row['model']}\t"
                f"R1={row['xray_pair_recall_at_1']:.3f}\tdoseR2={row['xray_dose_r2']:.3f}\t"
                f"cryoAcc={row['cryo_class_accuracy']:.3f}\tqAUROC={row['cryo_quality_auroc']:.3f}\t"
                f"map10={row['cryo_retrieval_map_at_10']:.4f}",
                flush=True,
            )
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distribute DINOv3 OOD sweeps across SSH-accessible RTX 3090 hosts.")
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="Create a manifest and start one queue worker per healthy 3090 host.")
    launch.add_argument("--runs", nargs="+", default=list(RUN_SPECS), choices=sorted(RUN_SPECS))
    launch.add_argument("--ckpt-mode", default="suggested", choices=["suggested", "latest", "all"])
    launch.add_argument("--ckpt-iters", nargs="+")
    launch.add_argument("--ckpt-map", nargs="+", help="Per-run checkpoint map, e.g. base=12299 vitl_oep1025=8199.")
    launch.add_argument("--protocol-grid", default="core", choices=sorted(PROTOCOLS))
    launch.add_argument("--protocol-names", nargs="+", help="Optional subset of protocol names from --protocol-grid.")
    launch.add_argument("--tasks", nargs="+", default=["xray", "cryo"], choices=["xray", "cryo"])
    launch.add_argument(
        "--hosts",
        nargs="+",
        default=list(DEFAULT_HOSTS),
        help="Worker SSH aliases. Defaults to the EVAL_OOD_HOSTS env var (comma/space separated).",
    )
    launch.add_argument("--max-hosts", type=int, default=0)
    launch.add_argument("--min-free-mb", type=int, default=16000)
    launch.add_argument("--gpu", default="0")
    launch.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    launch.add_argument("--output-dir", help="Run directory; default is benchmark_runs/eval_ood_remote/<timestamp>.")
    launch.add_argument("--ood-root", default=str(DEFAULT_OOD_ROOT))
    launch.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT))
    launch.add_argument("--batch-size", type=int)
    launch.add_argument("--batch-size-override", nargs="+", default=[f"{k}={v}" for k, v in DEFAULT_BATCH_SIZE_BY_RUN.items()])
    launch.add_argument("--num-workers", type=int, default=4)
    launch.add_argument("--seed", type=int, default=0)
    launch.add_argument("--xray-slices-per-volume", type=int, default=8)
    launch.add_argument("--xray-max-volumes", type=int)
    launch.add_argument("--cryo-max-projects", type=int)
    launch.add_argument("--cryo-max-particles-per-project", type=int, default=20000)
    launch.add_argument("--cryo-max-per-class", type=int)
    launch.add_argument("--id-max-samples", type=int, default=3000)
    launch.add_argument(
        "--id-datasets",
        nargs="+",
        default=["bloodmnist"],
        help="ID reference datasets for kNN OOD scoring; keep bloodmnist-only for fast checkpoint selection.",
    )
    launch.add_argument("--autocast-dtype", default="fp16", choices=["bf16", "fp16", "fp32"])
    launch.add_argument("--smoke", action="store_true")
    launch.add_argument("--overwrite-features", action="store_true")
    launch.add_argument("--overwrite-jobs", action="store_true")
    launch.add_argument("--plan-only", action="store_true")
    launch.add_argument("--dry-run", action="store_true", help="Probe and print SSH launch commands without starting workers.")
    launch.add_argument("--ssh-timeout", type=int, default=30)
    launch.add_argument("--job-timeout", type=int, default=0, help="Per-job timeout in seconds; 0 means no timeout.")

    worker = sub.add_parser("worker", help="Internal queue worker launched on remote hosts.")
    worker.add_argument("--manifest", required=True)
    worker.add_argument("--host-alias", required=True)
    worker.add_argument("--gpu", default="0")
    worker.add_argument("--job-timeout", type=int, default=0)

    status = sub.add_parser("status", help="Summarize a remote sweep run directory.")
    status.add_argument("--run-dir", required=True)
    status.add_argument("--show-failures", type=int, default=8)
    status.add_argument("--show-done", type=int, default=5)

    rank = sub.add_parser("rank", help="Rank completed OOD results for checkpoint/protocol selection.")
    rank.add_argument("--run-dir", required=True)
    rank.add_argument("--summary", help="Optional summary.csv path; defaults to <run-dir>/results/summary.csv.")
    rank.add_argument("--output", help="Optional ranking CSV path; defaults to <run-dir>/results/ranking_by_composite.csv.")
    rank.add_argument("--top-k", type=int, default=12)
    rank.add_argument("--include-ood", action="store_true", help="Include saturated xray/cryo OOD AUROC in the composite.")
    rank.add_argument("--best-per-run", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "launch":
        return launch_main(args)
    if args.command == "worker":
        return worker_main(args)
    if args.command == "status":
        return status_main(args)
    if args.command == "rank":
        return rank_main(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
