#!/usr/bin/env python3
"""Resource-gated placement controller for existing 20TB screening campaigns.

The evaluator, results and atomic lane claims belong to each existing campaign.
This controller selects one checkpoint/lane, uses a symlink view of that teacher,
and invokes the unchanged canonical fleet worker with --once.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

FROZEN = ("classification_a", "classification_b", "classification_c", "classification_d", "regression", "retrieval")
DENSE = ("segmentation_a", "segmentation_b", "segmentation_c", "segmentation_d", "detection")


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def pending(cfg, group):
    lanes = DENSE if group == "dense" else FROZEN
    for target in cfg["targets"]:
        arm, ck = target["arm"], target["checkpoint"]
        root = Path(cfg["eval_root"]) / arm
        config = root / "source/config.yaml"
        expected_config_sha = target.get("config_sha256")
        if expected_config_sha and hashlib.sha256(config.read_bytes()).hexdigest() != expected_config_sha:
            raise RuntimeError(f"Training config changed for {arm}/{ck}")
        teacher = Path(target["snapshot_root"]) / f"training_{ck}" / "teacher_checkpoint.pth"
        if not teacher.is_file() or teacher.stat().st_size < 1_000_000 or time.time() - teacher.stat().st_mtime < 120:
            continue
        state = root / "results/_state"
        for lane in lanes:
            key = f"ckpt_{ck}__{lane}"
            if (state / "done" / f"{key}.json").exists() or (state / "terminal" / f"{key}.json").exists():
                continue
            if (state / "claims" / f"{key}.lock").exists():
                continue
            yield target, lane, root, teacher


def gpu_memory(gpu):
    output = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
    uuid, used, total = output.strip().split(",")
    return uuid.strip(), int(used), int(total)


def run(args):
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    out = Path(cfg["control_root"])
    state = out / "workers" / f"{args.host}_{args.gpu}_{args.slot}.json"
    lockpath = out / "locks" / f"{args.host}_gpu{args.gpu}.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    ownlock = lockpath.with_name(lockpath.name + "." + args.slot)
    own = ownlock.open("a")
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = False

    def stop(*unused):
        nonlocal stopping
        stopping = True  # Drain the active lane; do not discard completed work.

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    worker = Path(cfg["worker"])
    digest = hashlib.sha256(worker.read_bytes()).hexdigest()
    if digest != cfg["worker_sha256"]:
        raise RuntimeError("Fleet worker fingerprint changed")
    while not stopping:
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("paused"):
            break
        if args.gpu in cfg.get('reserved_training_gpus', {}).get(args.host, []):
            break
        choices = list(pending(cfg, args.group))
        if args.dry_run:
            print(json.dumps({"host": args.host, "gpu": args.gpu, "group": args.group,
                "pending": [(t["arm"], t["checkpoint"], lane) for t, lane, _, _ in choices],
                "worker_sha256": digest}, indent=2), flush=True)
            return
        child = None
        admission_path = (lockpath.with_name(f'{args.host}_admission.lock')
                          if cfg.get('max_running_jobs') else lockpath)
        with admission_path.open("a") as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            uuid, used, total = gpu_memory(args.gpu)
            if cfg.get("gpu_uuids", {}).get(str(args.gpu), uuid) != uuid:
                raise RuntimeError("GPU identity changed")
            ram = next(int(s.split()[1]) / 1024**2 for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"))
            reservations = 0
            for other in (out / "workers").glob(f"{args.host}_{args.gpu}_*.json"):
                if other == state:
                    continue
                rec = json.loads(other.read_text())
                if rec.get("state") == "RUNNING" and time.time() - rec["started"] < 90:
                    reservations += rec["reserve_mib"]
            reserve = int(cfg.get(f"{args.group}_reserve_mib", 19000 if args.group == "dense" else 6000))
            running = 0
            if cfg.get('max_running_jobs'):
                for other in (out / 'workers').glob(f'{args.host}_*.json'):
                    rec = json.loads(other.read_text())
                    if rec.get('state') != 'RUNNING':
                        continue
                    try:
                        os.kill(int(rec['child_pid']), 0)
                        running += 1
                    except ProcessLookupError:
                        pass
            capacity = running < cfg.get('max_running_jobs', float('inf'))
            if choices and capacity and total - used - reservations >= reserve + 2048 and ram >= cfg.get("ram_reserve_gib", 48):
                target, lane, root, teacher = choices[0]
                ck, arm = target["checkpoint"], target["arm"]
                view = out / "snapshot_views" / arm / str(ck)
                point = view / f"training_{ck}"
                point.mkdir(parents=True, exist_ok=True)
                link = point / "teacher_checkpoint.pth"
                try:
                    link.symlink_to(teacher.resolve())
                except FileExistsError:
                    if link.resolve() != teacher.resolve():
                        raise RuntimeError(f"Snapshot view points to another teacher: {link}")
                jobs_cap = int(cfg.get(f'{args.group}_jobs_cap', 1))
                cmd = [cfg["python"], "-u", str(worker), "--repo", cfg["evaluator_repo"],
                    "--train-run", str(root / "source"), "--snapshot-root", str(view),
                    "--input-root", str(root / "adapters"), "--output-root", str(root / "results"),
                    "--benchmark-root", cfg["benchmark"], "--python-bin", cfg["python"],
                    "--gpu", uuid, "--worker", f"priority-{args.host}-{args.gpu}-{args.slot}",
                    "--official-epoch-length", str(target.get("official_epoch_length", 4098)),
                    "--effective-global-batch", str(target.get("effective_global_batch", 1024)),
                    "--full-eval-period", str(target.get("full_eval_period", 488)),
                    "--min-local-checkpoint-id", str(ck), "--expected-checkpoints", "1",
                    "--jobs-cap", str(jobs_cap), "--max-attempts", "1", "--poll-seconds", "30",
                    "--allow-busy-gpu", "--once", "--include-lanes", lane]
                env = dict(os.environ, PYTHONPATH=cfg["evaluator_repo"],
                    FROZEN_BATCH_SIZE=str(cfg.get('frozen_batch_size', 64)),
                    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                logpath = out / "logs" / f"{args.host}_{args.gpu}_{args.slot}_{arm}_{ck}_{lane}.log"
                logpath.parent.mkdir(parents=True, exist_ok=True)
                log = logpath.open("a")
                child = subprocess.Popen(cmd, cwd=cfg["evaluator_repo"], env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                save(state, dict(state="RUNNING", pid=os.getpid(), child_pid=child.pid, started=time.time(),
                    reserve_mib=reserve, gpu_uuid=uuid, arm=arm, checkpoint=ck, lane=lane, command=cmd,
                    evaluator_repo=cfg["evaluator_repo"], worker_sha256=digest, log=str(logpath)))
                print(json.dumps({"started": child.pid, "arm": arm, "checkpoint": ck, "lane": lane,
                    "host": args.host, "gpu": args.gpu}), flush=True)
        if child is not None:
            rc = child.wait()
            log.close()
            save(state, dict(state="IDLE", pid=os.getpid(), last_returncode=rc, updated=time.time()))
        else:
            save(state, dict(state="WAITING", pid=os.getpid(), pending=len(choices), updated=time.time(),
                gpu_used_mib=used, gpu_total_mib=total, ram_available_gib=ram))
        if args.once:
            break
        time.sleep(20)
    save(state, dict(state="DRAINED", pid=os.getpid(), updated=time.time()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--gpu", required=True, type=int)
    p.add_argument("--group", required=True, choices=("frozen", "dense"))
    p.add_argument("--slot", required=True)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    os.umask(0o000)  # Existing campaign state is shared across host-local users.
    run(p.parse_args())
