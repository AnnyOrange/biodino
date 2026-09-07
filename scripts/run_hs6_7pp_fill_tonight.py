#!/usr/bin/env python3
"""D-scale NCT/CHAMMI/PCAM fill (8-pass and 1-pass), pinned to the train host.

Eval stays on the machine that trained the run. Do not rsync checkpoints.
Workers claim with mkdir locks under outputs/auto_eval_logs/hs6_7pp_fill_20260831/locks.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/mnt/huawei_deepcad/dinov3")
PYTHON = os.environ.get("PYTHON_BIN", "/home/lxy/miniconda3/envs/dinov3/bin/python")
BENCH = os.environ.get("BENCHMARK_ROOT", "/mnt/huawei_deepcad/benchmark")
LOG = ROOT / "outputs/auto_eval_logs/hs6_7pp_fill_20260831"
LOCK = LOG / "locks"
HOST = os.environ.get("HS6_HOST_TAG") or os.uname().nodename
GPUS = [int(x) for x in os.environ.get("HS6_GPUS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
BATCH_CAP = int(os.environ.get("HS6_BATCH_CAP", "0") or 0)
META = {"S+": ("Splus", "lr2e4", 64), "B": ("B", "lr1p5e4", 64), "L": ("L", "lr1e4", 32), "H+": ("Hplus", "lr5e5", 8)}
POOLS = [(10, 823), (20, 1639), (50, 4103), (100, 8199)]
EP1_POOLS = [(10, 102), (20, 204), (50, 512)]  # 1M 1-pass is e15 ck1024, not D-scale
E8_EVAL = "e8_full_20260820"
EP1_EVAL = "e8_ep1_20260827"
MIN_CKPT = {"S+": 1_000_000_000, "B": 2_000_000_000, "L": 5_000_000_000, "H+": 10_000_000_000}
# Train-host pin from D-scale claim owners. H+ 0.2/0.5/1M weights live on H100, not NFS.
DSCALE_ROLE = {
    ("S+", 10): "qi",
    ("S+", 20): "qi",
    ("S+", 50): "qi",
    ("S+", 100): "deepcad",
    ("B", 10): "qi",
    ("B", 20): "qi",
    ("B", 50): "qi",
    ("B", 100): "qi",
    ("L", 10): "5090",
    ("L", 20): "5090",
    ("L", 50): "5090",
    ("L", 100): "5090",
    ("H+", 10): "qi",
    ("H+", 20): "h100",
    ("H+", 50): "h100",
    ("H+", 100): "h100",
}
H100_RUN_ROOT = Path(os.environ.get("HS6_H100_RUN_ROOT", "/data_2/suxin/runs"))


def current_role() -> str:
    explicit = os.environ.get("HS6_EVAL_ROLE", "").strip().lower()
    if explicit:
        return explicit
    tag = HOST.lower()
    if "qi" in tag:
        return "qi"
    if "h100" in tag or "suxin" in tag:
        return "h100"
    if "deepcad" in tag:
        return "deepcad"
    if "xr" in tag or "lyx" in tag:
        return "xr"
    if "5090" in tag or tag in {"server", "local-8x5090-lxy"}:
        return "5090"
    return "none"


def dscale(nick: str, frac: int) -> Path:
    mk, lr, _bs = META[nick]
    name = f"HS6_Dscale_{mk}_robust_biosafe256_gb1024_{lr}_wu3_tw30_nosig_e8_random{frac}_seed0_20260820"
    if current_role() == "h100" and nick == "H+" and frac != 10:
        return H100_RUN_ROOT / name
    return ROOT / "outputs/01_training_runs" / name


def done_cls(out_dir: Path) -> bool:
    path = out_dir / "last_result.json"
    if not path.is_file() or path.stat().st_size <= 20:
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    return isinstance(data, dict) and not data.get("error") and "macro_f1" in data


def ckpt_ok(path: Path, nick: str) -> bool:
    if not path.is_file():
        return False
    return path.stat().st_size >= MIN_CKPT.get(nick, 1_000_000_000)


def nfs_role() -> bool:
    return current_role() in {"qi", "deepcad", "5090", "cpu", "nfs"}


def epoch_ckpts(train: Path, nick: str) -> list[int]:
    ck = train / "ckpt"
    if not ck.is_dir():
        return []
    out = []
    for path in ck.iterdir():
        if path.name.isdigit() and ckpt_ok(path / "checkpoint.pth", nick):
            out.append(int(path.name))
    return sorted(out)


def jobs() -> list[dict]:
    """All 8 epoch checkpoints of each D-scale 8-pass run, NCT + CHAMMI."""
    out = []
    datasets = ("nct-crc-he", "chammi-allen-task1")
    h100 = current_role() == "h100"
    for nick in ("S+", "B", "L", "H+"):
        _mk, _lr, bs = META[nick]
        for frac, _last in POOLS:
            if h100:
                if nick != "H+" or frac == 10:
                    continue
            elif not nfs_role():
                continue
            train = dscale(nick, frac)
            cfg = train / "config.yaml"
            if not cfg.is_file():
                continue
            for ckpt in epoch_ckpts(train, nick):
                ckpt_path = train / "ckpt" / str(ckpt) / "checkpoint.pth"
                for ds in datasets:
                    od = train / "eval" / E8_EVAL / "bio_classification" / ds / str(ckpt)
                    if done_cls(od):
                        continue
                    out.append(
                        {
                            "name": f"E8-{nick}-r{frac}-{ckpt}-{ds}",
                            "ckpt": ckpt_path,
                            "cfg": cfg,
                            "od": od,
                            "ds": ds,
                            "bs": bs,
                        }
                    )
    rank = {"nct-crc-he": 0, "chammi-allen-task1": 1}
    out.sort(key=lambda job: (rank.get(job["ds"], 9), job["name"]))
    return out


def try_claim(name: str) -> bool:
    path = LOCK / name
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        return False
    (path / "host").write_text(f"{HOST} pid={os.getpid()}\n", encoding="utf-8")
    return True


def release(name: str) -> None:
    shutil.rmtree(LOCK / name, ignore_errors=True)


def launch(job: dict, gpu: int) -> subprocess.Popen:
    job["od"].mkdir(parents=True, exist_ok=True)
    log = LOG / f"{job['name']}.{HOST}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    code = Path(os.environ.get("HS6_CODE_ROOT", str(ROOT)))
    env["PYTHONPATH"] = str(code)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cmd = [
        PYTHON,
        "-m",
        "dinov3.eval.bio_frozen_eval.run_classification",
        "--checkpoint",
        str(job["ckpt"]),
        "--train-config",
        str(job["cfg"]),
        "--benchmark-root",
        BENCH,
        "--datasets",
        job["ds"],
        "--output-dir",
        str(job["od"]),
        "--model-name",
        f"dinov3-{job['od'].name}",
        "--resolution-protocol",
        "best",
        "--image-size",
        "224",
        "--batch-size",
        str(min(job["bs"], BATCH_CAP) if BATCH_CAP > 0 else job["bs"]),
        "--num-workers",
        "2",
        "--channel-policy",
        "auto",
        "--split-protocol",
        "current",
        "--autocast-dtype",
        "bf16",
        "--no-save-features",
    ]
    handle = log.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(code),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job["_log"] = handle
    print(f"START host={HOST} gpu={gpu} {job['name']} pid={proc.pid}", flush=True)
    return proc


def main() -> None:
    global LOG, LOCK
    if not (ROOT / "scripts").is_dir():
        LOG = Path(os.environ.get("HS6_LOG", "/data_2/suxin/hs6_7pp_fill_logs"))
        LOCK = LOG / "locks"
    LOG.mkdir(parents=True, exist_ok=True)
    LOCK.mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple[subprocess.Popen, dict]] = {}
    fail = 0
    idle_rounds = 0
    print(
        f"worker host={HOST} role={current_role()} python={PYTHON} gpus={GPUS} batch_cap={BATCH_CAP or 'none'}",
        flush=True,
    )
    while True:
        for gpu, (proc, job) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            try:
                job["_log"].close()
            except OSError as exc:
                print(f"WARN close-log {job['name']}: {exc}", flush=True)
            status = "DONE" if rc == 0 else "FAIL"
            print(f"{status} host={HOST} gpu={gpu} {job['name']} rc={rc}", flush=True)
            if rc != 0:
                fail += 1
                release(job["name"])
            del running[gpu]

        free = [g for g in GPUS if g not in running]
        launched = 0
        if free:
            for job in jobs():
                if not free:
                    break
                if done_cls(job["od"]):
                    continue
                if not try_claim(job["name"]):
                    continue
                if done_cls(job["od"]):
                    release(job["name"])
                    continue
                gpu = free.pop(0)
                running[gpu] = (launch(job, gpu), job)
                launched += 1

        pending = jobs()
        if not running and not pending:
            print(f"all finished host={HOST} fail={fail}", flush=True)
            return
        if not running and launched == 0:
            idle_rounds += 1
            if idle_rounds % 15 == 1:
                claimed = sorted(p.name for p in LOCK.iterdir() if p.is_dir())
                print(
                    f"idle host={HOST} pending={len(pending)} claimed={len(claimed)}",
                    flush=True,
                )
        else:
            idle_rounds = 0
        time.sleep(20)


if __name__ == "__main__":
    main()
