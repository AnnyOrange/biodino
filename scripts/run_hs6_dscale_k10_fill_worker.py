#!/usr/bin/env python3
"""Fill missing D-scale e8-last 10-shot. S+/B/L on nfs. Do not copy ckpts.

H+ 0.1M stays on H100 and is not launched from this role=nfs process.
Batch 64. Extract features, probe k=5/10 seeds 0/1/2, delete npz.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOST = os.environ.get("HS6_HOST_TAG") or os.uname().nodename
CODE = Path(os.environ.get("HS6_CODE_ROOT", "/mnt/huawei_deepcad/dinov3"))
RUN_ROOT = Path(os.environ.get("HS6_RUN_ROOT", str(CODE / "outputs/01_training_runs")))
PYTHON = os.environ.get("PYTHON_BIN", "/home/lxy/miniconda3/envs/dinov3/bin/python")
BENCH = Path(os.environ.get("BENCHMARK_ROOT", "/mnt/huawei_deepcad/benchmark"))
sys.path.insert(0, str(CODE))
os.environ["PYTHONPATH"] = str(CODE) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

LOG = Path(os.environ.get("HS6_LOG", str(CODE / "outputs/auto_eval_logs/hs6_dscale_k10_fill_20260904")))
LOCK = LOG / "locks"
CSV_PATH = LOG / "kshot.csv"
GPUS = [int(x) for x in os.environ.get("HS6_GPUS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
SLOTS = int(os.environ.get("HS6_SLOTS_PER_GPU", "2"))
BATCH = int(os.environ.get("HS6_BATCH", "64"))
FREE_MIN = int(os.environ.get("HS6_FREE_MIN_MIB", "8000"))
EVAL_TAG = "dscale_k10_fill_20260904"

NICKDIR = {"S+": "Splus", "B": "B", "L": "L", "H+": "Hplus"}
LR = {"S+": "lr2e4", "B": "lr1p5e4", "L": "lr1e4", "H+": "lr5e5"}
RAND = {"0.1M": "random10", "0.2M": "random20", "0.5M": "random50", "1M": "random100"}
LAST = {"0.1M": 823, "0.2M": 1639, "0.5M": 4103, "1M": 8199}
SAMPLES = {"0.1M": 104_877, "0.2M": 209_754, "0.5M": 524_385, "1M": 1_048_771}
NICK_RANK = {"S+": 0, "B": 1, "L": 2, "H+": 3}
DS_RANK = {"nct-crc-he": 0, "chammi-allen-task1": 1, "chammi-allen-task2": 2}

# role=nfs: S+/B/L only. H+ 0.1M stays on H100.
GAPS = (
    *[("S+", p, "nct-crc-he") for p in LAST],
    *[("B", p, "nct-crc-he") for p in LAST],
    *[("L", p, "nct-crc-he") for p in LAST],
    ("B", "0.1M", "chammi-allen-task1"),
    ("B", "1M", "chammi-allen-task1"),
    *[("S+", p, "chammi-allen-task2") for p in LAST],
    *[("B", p, "chammi-allen-task2") for p in LAST],
    *[("L", p, "chammi-allen-task2") for p in LAST],
)
LOCAL_NICKS = {"S+", "B", "L"}


def load_kshot():
    path = CODE / "scripts/run_hs6_kshot_from_cache.py"
    spec = importlib.util.spec_from_file_location("hs6_kshot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KSHOT = load_kshot()


def run_dir(nick: str, pool: str) -> Path:
    return (
        RUN_ROOT
        / (
            f"HS6_Dscale_{NICKDIR[nick]}_robust_biosafe256_gb1024_{LR[nick]}"
            f"_wu3_tw30_nosig_e8_{RAND[pool]}_seed0_20260820"
        )
    )


def already_have_k10(nick: str, pool: str, ds: str, ckpt: int) -> bool:
    if CSV_PATH.is_file():
        keys = KSHOT.existing_keys(CSV_PATH)
        need = {(nick, "data", pool, str(ckpt), ds, "10", s) for s in ("0", "1", "2")}
        if need <= keys:
            return True
    kdir = CODE / "plot/fig2/kshot"
    seeds = set()
    for path in (
        kdir / "kshot_merged.csv",
        kdir / "kshot_fill_20260901.csv",
        kdir / "kshot_h100.csv",
        kdir / "kshot_raw.csv",
        CSV_PATH,
    ):
        if not path.is_file():
            continue
        import csv

        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("model") == nick
                    and row.get("dataset") == ds
                    and row.get("axis") == "data"
                    and str(row.get("k")) == "10"
                    and str(row.get("ckpt")) == str(ckpt)
                    and row.get("error_macro_f1") not in (None, "")
                ):
                    seeds.add(str(row.get("seed", "")))
        if {"0", "1", "2"} <= seeds:
            return True
    return False


def features_exist(od: Path, dataset: str) -> bool:
    feat = od / "features" / dataset
    if not feat.is_dir():
        return False
    trains = list(feat.glob("*_train.npz"))
    tests = list(feat.glob("*_test.npz"))
    wholes = [p for p in feat.glob("*.npz") if "_train" not in p.name and "_test" not in p.name]
    return bool(trains and tests) or bool(wholes)


def jobs() -> list[dict]:
    out = []
    for nick, pool, ds in GAPS:
        if nick not in LOCAL_NICKS:
            continue
        ckpt = LAST[pool]
        if already_have_k10(nick, pool, ds, ckpt):
            continue
        train = run_dir(nick, pool)
        ckpt_path = train / "ckpt" / str(ckpt) / "checkpoint.pth"
        cfg = train / "config.yaml"
        if not ckpt_path.is_file() or not cfg.is_file():
            print(f"SKIP missing ckpt/cfg {nick} {pool} {ds} {ckpt_path}", flush=True)
            continue
        eval_dir = train / "eval" / EVAL_TAG
        od = eval_dir / "bio_classification" / ds / str(ckpt)
        out.append(
            {
                "rank": (DS_RANK[ds], NICK_RANK[nick], ckpt),
                "name": f"Dscale-{nick}-{pool}-{ckpt}-{ds}-k10",
                "ckpt": ckpt_path,
                "cfg": cfg,
                "od": od,
                "eval_dir": eval_dir,
                "ds": ds,
                "nick": nick,
                "ckpt_id": ckpt,
                "pool": pool,
                "samples": SAMPLES[pool],
            }
        )
    out.sort(key=lambda job: job["rank"])
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


def gpu_free_mib(gpu: int) -> int:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(float(raw.strip().split()[0]))
    except Exception:
        return 0


def launch(job: dict, gpu: int) -> subprocess.Popen:
    job["od"].mkdir(parents=True, exist_ok=True)
    log = LOG / f"{job['name']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(CODE)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["OMP_NUM_THREADS"] = "4"
    cmd = [
        PYTHON,
        "-m",
        "dinov3.eval.bio_frozen_eval.run_classification",
        "--checkpoint",
        str(job["ckpt"]),
        "--train-config",
        str(job["cfg"]),
        "--benchmark-root",
        str(BENCH),
        "--datasets",
        job["ds"],
        "--output-dir",
        str(job["od"]),
        "--model-name",
        f"dinov3-{job['ckpt_id']}",
        "--resolution-protocol",
        "best",
        "--image-size",
        "224",
        "--batch-size",
        str(BATCH),
        "--num-workers",
        "2",
        "--channel-policy",
        "auto",
        "--split-protocol",
        "current",
        "--autocast-dtype",
        "bf16",
        "--overwrite-results",
    ]
    handle = log.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(CODE),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job["_log"] = handle
    print(f"START host={HOST} gpu={gpu} {job['name']} pid={proc.pid} bs={BATCH}", flush=True)
    return proc


def run_kshot(job: dict) -> None:
    cache_dir = LOG / "splits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    loaded = KSHOT.load_xy(job["eval_dir"], job["ds"], job["ckpt_id"], cache_dir, BENCH)
    if loaded is None:
        raise FileNotFoundError(f"no features for {job['name']}")
    x_train, y_train, x_test, y_test, feat_path = loaded
    import numpy as np

    y_tr = np.asarray(y_train).reshape(-1)
    done = KSHOT.existing_keys(CSV_PATH)
    for k in (5, 10):
        for seed in (0, 1, 2):
            key = (job["nick"], "data", str(job["pool"]), str(job["ckpt_id"]), job["ds"], str(k), str(seed))
            if key in done:
                continue
            idx = KSHOT.kshot_indices(y_tr, k, seed)
            metrics = KSHOT.probe(x_train[idx], y_train[idx], x_test, y_test)
            full = KSHOT.full_macro_f1(job["eval_dir"], job["ds"], job["ckpt_id"])
            row = {
                "model": job["nick"],
                "axis": "data",
                "pool": job["pool"],
                "samples": job["samples"],
                "ckpt": job["ckpt_id"],
                "epoch": 8,
                "params": KSHOT.PARAMS[job["nick"]],
                "dataset": job["ds"],
                "k": k,
                "seed": seed,
                **metrics,
                "full_macro_f1": "" if full is None else f"{full:.8f}",
                "feature_train": feat_path,
            }
            KSHOT.append_row(CSV_PATH, row)
    feat_dir = job["od"] / "features"
    if feat_dir.is_dir():
        shutil.rmtree(feat_dir, ignore_errors=True)
    print(f"KSHOT {job['name']}", flush=True)


def finish_job(job: dict, rc: int) -> None:
    try:
        job["_log"].close()
    except Exception:
        pass
    if rc != 0 and not features_exist(job["od"], job["ds"]):
        print(f"FAIL host={HOST} {job['name']} rc={rc}", flush=True)
        release(job["name"])
        return
    try:
        run_kshot(job)
        print(f"DONE host={HOST} {job['name']} rc={rc}", flush=True)
        release(job["name"])
    except Exception as exc:
        print(f"KSHOT_FAIL host={HOST} {job['name']} {exc}", flush=True)
        release(job["name"])


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    LOCK.mkdir(parents=True, exist_ok=True)
    pending0 = jobs()
    print(
        f"dscale-k10-fill host={HOST} python={PYTHON} gpus={GPUS} slots={SLOTS} "
        f"batch={BATCH} pending={len(pending0)}",
        flush=True,
    )
    for job in pending0:
        print(f"  job {job['name']}", flush=True)
    running: dict[int, tuple[subprocess.Popen, dict, int]] = {}
    load = {g: 0 for g in GPUS}
    next_id = 0
    fail = 0
    while True:
        for key, (proc, job, gpu) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            load[gpu] = max(0, load[gpu] - 1)
            finish_job(job, rc)
            if rc != 0:
                fail += 1
            del running[key]
        pending = jobs()
        launched = 0
        for job in pending:
            if features_exist(job["od"], job["ds"]):
                if not try_claim(job["name"] + "-kshot"):
                    continue
                try:
                    run_kshot(job)
                except Exception as exc:
                    print(f"KSHOT_FAIL {job['name']} {exc}", flush=True)
                release(job["name"] + "-kshot")
                continue
            for gpu in GPUS:
                if load[gpu] >= SLOTS:
                    continue
                if gpu_free_mib(gpu) < FREE_MIN:
                    continue
                if not try_claim(job["name"]):
                    break
                running[next_id] = (launch(job, gpu), job, gpu)
                next_id += 1
                load[gpu] += 1
                launched += 1
                break
        pending = jobs()
        if not running and not pending:
            print(f"all finished host={HOST} fail={fail}", flush=True)
            return
        if not running and launched == 0:
            print(f"idle host={HOST} pending={len(pending)}", flush=True)
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
