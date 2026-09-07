#!/usr/bin/env python3
"""Frozen eval of C-scale last ckpts only (e1/e2/e4/e8). Pin to the train host.

Run on the host that owns each checkpoint; batch size and model selection are configurable.
C-scale datasets: bloodmnist, tissuemnist, cyclops-protein-loc only.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROLE = os.environ.get("HS6_EVAL_ROLE", "").strip().lower()
HOST = os.environ.get("HS6_HOST_TAG") or os.uname().nodename
if not ROLE:
    tag = HOST.lower()
    if "hxw" in tag or "xzj" in tag:
        ROLE = "hxw"
    elif "xr" in tag or "lyx" in tag:
        ROLE = "xr"
    elif "h100" in tag or "suxin" in tag:
        ROLE = "h100"
    else:
        ROLE = "nfs"

if ROLE == "hxw":
    CODE = Path(os.environ.get("HS6_CODE_ROOT", "/home/xzj/biodino"))
    RUN_ROOT = Path(os.environ.get("HS6_RUN_ROOT", "/mnt/data/biodino_fixed_pass/outputs/01_training_runs"))
    PYTHON = os.environ.get("PYTHON_BIN", "/home/xzj/miniconda3/envs/dinov3/bin/python")
    BENCH = os.environ.get("BENCHMARK_ROOT", "/mnt/data/benchmark")
    DEFAULT_NICKS = "L"
elif ROLE == "xr":
    CODE = Path(os.environ.get("HS6_CODE_ROOT", "/data/xuzijing/biodino"))
    RUN_ROOT = Path(os.environ.get("HS6_RUN_ROOT", "/data/xuzijing/biodino/outputs/01_training_runs"))
    PYTHON = os.environ.get("PYTHON_BIN", "/home/server/miniconda3/envs/dinov3/bin/python")
    BENCH = os.environ.get("BENCHMARK_ROOT", "/data/xuzijing/chadavit_wds/benchmark")
    DEFAULT_NICKS = "S+,B"
elif ROLE == "h100":
    CODE = Path(os.environ.get("HS6_CODE_ROOT", "/data_2/suxin/biodino"))
    RUN_ROOT = Path(os.environ.get("HS6_RUN_ROOT", "/data_2/suxin/runs"))
    PYTHON = os.environ.get("PYTHON_BIN", "/data_2/suxin/envs/dinov3/bin/python")
    BENCH = os.environ.get("BENCHMARK_ROOT", "/data_2/suxin")
    DEFAULT_NICKS = "H+"
else:
    CODE = Path(os.environ.get("HS6_CODE_ROOT", "/mnt/huawei_deepcad/dinov3"))
    RUN_ROOT = Path(os.environ.get("HS6_RUN_ROOT", "/mnt/huawei_deepcad/dinov3/outputs/01_training_runs"))
    PYTHON = os.environ.get("PYTHON_BIN", "/home/lxy/miniconda3/envs/dinov3/bin/python")
    BENCH = os.environ.get("BENCHMARK_ROOT", "/mnt/huawei_deepcad/benchmark")
    DEFAULT_NICKS = "S+"

NICKS = [x.strip() for x in os.environ.get("HS6_CSCALE_NICK", DEFAULT_NICKS).split(",") if x.strip()]
NICK = NICKS[0]
NAME_SUBSTR = os.environ.get("HS6_NAME_SUBSTR", "prop15")
sys.path.insert(0, str(CODE))
os.environ["PYTHONPATH"] = str(CODE) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
os.environ.setdefault("PYTHON_BIN", PYTHON)

LOG = Path(os.environ.get("HS6_LOG", str(CODE / "outputs/auto_eval_logs/hs6_cscale_last_20260903")))
LOCK = LOG / "locks"
CSV_PATH = LOG / "kshot.csv"
GPUS = [int(x) for x in os.environ.get("HS6_GPUS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
SLOTS = int(os.environ.get("HS6_SLOTS_PER_GPU", "3"))
BATCH = int(os.environ.get("HS6_BATCH", "64"))
FREE_MIN = int(os.environ.get("HS6_FREE_MIN_MIB", "8000"))
EVAL_TAG = os.environ.get("HS6_CSCALE_EVAL_TAG", "cscale_prop15_20260904")
MIN_CKPT = 10_000_000
CSCALE_DATASETS = (
    "bloodmnist",
    "tissuemnist",
    "cyclops-protein-loc",
)
FULL_LINEAR = CSCALE_DATASETS
K10_ORDER = CSCALE_DATASETS
LAST = {1: 1024, 2: 2049, 4: 4099, 8: 8199}
EPOCH_OF = {1024: 1, 2049: 2, 4099: 4, 8199: 8}


def load_kshot():
    path = Path(os.environ.get("HS6_KSHOT_SCRIPT", str(CODE / "scripts/run_hs6_kshot_from_cache.py")))
    spec = importlib.util.spec_from_file_location("hs6_kshot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KSHOT = load_kshot()


def cls_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 20:
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    return isinstance(data, dict) and not data.get("error") and "macro_f1" in data


def nick_matches(name: str, nick: str) -> bool:
    if nick == "B":
        return "_B_" in name
    if nick == "L":
        return "_L_" in name
    if nick == "H+":
        return "Hplus" in name
    if nick == "S+":
        return "Splus" in name
    return False


def discover_runs() -> list[dict]:
    out = []
    for path in sorted(RUN_ROOT.glob("HS6_Cscale_*_e[1248]_*")):
        name = path.name
        nick = next((n for n in NICKS if nick_matches(name, n)), None)
        if nick is None:
            continue
        if NAME_SUBSTR and NAME_SUBSTR not in name:
            continue
        epochs = None
        for e in (1, 2, 4, 8):
            if f"_e{e}_" in name:
                epochs = e
                break
        if epochs is None:
            continue
        ckpt = LAST[epochs]
        ckpt_path = path / "ckpt" / str(ckpt) / "checkpoint.pth"
        cfg = path / "config.yaml"
        if not (ckpt_path.is_file() and ckpt_path.stat().st_size > MIN_CKPT and cfg.is_file()):
            continue
        out.append(
            {
                "epochs": epochs,
                "ckpt": ckpt,
                "ckpt_path": ckpt_path,
                "cfg": cfg,
                "train": path,
                "nick": nick,
            }
        )
    out.sort(key=lambda r: (r["nick"], r["epochs"]))
    return out


def kshot_done_set() -> set[tuple[str, str, str, str, str]]:
    need = {(str(k), str(s)) for k in (5, 10) for s in (0, 1, 2)}
    have: dict[tuple[str, str, str, str, str], set] = {}
    if not CSV_PATH.is_file():
        return set()
    with CSV_PATH.open() as handle:
        for row in csv.DictReader(handle):
            key = (
                row.get("model", ""),
                row.get("axis", ""),
                row.get("pool", ""),
                str(row.get("ckpt", "")),
                row.get("dataset", ""),
            )
            have.setdefault(key, set()).add((row.get("k", ""), row.get("seed", "")))
    return {key for key, got in have.items() if need <= got}


def features_exist(od: Path, dataset: str) -> bool:
    feat = od / "features" / dataset
    if not feat.is_dir():
        return False
    trains = list(feat.glob("*_train.npz"))
    tests = list(feat.glob("*_test.npz"))
    wholes = [p for p in feat.glob("*.npz") if "_train" not in p.name and "_test" not in p.name]
    return bool(trains and tests) or bool(wholes)


def jobs() -> list[dict]:
    done_k = kshot_done_set()
    out = []
    for run in discover_runs():
        ckpt = run["ckpt"]
        eval_dir = run["train"] / "eval" / EVAL_TAG
        nick = run["nick"]
        for ds_i, ds in enumerate(FULL_LINEAR):
            od = eval_dir / "bio_classification" / ds / str(ckpt)
            if cls_ok(od / "last_result.json"):
                continue
            out.append(
                {
                    "rank": ds_i,
                    "name": f"Cscale-{nick}-e{run['epochs']}-{ckpt}-{ds}-full",
                    "kind": "cls",
                    "save_feat": False,
                    "ckpt": run["ckpt_path"],
                    "cfg": run["cfg"],
                    "od": od,
                    "eval_dir": eval_dir,
                    "ds": ds,
                    "nick": nick,
                    "ckpt_id": ckpt,
                    "axis": "compute",
                    "pool": "1M",
                    "samples": 1_048_771,
                    "epoch": EPOCH_OF[ckpt],
                }
            )
        for ds_i, ds in enumerate(K10_ORDER):
            if (nick, "compute", "1M", str(ckpt), ds) in done_k:
                continue
            od = eval_dir / "bio_classification" / ds / str(ckpt)
            out.append(
                {
                    "rank": 100 + ds_i,
                    "name": f"Cscale-{nick}-e{run['epochs']}-{ckpt}-{ds}-k10",
                    "kind": "kshot",
                    "save_feat": True,
                    "ckpt": run["ckpt_path"],
                    "cfg": run["cfg"],
                    "od": od,
                    "eval_dir": eval_dir,
                    "ds": ds,
                    "nick": nick,
                    "ckpt_id": ckpt,
                    "axis": "compute",
                    "pool": "1M",
                    "samples": 1_048_771,
                    "epoch": EPOCH_OF[ckpt],
                }
            )
    out.sort(key=lambda job: (job["rank"], job["name"]))
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
        PYTHON, "-m", "dinov3.eval.bio_frozen_eval.run_classification",
        "--checkpoint", str(job["ckpt"]),
        "--train-config", str(job["cfg"]),
        "--benchmark-root", str(BENCH),
        "--datasets", job["ds"],
        "--output-dir", str(job["od"]),
        "--model-name", f"dinov3-{job['ckpt_id']}",
        "--resolution-protocol", "best",
        "--image-size", "224",
        "--batch-size", str(BATCH),
        "--num-workers", "2",
        "--channel-policy", "auto",
        "--split-protocol", "current",
        "--autocast-dtype", "bf16",
    ]
    if not job["save_feat"]:
        cmd.append("--no-save-features")
    else:
        # Full linear wrote summary.csv with --no-save-features into the same
        # output dir. Without overwrite, k10 extract skip-loops and never
        # writes npz, so CPU probe cannot run.
        cmd.append("--overwrite-results")
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
    loaded = KSHOT.load_xy(job["eval_dir"], job["ds"], job["ckpt_id"], cache_dir, Path(BENCH))
    if loaded is None:
        raise FileNotFoundError(f"no features for {job['name']}")
    x_train, y_train, x_test, y_test, feat_path = loaded
    import numpy as np

    y_tr = np.asarray(y_train).reshape(-1)
    done = KSHOT.existing_keys(CSV_PATH)
    for k in (5, 10):
        for seed in (0, 1, 2):
            key = (job["nick"], job["axis"], str(job["pool"]), str(job["ckpt_id"]), job["ds"], str(k), str(seed))
            if key in done:
                continue
            idx = KSHOT.kshot_indices(y_tr, k, seed)
            metrics = KSHOT.probe(x_train[idx], y_train[idx], x_test, y_test)
            full = KSHOT.full_macro_f1(job["eval_dir"], job["ds"], job["ckpt_id"])
            row = {
                "model": job["nick"],
                "axis": job["axis"],
                "pool": job["pool"],
                "samples": job["samples"],
                "ckpt": job["ckpt_id"],
                "epoch": job["epoch"],
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
    if job["kind"] == "cls":
        status = "DONE" if rc == 0 else "FAIL"
        print(f"{status} host={HOST} {job['name']} rc={rc}", flush=True)
        if rc != 0:
            release(job["name"])
        return
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
    runs = discover_runs()
    print(
        f"cscale-last host={HOST} role={ROLE} nicks={NICKS} python={PYTHON} "
        f"gpus={GPUS} slots={SLOTS} batch={BATCH} runs={len(runs)} datasets={FULL_LINEAR}",
        flush=True,
    )
    for run in runs:
        print(f"  run {run['nick']} e{run['epochs']} ckpt={run['ckpt']} {run['train'].name}", flush=True)
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
            if job["kind"] == "kshot" and features_exist(job["od"], job["ds"]):
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
    main()
