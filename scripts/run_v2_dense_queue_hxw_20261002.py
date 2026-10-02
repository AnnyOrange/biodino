#!/usr/bin/env python3
"""Dense (v4 detection_b8 + v3 segmentation cells) evaluation queue for the Adaptive-v2 arms on hxw.

Replicates, cell for cell, the recipe behind the union-v4 no-GRAM continuation numbers
(/data/hs6_l_5tb_nogram_eval_20260921): detection = dinov3.eval.bio_detection.center_probe b8/224/5ep from the
pinned v3 source snapshot; segmentation = run_hplus_l_5tb_v4_monuseg_hxw_dynamic_20260928.py (copied with two
extra campaigns `v2_global` / `v2_global_local`). Memory-aware admission like run_hplus_new_id_queue_hxw_20260927.py.

  usage: python run_v2_dense_queue_hxw_20261002.py --gpus 0 1 2 3 [--per-gpu 4] [--arms global global_local]
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

E = Path("/data/hs6_l5_v2_recovery_eval_20260930")
PYTHON = "/home/xzj/eval_envs/hs6_protocol_v2/bin/python"
TRAIN_CONFIG = "/data/hs6_l_5tb_nogram_eval_20260921/source/config.yaml"   # original 5TB run (architecture)
DET_SOURCE = Path("/data/hs6_l_5tb_nogram_eval_20260921/bin/v3_source_snapshot")
SEG_RUNNER = E / "bin/run_v2_v4_segmentation_hxw_dynamic_20261002.py"
POINTS = [35135, 29767, 32207, 33671, 30743, 34647, 31231, 32695, 33183, 34159, 30255, 31719]  # endpoints first
DETECTION = ("conic", "bbbc038", "livecell")
PANNUKE = {"pannuke/fold1": "pannuke-fold1-train-fold2-val-fold3-test",
           "pannuke/fold2": "pannuke-fold2-train-fold1-val-fold3-test",
           "pannuke/fold3": "pannuke-fold3-train-fold2-val-fold1-test"}
SEGMENTATION = ("monuseg", "cellpose", "conic", "multimodal_cellseg", "pannuke/fold1", "pannuke/fold2",
                "pannuke/fold3", "tissuenet", "livecell")   # ascending duration (7 .. 93 min)
LIBS = "/home/xzj/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia"


def atomic(path, obj):
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n"); tmp.replace(path)


def seg_cell(arm, point, ds):
    dataset, split = ("pannuke", PANNUKE[ds]) if ds in PANNUKE else (
        ds, "official-baseline-fold0-nested-v1" if ds == "conic" else "formal-static-v1")
    return E / arm / "v3/cells" / f"point_{point}__{dataset}__{split}", dataset, split


def det_cell(arm, point, ds):
    return E / arm / "v4/detection_b8" / f"point_{point}" / ds


def valid(task):
    arm, point, fam, ds = task
    if fam == "segmentation":
        cell, _, _ = seg_cell(arm, point, ds)
        try:
            return json.loads((cell / "validation_report.json").read_text()).get("status") == "VALID_COMPLETE"
        except (OSError, ValueError):
            return False
    try:
        r = json.loads((det_cell(arm, point, ds) / "results_bio_detection.json").read_text())
        return (r.get("dataset") == ds and str(r.get("checkpoint")) == str(point) and r.get("batch_size") == 8
                and r.get("epochs") == 5 and r.get("image_size") == 224 and r.get("seed") == 0
                and r.get("test_patch_f1") is not None)
    except (OSError, ValueError):
        return False


def tasks(arms):
    for point in POINTS:
        for arm in arms:
            for ds in DETECTION:
                yield (arm, point, "detection", ds)
            for ds in SEGMENTATION:
                yield (arm, point, "segmentation", ds)


def vram_mib(task):
    return 8500 if task[2] == "segmentation" else 10000


def ram_gib(task):
    if task[2] != "segmentation":
        return 4
    if task[3] in ("livecell", "tissuenet"):
        return 32
    if task[3].startswith("pannuke/") or task[3] == "multimodal_cellseg":
        return 16
    return 8


def mem_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    raise RuntimeError("MemAvailable unavailable")


def gpu_free_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return [int(x.strip()) for x in out.splitlines()]


def launch(task, gpu, log_root):
    arm, point, fam, ds = task
    key = f"{arm}__{point}__{fam}__{ds.replace('/', '_')}"
    log = log_root / f"{key}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    env["LD_LIBRARY_PATH"] = f"{LIBS}/cuda_runtime/lib:{LIBS}/cuda_cupti/lib:" + env.get("LD_LIBRARY_PATH", "")
    if fam == "segmentation":
        cell, dataset, split = seg_cell(arm, point, ds)
        cmd = [PYTHON, "-u", str(SEG_RUNNER), "--campaign", f"v2_{arm}", "--point", str(point),
               "--dataset", dataset, "--gpu", str(gpu)]
        if ds in PANNUKE:
            cmd += ["--fold", PANNUKE[ds]]
        if cell.exists() and any(cell.iterdir()):
            cmd += ["--resume-existing"]
        cwd = E
    else:
        cell = det_cell(arm, point, ds); cell.mkdir(parents=True, exist_ok=True)
        checkpoint = str(E / arm / "adapters" / str(point) / "checkpoint.pth")
        cmd = [PYTHON, "-u", "-m", "dinov3.eval.bio_detection.center_probe",
               "--checkpoint", checkpoint, "--train-config", TRAIN_CONFIG,
               "--benchmark-root", "/data/benchmark", "--dataset", ds, "--output-dir", str(cell),
               "--batch-size", "8", "--num-workers", "2", "--image-size", "224", "--epochs", "5",
               "--lr", "0.001", "--autocast-dtype", "bf16", "--channel-policy", "auto",
               "--max-samples-per-split", "0", "--seed", "0",
               "--conic-split-protocol", "official-baseline-fold0-nested-v1"]
        env["PYTHONPATH"] = str(DET_SOURCE); cwd = DET_SOURCE
        atomic(cell / "command_manifest.json", {"protocol_id": "bio-eval-union-v4", "campaign": f"v2_{arm}",
               "point": point, "dataset": ds, "gpu": gpu, "batch_size": 8, "epochs": 5, "command": cmd,
               "train_config": TRAIN_CONFIG, "checkpoint": checkpoint,
               "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    with log.open("ab") as stream:
        child = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    print(dt.datetime.now().strftime("%H:%M"), "START", gpu, key, child.pid, flush=True)
    return {"child": child, "task": task, "key": key, "gpu": gpu, "started": time.time(), "log": log}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--per-gpu", type=int, default=4)
    p.add_argument("--arms", nargs="+", default=["global", "global_local"])
    p.add_argument("--min-ram-gib", type=float, default=100)
    a = p.parse_args()
    log_root = E / "logs/dense"; log_root.mkdir(parents=True, exist_ok=True)
    state = E / "logs/dense_queue_status.json"
    attempts = {}
    active = {}
    while True:
        for key, job in list(active.items()):
            rc = job["child"].poll()
            if rc is None:
                continue
            ok = rc == 0 and valid(job["task"])
            print(dt.datetime.now().strftime("%H:%M"), "DONE" if ok else "FAILED", job["key"], "rc", rc,
                  "%.0f min" % ((time.time() - job["started"]) / 60), flush=True)
            if not ok:
                attempts[job["key"]] = attempts.get(job["key"], 0) + 1
            del active[key]
        pending = [t for t in tasks(a.arms) if not valid(t)
                   and attempts.get(f"{t[0]}__{t[1]}__{t[2]}__{t[3].replace('/', '_')}", 0) < 2
                   and f"{t[0]}__{t[1]}__{t[2]}__{t[3].replace('/', '_')}" not in active]
        if not pending and not active:
            print("ALL DONE", flush=True); atomic(state, {"state": "done", "utc": dt.datetime.now(dt.timezone.utc).isoformat()})
            return
        free = gpu_free_mib()
        for gpu in a.gpus:
            running = [j for j in active.values() if j["gpu"] == gpu]
            if len(running) >= a.per_gpu or not pending:
                continue
            loading = sum(vram_mib(j["task"]) for j in running if time.time() - j["started"] < 60)
            loading_ram = sum(ram_gib(j["task"]) for j in active.values() if time.time() - j["started"] < 120)
            for t in pending:
                if free[gpu] - loading < vram_mib(t) + 2000:
                    break
                if mem_available_gib() - loading_ram - ram_gib(t) < a.min_ram_gib:
                    continue
                job = launch(t, gpu, log_root); active[job["key"]] = job
                pending.remove(t); free[gpu] -= vram_mib(t); loading += vram_mib(t)
                break
        atomic(state, {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "active": [j["key"] for j in active.values()],
                       "pending": len(pending), "failed_twice": [k for k, v in attempts.items() if v >= 2]})
        time.sleep(10)


if __name__ == "__main__":
    main()
