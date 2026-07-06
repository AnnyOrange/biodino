"""
Run zero-shot classic Cellpose (cyto3 / nuclei) + Omnipose on all datasets across
a GPU fleet, scored with the SAME metrics. Invoke with the cp3 env python
(cellpose 3.x + omnipose). Complements the cpsam baseline.
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

PY = sys.executable
ROOT = "/mnt/huawei_deepcad/dinov3"
SEG = "/mnt/huawei_deepcad/benchmark/segmentation"
DR = {"pannuke": "pannuke/extracted", "tissuenet": "tissuenet/extracted", "conic": "conic/extracted",
      "bbbc038": "bbbc038/extracted", "livecell": "LIVECell", "monuseg": "monuseg/extracted"}
CAP = {"pannuke": 300, "tissuenet": 300, "conic": 300, "bbbc038": 200, "livecell": 120, "monuseg": 60}
NUCLEI = {"pannuke", "conic", "bbbc038", "monuseg"}  # nucleus datasets → also run 'nuclei'


def build_jobs(datasets, with_omni=False):
    jobs = []  # (dataset, model, omni, tag)
    for ds in datasets:
        jobs.append((ds, "cyto3", False, f"cellpose_cyto3/{ds}"))
        if ds in NUCLEI:
            jobs.append((ds, "nuclei", False, f"cellpose_nuclei/{ds}"))
        if with_omni:
            jobs.append((ds, "cyto2_omni", True, f"omnipose/{ds}"))
    return jobs


def cmd(ds, model, omni, tag, gpu):
    c = [PY, "-m", "dinov3.eval.bio_segmentation.instance_seg.scripts.run_specialist",
         "--dataset", ds, "--data-root", os.path.join(SEG, DR[ds]), "--split", "val",
         "--model", model, "--gpu", "--max-images", str(CAP[ds]),
         "--output-dir", f"outputs/instance_seg/spec/{tag}"]
    if omni:
        c.append("--omni")
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DR))
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--with-omni", action="store_true")
    a = p.parse_args()
    os.chdir(ROOT)
    gpus = [g for g in a.gpus.split(",") if g]
    jobs = [j for j in build_jobs(a.datasets, with_omni=a.with_omni)
            if not (Path("outputs/instance_seg/spec") / j[3] / "results.json").exists()]
    print(f"[spec] {len(jobs)} jobs on {gpus}", flush=True)
    running, free, results = {}, list(gpus), {}
    while jobs or running:
        while free and jobs:
            gpu = free.pop(0); ds, model, omni, tag = jobs.pop(0)
            od = Path("outputs/instance_seg/spec") / tag; od.mkdir(parents=True, exist_ok=True)
            lf = open(str(od) + ".log", "w")
            pr = subprocess.Popen(cmd(ds, model, omni, tag, gpu),
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), stdout=lf, stderr=subprocess.STDOUT)
            running[gpu] = (pr, tag, lf)
            print(f"[spec] launch {tag} GPU{gpu} pid {pr.pid}", flush=True)
        time.sleep(10)
        for gpu, (pr, tag, lf) in list(running.items()):
            if pr.poll() is None:
                continue
            lf.close(); del running[gpu]; free.append(gpu)
            rj = Path("outputs/instance_seg/spec") / tag / "results.json"
            results[tag] = json.load(open(rj)).get("val") if rj.exists() else None
            print(f"[spec] done {tag}: {'ok' if results[tag] else 'FAIL '+str(pr.returncode)}", flush=True)
    print("[spec] all done", flush=True)


if __name__ == "__main__":
    main()
