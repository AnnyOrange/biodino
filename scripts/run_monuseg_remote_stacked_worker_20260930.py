#!/usr/bin/env python3
"""Extra worker slot for a remote MoNuSeg 30/7/14 site campaign so several tests share one GPU.

User instruction 2026-09-30: a GPU counts as properly used only when stacked tests push memory
above 50%.  The snapshot queue lets a dense cell start only on a GPU with no other project test;
each extra slot (distinct worker name, own host lock) ignores other slots' tests, while keeping the
queue's per-task memory reserve check (16000 MiB for MoNuSeg) and shared claims, so tasks are never
duplicated and the numerical path of every cell is unchanged.
"""
import argparse
import importlib.util
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--site", required=True)
p.add_argument("--slot", required=True)
a = p.parse_args()
here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("remote", here / "launch_monuseg_train30val7_remote_20260930.py")
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)
site = json.loads(Path(a.site).read_text())
fleet = remote.fleet_module(site)
fleet.queue.project_tests = lambda: {}
gpus = [int(g) for g in site["gpus"]]
fleet.worker(argparse.Namespace(output=Path(site["output"]), host=f"{site['name']}-{a.slot}", gpus=gpus,
                                target_per_gpu=1, max_host_jobs=len(gpus), max_global_jobs=400,
                                task_family="segmentation",
                                admission_guard=lambda gpu, actual, task: task["dataset"]["task"] == "segmentation"))
