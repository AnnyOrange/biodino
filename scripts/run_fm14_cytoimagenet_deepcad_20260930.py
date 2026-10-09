#!/usr/bin/env python3
"""CytoImageNet MoNuSeg 30/7/14 needs more than a 24 GB card (OOM on cpu15 and, historically, on a
shared 32 GB card).  Run the unchanged FM14 campaign worker once on deepcad GPU 2 (A100 40 GB),
inside the campaign's recorded deepcad authorization (GPU indices 0-3)."""
import argparse
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("fm14", "/mnt/huawei_deepcad/dinov3/scripts/launch_monuseg_train30val7_fm14_20260930.py")
fm14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm14)
fleet = fm14.fleet_module()
fleet.worker(argparse.Namespace(output=fm14.OUTPUT, host="deepcad", gpus=[2], target_per_gpu=1, max_host_jobs=1,
                                max_global_jobs=400, task_family="mixed",
                                admission_guard=lambda gpu, actual, task: task["dataset"]["task"] == "segmentation"))
