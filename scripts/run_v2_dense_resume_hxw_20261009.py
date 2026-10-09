#!/usr/bin/env python3
"""Keep hxw's existing v4 dense queue supplied with resumed w=3 checkpoints."""
import importlib.util
from pathlib import Path
import time

ROOT = Path('/data/hs6_l5_v2_recovery_eval_20260930')
spec = importlib.util.spec_from_file_location(
    'v2_dense_queue', ROOT / 'bin/run_v2_dense_queue_hxw_20261002.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.SEG_RUNNER = ROOT / 'bin/run_v2_seg_w3_hxw_20261009.py'
module.SEGMENTATION = tuple(dataset for dataset in module.SEGMENTATION
                            if dataset != 'monuseg')
original_tasks = module.tasks


def tasks(arms):
    adapters = ROOT / 'global_cls_w3/adapters'
    module.POINTS = sorted((int(path.name) for path in adapters.iterdir()
                            if path.name.isdigit() and 35623 <= int(path.name) <= 50263
                            and (path / 'checkpoint.pth').is_file()), reverse=True)
    yield from original_tasks(arms)


module.tasks = tasks
while True:
    if list(tasks(['global_cls_w3'])):
        module.main()
    time.sleep(120)
