#!/usr/bin/env python3
"""Register resumed w=3 checkpoints with the existing hxw dense evaluator."""
import importlib.util
from pathlib import Path

ROOT = Path('/data/hs6_l5_v2_recovery_eval_20260930')
spec = importlib.util.spec_from_file_location(
    'v2_dense_seg', ROOT / 'bin/run_v2_v4_segmentation_hxw_dynamic_20261002.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
arm = ROOT / 'global_cls_w3'
points = {int(path.name) for path in (arm / 'adapters').iterdir()
          if path.name.isdigit() and (path / 'checkpoint.pth').is_file()}
module.CAMPAIGNS['v2_global_cls_w3'] = (arm, points)
module.main()
