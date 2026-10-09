"""Pinned bridge to the audited benchmark_model companion implementation."""
import os
from pathlib import Path
import sys

os.environ.setdefault('DINOV3_CODE_ROOT', str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, '/mnt/huawei_deepcad/benchmark_model')
from run_fm_companion_rules import main

if __name__ == '__main__':
    main()
