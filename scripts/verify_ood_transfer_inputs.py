#!/usr/bin/env python3
"""Verify every resident input against the portable manifest before scheduling."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3.eval.ood_transfer.common import process_policy, verify_inputs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest', required=True)
parser.add_argument('--data-root', required=True)
parser.add_argument('--receipt', required=True)
args = parser.parse_args()
process_policy()
print(verify_inputs(args.manifest, args.data_root, args.receipt))
