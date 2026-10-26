#!/usr/bin/env python3
"""Validate misplaced paired cells, then resume workers with an absolute output path."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
ROOT = REPO / 'outputs/02_eval_runs/v2_20tb_paired_20261006'
MISPLACED = Path('/mnt/huawei_deepcad/dinov3_20tb_online_snapshot_20260918/outputs/02_eval_runs/v2_20tb_paired_20261006/cells')
STATE = ROOT / '_state'
PYTHON = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'


def main():
    while list((STATE / 'running').glob('*.json')):
        print('WAIT', len(list((STATE / 'running').glob('*.json'))), flush=True)
        time.sleep(30)
    sys.path.insert(0, str(REPO / 'scripts'))
    import run_v2_progress_fleet_20261006 as campaign

    manifest = json.loads((ROOT / 'campaign_manifest.json').read_text())
    tasks = {task['key']: task for task in manifest['tasks']}
    archive = ROOT / 'attempt_archive/20261014_relative_output'
    archive.mkdir(parents=True, exist_ok=True)
    for source in MISPLACED.iterdir():
        if not source.is_dir() or source.name not in tasks:
            continue
        key = source.name
        target = ROOT / 'cells' / key
        result = source / 'last_result.json'
        if result.is_file():
            shutil.copy2(result, target / 'last_result.json')
            invocation = json.loads((target / 'invocation_manifest.json').read_text())
            try:
                report = campaign.validate_cell(tasks[key], target, invocation)
                campaign.save(STATE / 'done' / f'{key}.json', report)
                print('VALID_COMPLETE', key, flush=True)
                continue
            except Exception as exc:
                print('RETRY', key, str(exc), flush=True)
        claim = STATE / 'claims' / key
        if claim.exists():
            shutil.move(str(claim), str(archive / f'claim_{key}'))
    pause = STATE / 'PAUSED.json'
    if pause.exists():
        shutil.move(str(pause), str(archive / 'PAUSED.json'))
    for host in ('cpu1', 'cpu5', 'cpu12'):
        log = ROOT / 'logs' / f'{host}.worker.absolute_20261014.log'
        command = (f'cd {REPO} && nohup {PYTHON} -u scripts/run_v2_progress_fleet_20261006.py '
                   f'worker --output {ROOT} --host {host} --gpus 0 --target-per-gpu 5 '
                   f'--max-host-jobs 5 --max-global-jobs 30 > {log} 2>&1 < /dev/null &')
        subprocess.run(['ssh', '-n', '-f', '-o', 'BatchMode=yes', host, command], check=True)
        print('RESTART', host, flush=True)


if __name__ == '__main__':
    main()
