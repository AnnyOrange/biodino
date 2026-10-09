#!/usr/bin/env python3
"""On 3090-qi: await index, audit it, then launch the authorized B arm."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/mnt/huawei_deepcad/dinov3')
INDEX = Path('/home/bbnc/20tb_resampling_20261009/index')
PYTHON = '/home/bbnc/anaconda3/envs/dinov3/bin/python'
GROUP = ROOT / 'outputs/01_training_runs/hs6_l_20tb_v2_recovery_fork38063_20261009'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--builder-pid', type=int, required=True)
    args = parser.parse_args()
    while not (INDEX / 'manifest.json').exists():
        os.kill(args.builder_pid, 0)
        time.sleep(20)
    print('Index complete; running actual decoder and mixture preflight', flush=True)
    subprocess.run([PYTHON, '-u', str(ROOT / 'scripts/preflight_20tb_source_sampler_20261009.py'),
                    '--manifest', str(INDEX / 'manifest.json')], check=True, cwd=ROOT)
    assert json.loads((INDEX / 'manifest.json').read_text())['status'] == 'PASS'
    audit = ROOT / 'outputs/00_reports/20tb_resampling_adaptive_v2_plan_20261009/actual_source_index'
    audit.mkdir(exist_ok=True)
    for path in INDEX.iterdir():
        if path.suffix in {'.json', '.csv', '.parquet'}:
            shutil.copyfile(path, audit / path.name)
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    used = [int(x) for x in memory.splitlines()]
    if len(used) != 8 or max(used) >= 512:
        raise RuntimeError('3090-qi GPUs acquired another job before launch: ' + str(used))
    run = GROUP / 'resampling_only_38063_8x3090qi'
    if (run / 'supervisor.pid').exists():
        raise FileExistsError('Supervisor already registered')
    env = dict(os.environ, REPO=str(ROOT), PYTHON_BIN=PYTHON, OUTPUT_DIR=str(run),
               LOG=str(run / 'console.log'), SOFT_GB='350', HARD_GB='100', RESTART_DELAY='30', MAX_FAILURES='3',
               LAUNCHER=str(ROOT / 'scripts/launch_20tb_resampling38063_3090qi_20261009.sh'))
    with (run / 'supervisor.log').open('ab') as log:
        proc = subprocess.Popen(['bash', str(ROOT / 'scripts/watch_hs6_l_20tb_route2_autorestart_20260927.sh')],
                                env=env, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                start_new_session=True)
    (run / 'supervisor.pid').write_text(str(proc.pid)+'\n')
    path = run / 'launch_manifest_20261009.json'
    manifest = json.loads(path.read_text())
    manifest.update(status='LAUNCHED_AWAITING_FIRST_UPDATE', launched=time.time(), supervisor_pid=proc.pid,
                    preflight=str(audit / 'PREFLIGHT.json'), sampler_manifest=str(audit / 'manifest.json'))
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(status='LAUNCHED', supervisor_pid=proc.pid, run=str(run))), flush=True)


if __name__ == '__main__':
    main()
