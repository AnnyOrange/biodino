#!/usr/bin/env python3
"""Verify transferred L assets, then hand off to the checkpoint supervisor."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SNAPSHOT = Path('/home/xzj/biodino_l20tb_domain_snapshot_20261018')
OUTPUT = Path('/home/xzj/storage/merged/l20tb_complement_bio5tb_from47823_len19520_hxw_20261019')
PYTHON = '/home/xzj/miniconda3/envs/dinov3/bin/python'
ASSETS = {
    'checkpoint': (OUTPUT / 'ckpt/49775/checkpoint.pth', 7118413459,
                   'e16c1394b4ac97f6ec1e962bb1c14f29a82df53d116756066e1adbf81d528121'),
    'bio5tb': (OUTPUT / 'teachers/bio5tb_19519.pth', 1402008175,
               'd3ac51352dff7f387f62aa9c03ea087937ff6cfebcc58210287afb9d08899c10'),
}
STATE = OUTPUT / 'asset_launch_state.json'
SEVEN_B_RUN = 'vit7b_20tb_recovered249_20261018'


def save(status, **fields):
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(status=status, updated_unix=time.time(), **fields), indent=2) + '\n')
    temp.replace(STATE)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def available_ram_gib():
    line = next(line for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))
    return int(line.split()[1]) / 1024**2


def seven_b_active():
    processes = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    return any(SEVEN_B_RUN in line and
               ('dinov3/train/train.py' in line or 'resume_7b_20tb_20261018.py' in line)
               for line in processes.splitlines())


def main():
    while True:
        sizes = {name: path.stat().st_size if path.exists() else 0
                 for name, (path, _, _) in ASSETS.items()}
        oversized = [name for name, (_, size, _) in ASSETS.items() if sizes[name] > size]
        if oversized:
            raise RuntimeError(f'Transferred assets exceed expected size: {oversized}')
        if all(sizes[name] == size for name, (_, size, _) in ASSETS.items()):
            break
        save('WAITING_FOR_ASSETS', bytes_received=sizes)
        time.sleep(60)
    save('VERIFYING_ASSETS', bytes_received=sizes)
    actual = {name: sha256(path) for name, (path, _, _) in ASSETS.items()}
    mismatch = [name for name, (_, _, digest) in ASSETS.items() if actual[name] != digest]
    if mismatch:
        save('HASH_MISMATCH', sha256=actual, mismatched=mismatch)
        raise RuntimeError(f'Transferred assets failed SHA256: {mismatch}')
    config = OUTPUT / 'candidate_hxw.yaml'
    subprocess.run([PYTHON, str(SNAPSHOT / 'scripts/prepare_hxw_l20tb_complement_20261019.py'),
                    '--source-config', str(OUTPUT / 'source_candidate.yaml'),
                    '--output-config', str(config), '--output-dir', str(OUTPUT),
                    '--hoptimus', '/home/xzj/storage/merged/cooldown_20261013/teachers/H-optimus-0',
                    '--dinov3l', '/home/xzj/route2_20tb_20261006/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
                    '--bio5tb', str(ASSETS['bio5tb'][0])], check=True)
    while True:
        ram = available_ram_gib()
        active_7b = seven_b_active()
        if ram >= 100 and not active_7b:
            break
        save('WAITING_FOR_HOST_MEMORY', sha256=actual, available_ram_gib=ram,
             seven_b_active=active_7b, required_ram_gib=100)
        time.sleep(60)
    save('ASSETS_VERIFIED_STARTING', sha256=actual, config=str(config))
    sys.stdout.flush()
    os.execv(PYTHON, [PYTHON, '-u', str(SNAPSHOT / 'scripts/launch_l20tb_domain_20261018.py'),
                      '--snapshot', str(SNAPSHOT), '--output', str(OUTPUT), '--python', PYTHON,
                      '--gpus', '4,5,6,7', '--port', '29725', '--config', str(config)])


if __name__ == '__main__':
    main()
