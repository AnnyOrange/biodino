#!/usr/bin/env python3
"""Admit the ten local 47,823-fork lw5 exports to the existing locked queue."""
import json
from pathlib import Path
import time

import manage_20tb_selected29_fill_20261021 as fill


def main():
    run = fill.RUNS / 'cooldown_20261013/20tb_distill_cooldown_len4880_from47823_lw5'
    config = run / 'config.yaml'
    assert config.is_file()
    admitted = []
    for checkpoint in range(48311, 52704, 488):
        path = run / 'eval' / f'training_{checkpoint}' / 'teacher_checkpoint.pth'
        assert path.is_file() and path.stat().st_size > 1_000_000, path
        asset = dict(arm='20tb_from47823_lw5', checkpoint_id=str(checkpoint),
                     path=str(path), config=str(config), kind='dinov3', model_id='', reserve_mib=4096)
        fill.admit('lw5', asset)
        admitted.append(dict(checkpoint=checkpoint, asset=asset,
                             checkpoint_validation='locked worker hashes and inspects before evaluation'))
    record = dict(updated_unix=time.time(), fork=47823, target=52703,
                  protocol='SELECTED29_LOCKED_E20_3SEEDS_20261019',
                  campaign=str(fill.BASE / 'lw5'), transfer='NONE_SHARED_LOCAL_CHECKPOINTS',
                  execution_hosts=['3090-qi', 'deepcad'], checkpoints=admitted, tasks=29 * len(admitted))
    fill.save(fill.BASE / 'early_lw5_admission.json', record)
    print(json.dumps(dict(admitted_checkpoints=len(admitted), tasks=record['tasks'], transfer=record['transfer'])))


if __name__ == '__main__':
    main()
