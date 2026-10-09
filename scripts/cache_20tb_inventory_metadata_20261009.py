#!/usr/bin/env python3
"""Cache only historic, inactive SQLite metadata; never copy tar payloads."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923/route2_full_random_shuffle_candidate_20260927')
DEST = Path('/home/bbnc/20tb_resampling_20261009/reuse_cache')


def copy(path):
    dest = DEST / path.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.with_suffix('.ready').exists():
        return
    files = [path]
    wal = Path(str(path) + '-wal')
    if wal.exists():
        files.append(wal)
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    for source in files:
        shutil.copyfile(source, dest.parent / source.name)
    after = [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    if before != after:
        raise RuntimeError('Inventory modified during copy: ' + str(path))
    dest.with_suffix('.ready').write_text(json.dumps(before))
    print(json.dumps(dict(cached=str(dest), bytes=sum(s[0] for s in before))), flush=True)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy, sorted(ROOT.glob('*/work/worker_*/*.sqlite'))))
