#!/usr/bin/env python3
"""Resume shared evaluation on deepcad with its user cgroup and future RAM accounted for."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fill', HERE / 'run_v2_progress_fleet_fill_20261006.py')
fill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill)
OUT = fill.OUTPUT
CGROUP = Path(f'/sys/fs/cgroup/memory/user.slice/user-{os.getuid()}.slice')
GIB = 1024 ** 3
_cache = (0, {})


def budget():
    global _cache
    if time.time() - _cache[0] < 3:
        return _cache[1]
    limit = int((CGROUP / 'memory.limit_in_bytes').read_text()) / GIB
    used = int((CGROUP / 'memory.usage_in_bytes').read_text()) / GIB
    stats = dict((k, int(v)) for k, v in (line.split() for line in (CGROUP / 'memory.stat').read_text().splitlines()))
    # Clean inactive file pages are reclaimable; anonymous training buffers,
    # active cache, dirty pages and writeback remain charged against the budget.
    reclaimable = max(0, stats.get('total_inactive_file', 0) - stats.get('total_dirty', 0)
                      - stats.get('total_writeback', 0)) / GIB
    ram = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) / 1024**2
    app = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    ready = {int(p):float(m) for line in app.splitlines() if len((parts:=line.split(','))) == 2 for p,m in [parts] if m.strip().isdigit() and float(m)>1024}
    ps = subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
    candidates = {}
    for line in ps.splitlines()[1:]:
        parts=line.strip().split(None,2)
        if len(parts)==3 and '-m dinov3.eval.' in parts[2] and str(OUT/'cells')+'/' in parts[2]:
            candidates[int(parts[0])] = int(parts[1])
    parents = [pid for pid,parent in candidates.items() if parent not in candidates]
    pss = 0
    for pid in candidates:
        try:
            pss += int(next(l.split()[1] for l in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines() if l.startswith('Pss:'))) / 1024**2
        except (OSError, StopIteration):
            continue
    unready = sum(pid not in ready for pid in parents)
    # Reserve initialization until the evaluator actually has a loaded CUDA model.
    # Keep an absolute eval budget for the training shuffle buffers still warming up.
    future = 8 * unready + 2 * (len(parents)-unready)
    status = dict(time=time.time(),cgroup_limit_gib=limit,cgroup_used_gib=used,
                  reclaimable_clean_inactive_file_gib=reclaimable,
                  effective_headroom_gib=min(ram,limit-used+reclaimable),eval_pss_gib=pss,
                  unready=unready,active_parent_processes=len(parents),future_eval_gib=future,
                  eval_budget_gib=80,min_cgroup_headroom_gib=64)
    status['admit'] = status['effective_headroom_gib'] >= 64 + 8*unready and pss + future + 8 <= 80
    fill.save(OUT/'_state/workers/deepcad_memory_guard.json',status)
    _cache=(time.time(),status)
    return status


if __name__ == '__main__':
    os.umask(0o000)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--status',action='store_true');a=p.parse_args()
    if a.status:
        print(json.dumps(budget(),indent=2))
    else:
        manifest=json.loads((OUT/'campaign_manifest.json').read_text())
        for path,digest in manifest['external_source_hashes'].items():
            if fill.sha256(path) != digest:raise RuntimeError('Unregistered source change: '+path)
        args=argparse.Namespace(output=OUT,host='deepcad',gpus=list(range(8)),target_per_gpu=12,
            max_host_jobs=24,max_global_jobs=160,task_family='frozen',memory_target=.75,
            admission_guard=lambda gpu,actual,task: budget()['admit'])
        fill.worker(args)
