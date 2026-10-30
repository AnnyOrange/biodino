"""User-reserved devices are excluded from every campaign scheduling path."""
import json
import time
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'outputs/02_eval_runs/unified_paper'
_CACHE={}

def cached_policy(name):
    path=OUT/name;now=time.monotonic();prior=_CACHE.get(path)
    if prior is not None and now-prior[0]<5:return prior[1]
    try:data=json.loads(path.read_text())
    except FileNotFoundError:data={}
    _CACHE[path]=(now,data)
    return data

def worker_disabled(ident):
    return ident in cached_policy('disabled_worker_slots.json')

def excluded(host,gpu):
    # Pending queues can inspect thousands of tasks in a pass. Reading this
    # shared NFS file for every candidate creates a metadata storm. New child
    # processes still read fresh policy at their own admission boundary.
    policy=cached_policy('gpu_exclusions.json')
    if any(host.startswith(prefix) for prefix in policy.get('disabled_host_prefixes', [])):
        return True
    return int(gpu)in policy.get('hosts',{}).get(host,{}).get('gpus',[])
