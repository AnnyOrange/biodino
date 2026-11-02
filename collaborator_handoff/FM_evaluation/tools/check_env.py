#!/usr/bin/env python3
import argparse,importlib,importlib.metadata,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--cpu-only',action='store_true');a=p.parse_args()
required=json.loads((Path(__file__).resolve().parents[1]/'environment_versions.json').read_text());errors=[];versions={}
for package,version in required.items():
 try:
  actual=importlib.metadata.version(package);versions[package]=actual
  if actual.split('+')[0]!=version.split('+')[0]:errors.append(f'{package}: {actual} != {version}')
 except Exception as e:errors.append(f'{package}: {e}')
for name in ['torch','torchvision','numpy','scipy','sklearn','pandas','tifffile','imagecodecs','omegaconf','timm','pyarrow','skimage','numba','networkx','h5py','PIL','cv2','einops','matplotlib']:
 try:importlib.import_module(name)
 except Exception as e:errors.append(f'import {name}: {e}')
if not a.cpu_only:
 try:
  import torch
  if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
  free,total=torch.cuda.mem_get_info()
  if free<80*2**30 or free<total*.2:raise RuntimeError('Need at least 80 GiB free GPU memory for the 7B acceptance baseline')
  x=torch.ones((16,16),device='cuda',dtype=torch.bfloat16);assert torch.isfinite(x@x).all();torch.cuda.synchronize()
  del x;torch.cuda.empty_cache()
  from pathlib import Path
  available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
  for base in [Path('/sys/fs/cgroup')]:
   limit=base/'memory.max';used=base/'memory.current'
   if limit.is_file() and limit.read_text().strip()!='max':
    stats=dict(line.split() for line in (base/'memory.stat').read_text().splitlines())
    working_set=max(0,int(used.read_text())-int(stats.get('inactive_file',0)))
    available=min(available,int(limit.read_text())-working_set)
  if available<128*2**30:raise RuntimeError('Need at least 128 GiB currently available host/cgroup RAM for the serial 7B baseline')
 except Exception as e:errors.append(str(e))
report=dict(status='FAIL' if errors else 'CPU_ONLY_PASS' if a.cpu_only else 'ENVIRONMENT_PASS',versions=versions,errors=errors)
print(json.dumps(report,indent=2));sys.exit(1 if errors else 0)
