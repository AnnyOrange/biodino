#!/usr/bin/env python3
"""Keep six independent H+ ID tests on each explicitly assigned free GPU."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/data/hs6_hplus_5tb_eval_20260921')
ADMIN = ROOT / 'new_checkpoint_queue_20260927'
OUTPUT = ROOT / 'continuation_v4/nonseg'
REPO = Path('/home/xzj/biodino_eval_git_20260917')
PYTHON = '/home/xzj/eval_envs/hs6_protocol_v2/bin/python'
GROUPS = {
 'classification_a': 'bloodmnist pathmnist tissuemnist breastmnist organamnist organcmnist organsmnist',
 'classification_b': 'dermamnist octmnist pneumoniamnist retinamnist chestmnist bbbc048-cellcycle',
 'classification_c': 'cyclops-protein-loc midog25-atypical pcam nct-crc-he lc25000 chammi-allen-task1',
 'classification_d': 'chammi-allen-task2 chammi-cp-task1 chammi-cp-task2 chammi-cp-task3 chammi-hpa-task1 chammi-hpa-task2',
 'regression': 'bbbc013 bbbc005 conic-cell-count livecell-cell-count',
 'retrieval': 'lc25000 nct-crc-he-100 nct-crc-he-1k crc-val-he-7k hpa-subcellular rxrx1-cross',
}
SMALL = 'organamnist organcmnist organsmnist pathmnist dermamnist pneumoniamnist tissuemnist retinamnist octmnist breastmnist pcam nct-crc-he'.split()
LARGE = {'bloodmnist', 'chestmnist', 'bbbc048-cellcycle', 'cyclops-protein-loc', 'midog25-atypical'}
PANNUKE = {
 'pannuke/fold1': 'pannuke-fold1-train-fold2-val-fold3-test',
 'pannuke/fold2': 'pannuke-fold2-train-fold1-val-fold3-test',
 'pannuke/fold3': 'pannuke-fold3-train-fold2-val-fold1-test',
}
SEGMENTATION = ('cellpose', 'conic', 'livecell', 'monuseg', 'multimodal_cellseg',
                'pannuke/fold1', 'pannuke/fold2', 'pannuke/fold3', 'tissuenet')
SEGMENTATION_PRIORITY = ('cellpose', 'conic', 'monuseg', 'multimodal_cellseg',
                         'pannuke/fold1', 'pannuke/fold2', 'pannuke/fold3',
                         'tissuenet', 'livecell')
DETECTION = ('bbbc038', 'conic', 'livecell')
RXRX_BASE = Path('/data/hs6_5tb_v4_rxrx3_20260924')
DETECTION_SOURCE = Path('/data/hs6_l_5tb_nogram_eval_20260921/bin/v3_source_snapshot')
RXRX_SOURCE = Path('/data/hs6_l_5tb_nogram_eval_20260921/bin/v4_monuseg_source_snapshot_20260924')


def atomic(path, obj):
 temp = path.with_suffix('.tmp')
 temp.write_text(json.dumps(obj, indent=2) + '\n'); temp.replace(path)


def sha256(path):
 result = hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda: stream.read(8 << 20), b''): result.update(block)
 return result.hexdigest()


def tasks():
 points = sorted((int(p.parent.name) for p in (ROOT / 'adapters').glob('*/checkpoint.pth')
                  if p.parent.name.isdigit() and
                  (int(p.parent.name) < 13663 or (p.parent / 'verified_sha256.json').exists())), reverse=True)
 cells = [(group, ds) for group, names in GROUPS.items() for ds in names.split()]
 cells.sort(key=lambda x: SMALL.index(x[1]) if x[1] in SMALL else len(SMALL))
 # Small datasets across all staged checkpoints provide steady fill slots.
 for group, ds in cells:
  for point in points:
   yield point, group, ds
 # Continue with the remaining ID families when the flat tests have drained.
 for point in points:
  for ds in DETECTION: yield point, 'detection_proxy', ds
  yield point, 'rxrx3', 'rxrx3-core'
 for ds in SEGMENTATION_PRIORITY:
  for point in points: yield point, 'segmentation', ds


def paths(task):
 point, group, ds = task
 if group == 'detection_proxy':
  return ROOT / 'v4/detection_b8' / f'point_{point}' / ds, ADMIN / 'claims' / f'{point}__{group}__{ds}'
 if group == 'rxrx3':
  return RXRX_BASE / 'hplus' / f'point_{point}' / 'models' / f'hplus_{point}', ADMIN / 'claims' / f'{point}__{group}__{ds}'
 if group == 'segmentation':
  dataset, split = ('pannuke', PANNUKE[ds]) if ds in PANNUKE else (ds, 'official-baseline-fold0-nested-v1' if ds == 'conic' else 'formal-static-v1')
  return ROOT / 'v3/cells' / f'point_{point}__{dataset}__{split}', ADMIN / 'claims' / f'{point}__{group}__{ds.replace("/", "_")}'
 family = group.split('_')[0]
 folder = OUTPUT / f'point_{point}' / group / f'bio_{family}' / ds / str(point)
 return folder, ADMIN / 'claims' / f'{point}__{group}__{ds}'


def valid(folder, family):
 try:
  obj = json.loads((folder / 'last_result.json').read_text())
  rows = obj.get('rows', [obj])
  keys = {'classification': ('accuracy', 'macro_auc'), 'regression': ('mae',),
          'retrieval': ('recall_at_1',)}[family]
  return any(any(row.get(k) is not None for k in keys) for row in rows)
 except (OSError, ValueError): return False


def valid_task(task):
 folder, _ = paths(task)
 if task[1] == 'segmentation':
  try: return json.loads((folder / 'validation_report.json').read_text()).get('status') == 'VALID_COMPLETE'
  except (OSError, ValueError): return False
 if task[1] == 'detection_proxy':
  try:
   result = json.loads((folder / 'results_bio_detection.json').read_text())
   return (result.get('dataset') == task[2] and str(result.get('checkpoint')) == str(task[0])
           and result.get('batch_size') == 8 and result.get('epochs') == 5
           and result.get('image_size') == 224 and result.get('seed') == 0
           and result.get('test_patch_f1') is not None)
  except (OSError, ValueError): return False
 if task[1] == 'rxrx3':
  try:
   result = json.loads((folder / 'results.json').read_text())
   test = result['tests']['rxrx3']
   return (result.get('status') == 'VALID_COMPLETE' and result.get('model') == f'hplus_{task[0]}'
           and result.get('batch_size') == 64 and result.get('teacher_branch') == 'teacher'
           and test.get('status') == 'FORMAL' and test.get('proxy') is False
           and test.get('protocol_id') == 'crispr-query-guide-plate-disjoint-all-eligible-genes-v1'
           and test.get('n_query') == 734 and test.get('n_gallery') == 734
           and all(test.get(k) is not None for k in ('recall_at_1', 'mrr', 'nmi')))
  except (OSError, ValueError, KeyError): return False
 family = task[1].split('_')[0]
 if valid(folder, family): return True
 if task[0] < 13663:
  return valid(ROOT / 'old' / folder.relative_to(OUTPUT), family)
 return False


def reservation_mib(task):
 if task[1] == 'segmentation': return 8500
 if task[1] in ('detection_proxy', 'rxrx3'): return 10000
 return 8500 if task[2] in LARGE else 5000


def required_free_mib(task):
 return reservation_mib(task) + (1500 if reservation_mib(task) > 5000 else 2000)


def is_large(task):
 return reservation_mib(task) > 5000


def ram_reservation_gib(task):
 if task[1] != 'segmentation': return 0
 if task[2] in ('livecell', 'tissuenet'): return 32
 if task[2].startswith('pannuke/') or task[2] == 'multimodal_cellseg': return 16
 return 8


def mem_available_gib():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'): return int(line.split()[1]) / 1024**2
 raise RuntimeError('MemAvailable unavailable')


def is_oom(folder, log_path=None):
 try:
  error = str(json.loads((folder / 'failed_result.json').read_text()).get('error', ''))
 except (OSError, ValueError):
  try:
   with (log_path or folder / 'queue_test.log').open('rb') as log:
    log.seek(0, os.SEEK_END); log.seek(max(0, log.tell() - 8192))
    error = log.read().decode(errors='replace')
  except OSError: return False
 return 'out of memory' in error.lower() or 'cudaerrormemoryallocation' in error.lower()


def launch(task, gpu, slot):
 point, group, ds = task; family = group.split('_')[0]
 folder, claim = paths(task)
 checkpoint = str(ROOT / 'adapters' / str(point) / 'checkpoint.pth')
 if group == 'segmentation':
  runner = ROOT / 'bin/run_hplus_l_5tb_v4_monuseg_hxw_dynamic_20260928.py'
  command = [PYTHON, '-u', str(runner), '--campaign', 'hplus', '--point', str(point),
             '--dataset', 'pannuke' if ds in PANNUKE else ds, '--gpu', str(gpu)]
  if ds in PANNUKE: command += ['--fold', PANNUKE[ds]]
  if folder.exists() and any(folder.iterdir()): command += ['--resume-existing']
  cwd = ROOT; source = RXRX_SOURCE
 elif group == 'detection_proxy':
  folder.mkdir(parents=True, exist_ok=True)
  command = [PYTHON, '-u', '-m', 'dinov3.eval.bio_detection.center_probe',
    '--checkpoint', checkpoint, '--train-config', str(ROOT / 'source/config.yaml'),
    '--benchmark-root', '/data/benchmark', '--dataset', ds, '--output-dir', str(folder),
    '--batch-size', '8', '--num-workers', '2', '--image-size', '224', '--epochs', '5',
    '--lr', '0.001', '--autocast-dtype', 'bf16', '--channel-policy', 'auto',
    '--max-samples-per-split', '0', '--seed', '0',
    '--conic-split-protocol', 'official-baseline-fold0-nested-v1']
  cwd = DETECTION_SOURCE; source = DETECTION_SOURCE
  atomic(folder / 'command_manifest.json', {'protocol_id': 'bio-eval-union-v4',
    'campaign': 'hplus', 'point': point, 'dataset': ds, 'gpu': gpu,
    'batch_size': 8, 'epochs': 5, 'command': command,
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
 elif group == 'rxrx3':
  cell = folder.parent.parent
  cell.mkdir(parents=True, exist_ok=True)
  prior = folder / 'results.json'
  if prior.exists() and not valid_task(task):
   prior.rename(folder / f'results.previous_failed_{int(time.time())}.json')
  cache = RXRX_BASE / 'cache/rxrx3-core'
  protocol = RXRX_BASE / 'protocol_v4.json'
  locked = json.loads(protocol.read_text())['retrieval_splits']['rxrx3-core']['manifest_sha256']
  if sha256(cache / 'split_manifest.jsonl') != locked:
   raise RuntimeError('RxRx3 split manifest differs from v4 lock')
  atomic(cell / 'campaign_manifest.json', {'protocol_id': 'bio-eval-union-v4',
    'protocol_sha256': sha256(protocol),
    'evaluator_sha256': sha256(RXRX_BASE / 'scripts/run_external4_fixedbudget_model.py'),
    'dataset_sha256': {name: sha256(cache / name) for name in
      ('metadata.json', 'split_manifest.jsonl', 'images.npy', 'labels.npy')},
    'campaign': 'hplus', 'point': point, 'checkpoint': checkpoint,
    'checkpoint_sha256': sha256(checkpoint),
    'train_config': str(ROOT / 'source/config.yaml'),
    'train_config_sha256': sha256(ROOT / 'source/config.yaml'),
    'gpu': gpu, 'formal_batch_size': 64, 'inference_microbatch': 32,
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
  command = [PYTHON, '-u', str(RXRX_BASE / 'scripts/run_external4_fixedbudget_model.py'),
    '--model', f'hplus_{point}', '--campaign', str(cell),
    '--cache-root', str(RXRX_BASE / 'cache'), '--datasets', 'rxrx3',
    '--checkpoint', checkpoint, '--train-config', str(ROOT / 'source/config.yaml'),
    '--device', 'cuda:0', '--batch-size', '32', '--logical-batch-size', '64']
  cwd = RXRX_SOURCE; source = RXRX_SOURCE
 else:
  folder.mkdir(parents=True, exist_ok=True)
  module = 'run_retrieval_clustering' if family == 'retrieval' else 'run_classification'
  command = [PYTHON, '-u', '-m', f'dinov3.eval.bio_frozen_eval.{module}',
    '--checkpoint', checkpoint, '--train-config', str(ROOT / 'source/config.yaml'),
    '--benchmark-root', '/data/benchmark', '--datasets', ds, '--output-dir', str(folder),
    '--model-name', f'dinov3-{point}', '--n-last-blocks', '1', '--autocast-dtype', 'bf16',
    '--batch-size', '64', '--num-workers', '1', '--seed', '0', '--channel-policy', 'auto',
    '--channel-tta-samples', '8', '--channel-policy-seed', '0']
  if family == 'classification': command += ['--train-fraction', '0.8', '--split-protocol', 'current', '--resolution-protocol', 'best', '--image-size', '224']
  if family == 'regression': command += ['--resolution-protocol', 'best', '--image-size', '224']
  cwd = REPO; source = REPO
 env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 env['PYTHONPATH'] = str(source)
 libs = '/home/xzj/miniconda3/envs/dinov3/lib/python3.11/site-packages/nvidia'
 env['LD_LIBRARY_PATH'] = f'{libs}/cuda_runtime/lib:{libs}/cuda_cupti/lib:' + env.get('LD_LIBRARY_PATH', '')
 (ADMIN / 'logs').mkdir(exist_ok=True)
 log = (ADMIN / 'logs' / (claim.name + '.log')) if group in ('segmentation', 'detection_proxy', 'rxrx3') else (folder / 'queue_test.log')
 with log.open('ab') as stream:
  child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
 started_epoch = time.time()
 atomic(claim / 'owner.json', {'pid': child.pid, 'gpu': gpu, 'slot': slot, 'task': task,
   'command': command, 'started_epoch': started_epoch})
 print('START', gpu, slot, task, child.pid, flush=True)
 return {'child': child, 'task': task, 'folder': folder, 'claim': claim,
   'gpu': gpu, 'slot': slot, 'started_epoch': started_epoch, 'log': log}


class ExistingChild:
 def __init__(self, pid, point): self.pid, self.point = pid, point
 def poll(self):
  try:
   proc = Path('/proc') / str(self.pid)
   command = (proc / 'cmdline').read_bytes().decode(errors='replace')
   state = (proc / 'stat').read_text().split(') ')[1].split()[0]
   matches = (f'/adapters/{self.point}/checkpoint.pth' in command
              or f'--point\x00{self.point}\x00' in command)
   return None if state != 'Z' and matches else 0
  except OSError: return 0


def main():
 parser = argparse.ArgumentParser(); parser.add_argument('--gpus', nargs='+', type=int, required=True)
 parser.add_argument('--slots', type=int, default=6); args = parser.parse_args()
 for name in ('claims', 'failures'): (ADMIN / name).mkdir(parents=True, exist_ok=True)
 import fcntl
 with (ADMIN / 'pool.lock').open('w') as lock:
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  active = {}
  for claim in (ADMIN / 'claims').iterdir():
   owner_path = claim / 'owner.json'
   try:
    owner = json.loads(owner_path.read_text()); task = tuple(owner['task'])
    child = ExistingChild(owner['pid'], task[0])
    if child.poll() is None:
     folder, _ = paths(task)
     active[(owner['gpu'], owner['slot'])] = {'child': child, 'task': task,
       'folder': folder, 'claim': claim, 'gpu': owner['gpu'], 'slot': owner['slot'],
       'started_epoch': owner.get('started_epoch', 0),
       'log': (ADMIN / 'logs' / (claim.name + '.log')) if task[1] in ('segmentation', 'detection_proxy', 'rxrx3') else folder / 'queue_test.log'}
    else: owner_path.unlink(); claim.rmdir()
   except (OSError, ValueError, KeyError): continue
  underfilled_since = {}
  last_underfilled_log = {}
  last_backlog_scan = 0
  backlog = {}
  while True:
   for key, job in list(active.items()):
    rc = job['child'].poll()
    if rc is None: continue
    failure_path = ADMIN / 'failures' / (job['claim'].name + '.json')
    if rc != 0 or not valid_task(job['task']):
     try: previous = json.loads(failure_path.read_text())
     except (OSError, ValueError): previous = {}
     attempts = previous.get('attempts', 1) + 1
     oom = is_oom(job['folder'], job['log'])
     atomic(failure_path, {'returncode': rc, 'task': job['task'],
       'log': str(job['log']),
       'attempts': attempts, 'retryable_oom': oom,
       'retry_after_epoch': time.time() + (min(1800, 300 * 2 ** min(attempts - 2, 3)) if oom else 300)})
     shutil.rmtree(job['folder'] / 'features', ignore_errors=True)
    else:
     print('DONE', job['task'], flush=True)
     shutil.rmtree(job['folder'] / 'features', ignore_errors=True)
     failure_path.unlink(missing_ok=True)
    (job['claim'] / 'owner.json').unlink(missing_ok=True); job['claim'].rmdir(); del active[key]
   just_launched_mib = {}
   for gpu in args.gpus:
    for slot in range(args.slots):
     if (gpu, slot) in active: continue
     if shutil.disk_usage('/data').free < 80 * 2**30: continue
     # Three batch-64 large tests use about 24.6 GiB on a 32 GiB 5090.
     # The former one-large limit left a GPU at only 25% after small jobs ended.
     large_active = sum(j['gpu'] == gpu and is_large(j['task']) for j in active.values())
     memory_rows = subprocess.check_output(
       ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
     loading_reserve = sum(reservation_mib(j['task'])
       for j in active.values() if j['gpu'] == gpu and time.time() - j.get('started_epoch', 0) < 45)
     free_mib = int(memory_rows.splitlines()[gpu].strip()) - loading_reserve - just_launched_mib.get(gpu, 0)
     for task in tasks():
      folder, claim = paths(task)
      if valid_task(task): continue
      if task[1] == 'segmentation':
       if sum(j['task'][1] == 'segmentation' for j in active.values()) >= 28: continue
       if sum(j['gpu'] == gpu and j['task'][1] == 'segmentation' for j in active.values()) >= 6: continue
       loading_ram = sum(ram_reservation_gib(j['task']) for j in active.values()
         if time.time() - j.get('started_epoch', 0) < 120)
       if mem_available_gib() - loading_ram - ram_reservation_gib(task) < 120: continue
      elif is_large(task) and large_active >= 3: continue
      if free_mib < required_free_mib(task): continue
      failure_path = ADMIN / 'failures' / (claim.name + '.json')
      if failure_path.exists():
       try: failure = json.loads(failure_path.read_text())
       except (OSError, ValueError): failure = {'attempts': 3}
       log_path = Path(failure.get('log', str(folder / 'queue_test.log')))
       if (not failure.get('retryable_oom', is_oom(folder, log_path)) and failure.get('attempts', 1) >= 3) or time.time() < failure.get('retry_after_epoch', 0): continue
      try: claim.mkdir()
      except FileExistsError: continue
      try: active[(gpu, slot)] = launch(task, gpu, slot)
      except Exception as exc:
       atomic(failure_path, {'task': task, 'attempts': 3, 'launch_error': repr(exc),
                             'retryable_oom': False, 'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
       claim.rmdir()
       print('LAUNCH_FAILED', task, repr(exc), flush=True)
       continue
      # nvidia-smi lags model loading; reserve memory for this new process now.
      just_launched_mib[gpu] = just_launched_mib.get(gpu, 0) + reservation_mib(task)
      break
   memory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
   now = time.time()
   if now - last_backlog_scan >= 60:
    backlog = {}
    for task in tasks():
     if not valid_task(task): backlog[task[1]] = backlog.get(task[1], 0) + 1
    last_backlog_scan = now
   underfilled = {}
   for row in memory.splitlines():
    gpu, used, total = (int(x.strip()) for x in row.split(','))
    if gpu not in args.gpus: continue
    if used < 0.5 * total:
     start = underfilled_since.setdefault(gpu, now)
     underfilled[str(gpu)] = {'used_mib': used, 'total_mib': total, 'seconds': round(now-start, 1)}
     if now - start >= 30 and now - last_underfilled_log.get(gpu, 0) >= 60:
      print('UNDERFILLED', gpu, used, total, 'seconds', round(now-start, 1), flush=True)
      last_underfilled_log[gpu] = now
    else: underfilled_since.pop(gpu, None)
   atomic(ADMIN / 'status.json', {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
      'gpus': args.gpus, 'target_tests_per_gpu': args.slots,
      'active_tests': [{'gpu':j['gpu'], 'slot':j['slot'], 'pid':j['child'].pid, 'task':j['task']} for j in active.values()],
      'gpu_memory': memory, 'underfilled_gpus': underfilled,
      'pending_by_group': backlog, 'pending_total': sum(backlog.values())})
   time.sleep(5)


if __name__ == '__main__': main()
