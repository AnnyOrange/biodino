"""Approved independent frozen component queue; never produces full-v3 aggregates."""
from __future__ import annotations
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREADS = {name: '1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
GROUP_HASHES = {
    'cyclops-protein-loc': '886c960016d30fe011a45d8fbf94520fae97f8d7b4b8dcc287ea9087859b9d8c',
    'bbbc048-cellcycle': 'c8ea224712cc6e0564c36cebc28d33ce11fb659afb82568b25e1c5114d703f31',
    'midog25-atypical': 'f5c1806316e1cbfa45888059cf1ff1394723bd0fb03c7a1089d3c4ebe159011d',
    'bbbc005': '6bcfd65a7bd38e9a2e919f850409bc5c59eb59b6beece3ef62ce5d15840575b4',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args], text=True).strip()


def identities(dataset, split):
    import numpy as np
    if hasattr(dataset, 'samples'):
        records = [(str(s.image_path), float(s.target)) if hasattr(s,'image_path')
                   else (str(s[0]), int(s[1])) for s in dataset.samples]
        labels = np.asarray([row[1] for row in records])
    elif hasattr(dataset, 'index'):
        records = [(str(dataset.files[fi])+f':{ri}', int(dataset.labels[i]))
                   for i,(fi,ri) in enumerate(dataset.index)]
        labels = dataset.labels
    elif hasattr(dataset, 'path'):
        labels = dataset.labels
        records = [(str(dataset.path)+f':{split}:{i}', np.asarray(label).tolist()) for i,label in enumerate(labels)]
    else:
        raise ValueError(f'Unsupported identity audit: {type(dataset).__name__}')
    return records, labels


def dataset_preflight(name, task, benchmark):
    import numpy as np
    from dinov3.eval.bio_frozen_eval.registry import build_dataset, NATIVE_TEST_SPLIT_DATASETS
    from dinov3.eval.bio_frozen_eval.run_classification import split_protocol_for_dataset, resolve_image_size, resolve_dataset_resize_size
    from dinov3.eval.bio_frozen_eval.group_keys import GROUP_SPLIT_DATASETS, group_keys
    from dinov3.eval.bio_frozen_eval.make_group_splits import group_split_indices, split_path
    dataset, actual_task = build_dataset(name, 'train', None, None, benchmark_root=benchmark)
    train, labels = identities(dataset, 'train')
    all_records = list(train)
    source_files = list(getattr(dataset, 'files', []))
    hashes = {}
    for attribute in ('path','meta_path','csv_path'):
        if hasattr(dataset, attribute):
            path = Path(getattr(dataset, attribute))
            hashes[str(path)] = sha256(path)
    if name == 'bbbc013':
        if len(dataset) != 96: raise ValueError('BBBC013 requires all 96 wells')
        counts = dict(n_train=36,n_test=96)
        split_record = dict(protocol=split_protocol_for_dataset(name), samples=train,
                            n_compounds=2,n_folds=8,fold_train=36,fold_test=12)
        hashes[str(dataset.root/'BBBC013_v1_platemap_all.txt')] = sha256(dataset.root/'BBBC013_v1_platemap_all.txt')
    elif name in NATIVE_TEST_SPLIT_DATASETS:
        test_ds, _ = build_dataset(name, 'test', None, None, benchmark_root=benchmark)
        test, test_labels = identities(test_ds, 'test')
        all_records.extend(test)
        source_files.extend(getattr(test_ds, 'files', []))
        for attribute in ('path','meta_path','csv_path'):
            if hasattr(test_ds, attribute):
                path = Path(getattr(test_ds, attribute))
                hashes[str(path)] = sha256(path)
        if {x[0] for x in train} & {x[0] for x in test}: raise ValueError('Official train/test identity overlap')
        if actual_task == 'classification' and not set(test_labels.tolist()) <= set(labels.tolist()):
            raise ValueError('Test labels absent from official Train')
        counts = dict(n_train=len(train),n_test=len(test))
        split_record = dict(protocol=split_protocol_for_dataset(name), train=train,test=test)
        del test_ds
    elif name in GROUP_SPLIT_DATASETS:
        tr, te = group_split_indices(name, dataset, benchmark)
        groups = group_keys(name,dataset,benchmark)
        if {groups[i] for i in tr} & {groups[i] for i in te}: raise ValueError('Source group leakage')
        counts = dict(n_train=len(tr),n_test=len(te))
        spec = json.loads(split_path(name).read_text())
        if counts != {key:int(spec[key]) for key in counts}: raise ValueError('Committed split count mismatch')
        hashes[str(split_path(name))] = sha256(split_path(name))
        if hashes[str(split_path(name))] != GROUP_HASHES[name]: raise ValueError('Rules/02 locked split hash mismatch')
        split_record = dict(protocol='group-split',train=[train[i] for i in tr],test=[train[i] for i in te])
    else:
        raise ValueError('Stage 1 forbids an unverified random split')
    if task == 'classification' and actual_task not in ('classification','multilabel_classification'):
        raise ValueError('Registry task mismatch')
    size = resolve_image_size(name,'best',224)
    resize = resolve_dataset_resize_size(name,size,0)
    source_inventory = []
    if source_files:
        source_inventory = [(str(path),path.stat().st_size,path.stat().st_mtime_ns) for path in source_files]
    elif hasattr(dataset,'path'):
        path = Path(dataset.path)
        source_inventory = [(str(path),path.stat().st_size,path.stat().st_mtime_ns)]
    else:
        for record in all_records:
            path = Path(record[0]); stat = path.stat()
            source_inventory.append((str(path),stat.st_size,stat.st_mtime_ns))
    del dataset
    gc.collect()
    return dict(status='PASS',task=task,dataset=name,split=split_record['protocol'],counts=counts,
                split_identity_sha256=fingerprint(split_record),dataset_inventory_sha256=fingerprint(source_inventory),
                source_hashes=hashes,image_size=size,resize_size=resize,requires_pyarrow=bool(source_files))


def retrieval_preflight(name, benchmark):
    from dinov3.eval.bio_frozen_eval.retrieval_clustering import build_retrieval_dataset
    protocol = benchmark/'Retrieval_Clustering/protocols/v1'
    hashes = {}
    if name in ('nct-crc-he-1k','crc-val-he-7k'):
        ds, _ = build_retrieval_dataset(name, benchmark_root=benchmark)
        # Hash the fixed decoded row identities, labels, and original image bytes.
        digest = hashlib.sha256()
        for raw,label,identity in ds.rows:
            digest.update(f'{identity}:{label}:'.encode()); digest.update(hashlib.sha256(raw).digest())
        counts = dict(n_samples=len(ds))
        # The named 1K release used by FM baselines contains 111 x 9 = 999 rows.
        if len(ds) != (999 if name=='nct-crc-he-1k' else 7180): raise ValueError('Fixed retrieval count mismatch')
        split_hash = digest.hexdigest()
        del ds
    else:
        filename = 'hpa_same_gene_query_gallery.csv' if name=='hpa-subcellular' else 'rxrx1_official_cross_experiment_core.csv'
        path = protocol/filename
        rows = list(csv.DictReader(path.open()))
        identity_key = 'image_path' if name=='hpa-subcellular' else 'site_id'
        label = lambda row: row['label'] if name=='hpa-subcellular' else (row['cell_type'],row['sirna_id'])
        gallery = [r for r in rows if r['role']=='gallery']; query = [r for r in rows if r['role']=='query']
        if {r[identity_key] for r in gallery} & {r[identity_key] for r in query}: raise ValueError('Query/gallery overlap')
        eligible = {label(r) for r in gallery} & {label(r) for r in query}
        if not eligible: raise ValueError('No eligible query labels')
        if name=='rxrx1-cross':
            gallery = [r for r in gallery if label(r) in eligible]; query = [r for r in query if label(r) in eligible]
            if {r['experiment'] for r in gallery} & {r['experiment'] for r in query}: raise ValueError('RxRx1 experiment leakage')
            if not (benchmark/'Retrieval_Clustering/RxRx1/archives/rxrx1-images.zip').is_file(): raise FileNotFoundError('RxRx1 archive missing')
        elif not {label(r) for r in query} <= {label(r) for r in gallery}: raise ValueError('Missing gallery gene')
        counts = dict(n_gallery=len(gallery),n_query=len(query))
        if name=='rxrx1-cross':
            counts['cell_types'] = {cell:dict(n_gallery=sum(r['cell_type']==cell for r in gallery),
                                              n_query=sum(r['cell_type']==cell for r in query))
                                    for cell in sorted({r['cell_type'] for r in query})}
        hashes[str(path)] = sha256(path)
        if name=='hpa-subcellular':
            cluster_path = protocol/'hpa_single_location_clustering.csv'
            cluster = list(csv.DictReader(cluster_path.open()))
            counts.update(cluster_samples=len(cluster),robust_samples=sum(r['robust_ge10']=='1' for r in cluster))
            hashes[str(cluster_path)] = sha256(cluster_path)
            root = benchmark/'Retrieval_Clustering/HPA_Subcellular'
            if not all((root/r['image_path']).is_file() for r in rows+cluster): raise FileNotFoundError('HPA manifest image missing')
        split_hash = fingerprint(hashes)
    gc.collect()
    return dict(status='PASS',task='retrieval',dataset=name,counts=counts,source_hashes=hashes,
                split_identity_sha256=split_hash,dataset_inventory_sha256=split_hash,image_size=224,resize_size=256,
                requires_pyarrow=name in ('nct-crc-he-1k','crc-val-he-7k'))


def prepare(args):
    if git('status','--porcelain'): raise RuntimeError('Prepare from a clean Git checkout')
    if (args.output/'campaign_manifest.json').exists(): raise RuntimeError('Do not overwrite an existing campaign')
    protocol = json.loads((ROOT/'Evaluation Rules/protocol_v3.json').read_text())
    datasets = []
    for task in ('classification','regression','retrieval'):
        for name in protocol['tier_a'][task]+protocol['tier_b'][task]:
            if name=='rxrx3-core':
                datasets.append(dict(task=task,dataset=name,status='PLANNED',reason='Dedicated worker pending'))
                continue
            try:
                spec = retrieval_preflight(name,args.benchmark) if task=='retrieval' else dataset_preflight(name,task,args.benchmark)
            except Exception as error:
                spec = dict(task=task,dataset=name,status='PLANNED',reason=f'{type(error).__name__}: {error}')
            datasets.append(spec)
            save(args.output/'dataset_preflight.json',datasets)
            print(name,spec['status'],spec.get('reason',''),flush=True)
    assets = list(csv.DictReader(args.assets.open()))
    priority = {18543:0,26351:1,29279:2,12687:3}
    assets.sort(key=lambda row:(priority.get(int(row['checkpoint_id']),4) if row['checkpoint_id'].isdigit() else 4,
                                row['arm'],int(row['checkpoint_id']) if row['checkpoint_id'].isdigit() else -1))
    tasks = []
    ready = [d for d in datasets if d['status']=='PASS']
    order = {'breastmnist':0,'bbbc013':1,'nct-crc-he-1k':2,'organcmnist':3,'pneumoniamnist':4}
    ready.sort(key=lambda d:order.get(d['dataset'],5))
    for dataset in ready:
        for asset in assets:
            key = f"{asset['arm']}_ck{asset['checkpoint_id']}__{dataset['task']}__{dataset['dataset']}"
            tasks.append(dict(key=key,asset=asset,dataset=dataset))
    save(args.output/'campaign_manifest.json',dict(
        protocol_id='bio-eval-frozen-independent-stage1',authorization=args.authorization,
        status='RUNNING',full_v3_aggregate_allowed=False,git_commit=git('rev-parse','HEAD'),git_status_porcelain='',
        protocol_sha256=sha256(ROOT/'Evaluation Rules/protocol_v3.json'),plan_sha256=sha256(ROOT/args.plan),
        checkpoint_assets=assets,datasets=datasets,tasks=tasks,batch_size=64,autocast_dtype='bf16',
        n_last_blocks=1,use_avgpool=True,seed=0,num_workers=2,environment=THREADS,
        benchmark_root=str(args.benchmark),no_checkpoint_or_data_transfers=True,legacy_reuse=False,
        features_persisted=False,created_unix=time.time()))
    print('PREPARED',len(tasks),'component cells',flush=True)


def project_tests():
    candidates = {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            args = [s.decode(errors='replace') for s in (path/'cmdline').read_bytes().split(b'\0') if s]
            module = args[args.index('-m')+1] if '-m' in args else ''
            if not (module.startswith('dinov3.eval.') and module not in ('dinov3.eval.bio_benchmark','dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline')): continue
            parent = int(next(x.split()[1] for x in (path/'status').read_text().splitlines() if x.startswith('PPid:')))
            env = dict(x.split(b'=',1) for x in (path/'environ').read_bytes().split(b'\0') if b'=' in x)
            gpu = env.get(b'CUDA_VISIBLE_DEVICES',b'').decode()
            if not gpu and '--device' in args: gpu = args[args.index('--device')+1].removeprefix('cuda:')
            candidates[int(path.name)] = (parent,gpu)
        except (OSError,StopIteration,ValueError,IndexError): pass
    counts = {}
    for parent,gpu in candidates.values():
        if parent not in candidates: counts[gpu] = counts.get(gpu,0)+1
    return counts


def resources():
    result = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    return {int(row[0]):(int(row[1]),int(row[2])) for row in csv.reader(result.splitlines())}


def teacher_branch_record(payload):
    if not isinstance(payload,dict): raise RuntimeError('Explicit teacher branch required')
    if 'teacher' in payload:
        if not isinstance(payload['teacher'],dict) or not payload['teacher']:
            raise RuntimeError('Invalid explicit teacher container')
        return dict(teacher_key='teacher',consolidated_checkpoint_key='teacher',teacher_tensor_count=len(payload['teacher']))
    state = payload.get('model')
    if isinstance(state,dict):
        count = sum(key.replace('module.','').startswith('teacher.backbone.') for key in state)
        if count:
            return dict(teacher_key='model.teacher.backbone',consolidated_checkpoint_key='model',
                        teacher_prefix='teacher.backbone.',teacher_tensor_count=count)
    raise RuntimeError('Explicit teacher branch required; student-only weights forbidden')


def checkpoint_record(output, asset):
    path = Path(asset['path'])
    key = fingerprint(str(path))
    record_path = output/'_state/inputs'/f'{key}.json'
    with (output/'_state/checkpoint_io.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.is_file() and (
                p.name == '.metadata' or p.suffix in ('.distcp','.safetensors','.bin','.pth','.h5','.json')))
            inventory = [(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in files]
            if not files: raise RuntimeError('Empty model asset directory')
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record['inventory'] != [list(v) for v in inventory]: raise RuntimeError('Model asset directory changed')
                if asset.get('config') and record['config_sha256'] != sha256(asset['config']): raise RuntimeError('Config changed')
                return record
            if asset.get('kind') == 'external':
                teacher = dict(teacher_key='published_frozen_external',model_id=asset['model_id'])
            else:
                from torch.distributed.checkpoint import FileSystemReader
                metadata = FileSystemReader(str(path)).read_metadata()
                keys = [key for key in metadata.state_dict_metadata if 'teacher.backbone.' in key]
                if not keys: raise RuntimeError('DCP has no explicit teacher backbone')
                teacher = dict(teacher_key='DCP.model.teacher.backbone',teacher_tensor_count=len(keys))
            hashes = {p.name:sha256(p) for p in files}
            after = [(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in files]
            if after != inventory: raise RuntimeError('Model assets still being written')
            record = dict(path=str(path),inventory=inventory,bytes=sum(v[1] for v in inventory),
                          sha256=fingerprint(hashes),file_sha256=hashes,**teacher,
                          config_path=asset.get('config',''),config_sha256=sha256(asset['config']) if asset.get('config') else '')
            save(record_path,record)
            return record
        stat = path.stat()
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if (record['bytes'],record['mtime_ns']) != (stat.st_size,stat.st_mtime_ns): raise RuntimeError('Checkpoint changed after registration')
            if record['config_sha256'] != sha256(asset['config']): raise RuntimeError('Config changed after registration')
            return record
        import torch
        payload = torch.load(path,map_location='cpu',weights_only=False)
        teacher = teacher_branch_record(payload)
        del payload; gc.collect()
        record = dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=sha256(path),**teacher,
                      config_path=asset['config'],config_sha256=sha256(asset['config']))
        after = path.stat()
        if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns): raise RuntimeError('Checkpoint is still being written')
        save(record_path,record)
        return record


def command(task, output, benchmark):
    if task['asset'].get('kind') == 'external':
        return [sys.executable,'-u','-m','dinov3.eval.bio_frozen_eval.run_external_standard',
                '--model',task['asset']['model_id'],'--dataset',task['dataset']['dataset'],
                '--task',task['dataset']['task'],'--output',str(output),'--benchmark',str(benchmark),
                '--batch-size','64','--num-workers','2','--seed','0']
    retrieval = task['dataset']['task']=='retrieval'
    module = 'run_retrieval_clustering' if retrieval else 'run_classification'
    args = [sys.executable,'-u','-m','dinov3.eval.bio_frozen_eval.'+module,
            '--checkpoint',task['asset']['path'],'--train-config',task['asset']['config'],
            '--benchmark-root',benchmark,'--datasets',task['dataset']['dataset'],
            '--output-dir',str(output),'--model-name',task['key'].split('__')[0],
            '--device','cuda:0','--batch-size','64','--num-workers','2','--seed','0',
            '--autocast-dtype','bf16','--n-last-blocks','1','--channel-policy','auto',
            '--overwrite-results','--overwrite-features','--no-save-features']
    if not retrieval: args += ['--split-protocol','current','--resolution-protocol','best','--image-size','224','--resize-size','0']
    return args


def verify_source(state, path, digest):
    """Hash once per fixed source version, then reject changed size/mtime."""
    path = Path(path)
    key = fingerprint(str(path))
    directory = state/'sources'; directory.mkdir(exist_ok=True)
    record_path = directory/f'{key}.json'
    with (directory/f'{key}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        stat = path.stat()
        identity = dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=digest)
        if record_path.exists():
            if json.loads(record_path.read_text()) != identity: raise RuntimeError('Source changed after registration')
        else:
            if sha256(path) != digest: raise RuntimeError('Dataset/split changed after preflight')
            after = path.stat()
            if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns): raise RuntimeError('Source being modified')
            save(record_path,identity)


def validate_cell(task, directory, invocation):
    result = json.loads((directory/'last_result.json').read_text())
    rows = result.get('rows',[result])
    spec = task['dataset']
    for row in rows:
        if row.get('error') or row['dataset']!=spec['dataset']: raise ValueError('Wrong dataset or error result')
        for key in (('checkpoint',) if task['asset'].get('kind') == 'external' else ('checkpoint','train_config')):
            expected = task['asset']['path' if key=='checkpoint' else 'config']
            if Path(row[key]).resolve()!=Path(expected).resolve(): raise ValueError('Wrong model input')
    if spec['task']!='retrieval':
        for key in ('n_train','n_test'):
            if int(result[key])!=spec['counts'][key]: raise ValueError('Sample count mismatch: '+key)
        for key,value in (('batch_size',64),('seed',0),('image_size',spec['image_size']),('resize_size',spec['resize_size']),('split',spec['split'])):
            if result[key]!=value: raise ValueError('Protocol mismatch: '+key)
        metric = 'r2' if spec['task']=='regression' else 'macro_auc' if spec['dataset']=='chestmnist' else 'balanced_accuracy'
        if not math.isfinite(float(result[metric])): raise ValueError('Nonfinite primary metric')
    else:
        retrieval_rows = [r for r in rows if 'recall_at_1' in r]
        cluster_rows = [r for r in rows if 'nmi' in r]
        if not retrieval_rows or not cluster_rows: raise ValueError('Missing retrieval/clustering results')
        for row in retrieval_rows:
            if not math.isfinite(float(row['recall_at_1'])): raise ValueError('Nonfinite recall')
            expected_counts = spec['counts'].get('cell_types',{}).get(row.get('aggregation'),spec['counts'])
            for key in ('n_query','n_gallery','n_samples'):
                if key in expected_counts and int(row[key])!=expected_counts[key]: raise ValueError('Retrieval count mismatch')
        for row in cluster_rows:
            if not math.isfinite(float(row['nmi'])): raise ValueError('Nonfinite NMI')
            if spec['dataset']=='hpa-subcellular':
                expected = spec['counts']['robust_samples' if 'ge10-34' in row['protocol'] else 'cluster_samples']
            elif spec['dataset']=='rxrx1-cross':
                expected = spec['counts']['n_query'] + spec['counts']['n_gallery']
            else:
                expected = spec['counts']['n_samples']
            if int(row['n_samples']) != expected: raise ValueError('Clustering count mismatch')
    result['_component_provenance'] = invocation
    save(directory/'component_result.json',result)
    report = dict(status='VALID_COMPLETE',scope='INDEPENDENT_COMPONENT_ONLY',full_v3_aggregate_allowed=False,
                  result_sha256=sha256(directory/'component_result.json'),input_fingerprint=fingerprint(invocation),
                  expected_counts=spec['counts'],validator_commit=invocation['git_commit'],validated_unix=time.time())
    save(directory/'validation_report.json',report)
    return report


def worker(args):
    os.environ.update(THREADS)
    manifest = json.loads((args.output/'campaign_manifest.json').read_text())
    commit = git('rev-parse','HEAD')
    if commit!=manifest['git_commit'] or git('status','--porcelain'): raise RuntimeError('Wrong or dirty checkout')
    import torch
    import sklearn
    import importlib.util
    has_pyarrow = importlib.util.find_spec('pyarrow') is not None
    has_external = all(importlib.util.find_spec(name) is not None for name in ('transformers','timm','safetensors'))
    if int(torch.__version__.split('.')[0])<2 or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Modern CUDA BF16 evaluator environment required')
    if args.host=='deepcad' and not set(args.gpus)<=set(range(4)): raise RuntimeError('Deepcad restricted to GPU0-3')
    state = args.output/'_state'; state.mkdir(exist_ok=True)
    for name in ('claims','done','workers','running'): (state/name).mkdir(exist_ok=True)
    host_lock = (state/'workers'/f'{args.host}.lock').open('a')
    fcntl.flock(host_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (state/'manifest.lock').open('a') as registration:
        fcntl.flock(registration,fcntl.LOCK_EX)
        registered = json.loads((args.output/'campaign_manifest.json').read_text())
        registered.setdefault('execution_hosts',{})[args.host] = dict(
            hostname=os.uname().nodename,pid=os.getpid(),gpu_indices=args.gpus,git_commit=commit,git_status_porcelain='',
            python=sys.version,python_executable=sys.executable,torch=torch.__version__,sklearn=sklearn.__version__,
            verified_input_manifests='_state/inputs/*.json',job_manifests='cells/*/invocation_manifest.json')
        save(args.output/'campaign_manifest.json',registered)
    status_path = state/'workers'/f'{args.host}.json'
    stopping = False
    def stop(signum,frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    active = {}
    last_snapshot = 0
    while not stopping:
        for key,(process,log,task,claim,invocation) in list(active.items()):
            code = process.poll()
            if code is None: continue
            log.close()
            directory = args.output/'cells'/key
            try:
                if code: raise RuntimeError(f'Evaluator exit {code}')
                report = validate_cell(task,directory,invocation)
                save(state/'done'/f'{key}.json',report)
            except Exception as error:
                save(directory/'validation_report.json',dict(status='INVALID_PROTOCOL' if code==0 else 'FAILED',error=str(error)))
                save(state/'PAUSED.json',dict(task=key,host=args.host,error=str(error),time=time.time()))
            del active[key]
            (state/'running'/f'{key}.json').unlink(missing_ok=True)
        cards = resources(); counts = project_tests()
        ram = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024**2
        snapshot = dict(host=args.host,pid=os.getpid(),git_commit=commit,git_status_porcelain='',
                        python=sys.version,python_executable=sys.executable,torch=torch.__version__,sklearn=sklearn.__version__,
                        gpu_memory_mib=cards,actual_test_counts=counts,own_active=list(active),ram_available_gib=ram,
                        updated_unix=time.time(),status='PAUSED' if (state/'PAUSED.json').exists() else 'RUNNING')
        save(status_path,snapshot)
        if time.time()-last_snapshot>=1800:
            save(state/'workers'/f'{args.host}.resources.{int(time.time())}.json',snapshot)
            last_snapshot = time.time()
        if (state/'PAUSED.json').exists():
            if not active: break
            time.sleep(5); continue
        if ram<32 or __import__('shutil').disk_usage(args.output).free<64*1024**3:
            time.sleep(10); continue
        launched = False
        for gpu in sorted(args.gpus,key=lambda g:(counts.get(str(g),0),cards[g][0]/cards[g][1],g)):
            used,total = cards[gpu]
            own = sum(v[4]['gpu']==gpu for v in active.values())
            actual = max(counts.get(str(gpu),0),own)
            if actual>=args.target_per_gpu or len(active)>=args.max_host_jobs or (used/total>=0.6 and (actual>=3 or own)): continue
            for task in manifest['tasks']:
                key = task['key']
                if task['dataset'].get('requires_pyarrow') and not has_pyarrow: continue
                if task['asset'].get('kind') == 'external' and not has_external: continue
                if (state/'done'/f'{key}.json').exists() or (state/'claims'/key).exists(): continue
                reserve = max(18432 if task['dataset']['image_size']>=384 else 4096,
                              int(task['asset'].get('reserve_mib') or 0))
                pending = sum(int(v[4].get('reserve_mib',0)) for v in active.values()
                              if v[4]['gpu']==gpu and time.time()-v[4]['started_unix']<30)
                if total-used-pending<reserve: continue
                claim = state/'claims'/key
                with (state/'admission.lock').open('a') as admission:
                    fcntl.flock(admission,fcntl.LOCK_EX)
                    if len(list((state/'running').glob('*.json'))) >= args.max_global_jobs: break
                    try: claim.mkdir()
                    except FileExistsError: continue
                    save(state/'running'/f'{key}.json',dict(host=args.host,gpu=gpu,time=time.time()))
                save(claim/'owner.json',dict(host=args.host,pid=os.getpid(),gpu=gpu,time=time.time()))
                try:
                    inputs = checkpoint_record(args.output,task['asset'])
                    for path,digest in task['dataset']['source_hashes'].items():
                        verify_source(state,path,digest)
                    directory = args.output/'cells'/key; directory.mkdir(parents=True,exist_ok=True)
                    cmd = command(task,directory,manifest['benchmark_root'])
                    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),**THREADS)
                    invocation = dict(git_commit=commit,git_status_porcelain='',host=args.host,gpu=gpu,
                        python=sys.version,python_executable=sys.executable,torch=torch.__version__,sklearn=sklearn.__version__,
                        command=cmd,environment={key:env[key] for key in ('CUDA_VISIBLE_DEVICES',*THREADS)},
                        checkpoint=inputs,dataset=task['dataset'],batch_size=64,autocast_dtype='bf16',
                        n_last_blocks=1,use_avgpool=True,seed=0,num_workers=2,features_persisted=False,started_unix=time.time())
                    invocation['reserve_mib'] = reserve
                    invocation['asset_kind'] = task['asset'].get('kind','dinov3')
                    invocation['external_source_hashes'] = manifest.get('external_source_hashes',{})
                    save(directory/'invocation_manifest.json',invocation)
                    log = (directory/'run.log').open('w')
                    log.write('COMMAND '+__import__('shlex').join(cmd)+'\n'); log.flush()
                    process = subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[key] = (process,log,task,claim,invocation)
                    print('START',args.host,gpu,process.pid,key,flush=True)
                    launched = True
                except Exception as error:
                    save(state/'PAUSED.json',dict(task=key,host=args.host,error=str(error),time=time.time()))
                    (state/'running'/f'{key}.json').unlink(missing_ok=True)
                break
            if launched: break
        if not active and all((state/'done'/f"{t['key']}.json").exists() for t in manifest['tasks']): break
        time.sleep(3)
    if stopping:
        for process,log,*_ in active.values():
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            log.close()
    snapshot['status'] = 'STOPPED' if stopping else 'PAUSED' if (state/'PAUSED.json').exists() else 'COMPONENT_QUEUE_FINISHED'
    save(status_path,snapshot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','worker'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets',type=Path)
    parser.add_argument('--benchmark',type=Path,default=Path('/mnt/huawei_deepcad/benchmark'))
    parser.add_argument('--host',default=os.uname().nodename)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0])
    parser.add_argument('--max-host-jobs',type=int,default=6)
    parser.add_argument('--max-global-jobs',type=int,default=24)
    parser.add_argument('--target-per-gpu',type=int,choices=range(1,6),default=3)
    parser.add_argument('--plan',default='Evaluation Rules/plans/shared_frozen_stage1_20260917.md')
    parser.add_argument('--authorization',default='User approved staged components on 2026-09-17')
    args = parser.parse_args()
    if args.mode=='prepare': prepare(args)
    else: worker(args)


if __name__=='__main__':main()
