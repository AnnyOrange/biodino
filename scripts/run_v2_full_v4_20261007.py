#!/usr/bin/env python3
"""Complete the missing v4 families using previously admitted immutable evaluators.

The paired component campaign remains separate. This queue adds 12 missing
executions per checkpoint plus the amended MoNuSeg execution (six fits).
No test scores are used to select checkpoints or change protocols.
"""
import argparse, fcntl, hashlib, importlib.util, json, os, signal, subprocess, sys, time
from pathlib import Path

REPO = Path('/mnt/huawei_deepcad/dinov3')
ROOT = REPO/'outputs/02_eval_runs/v2_full_v4_20261007'
PAIRED = REPO/'outputs/02_eval_runs/v2_20tb_paired_20261006'
TEMPLATE = REPO/'outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927'
MONU = REPO/'outputs/02_eval_runs/monuseg_train30val7_test14_retest_20260929'
MONUSRC = Path('/mnt/huawei_deepcad/dinov3_monuseg_train30val7_snapshot_20260929')
PY = '/mnt/huawei_deepcad/eval_envs/retest_20260918_v2/bin/python'
THREADS = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')

def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare():
    for sub in ['tasks','claims','logs','runtime','provenance','workers','registered']:(ROOT/sub).mkdir(parents=True,exist_ok=True)
    with (ROOT/'register.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ref=json.loads((PAIRED/'campaign_manifest.json').read_text())
        if (ROOT/'5tb_assets.json').exists():ref['checkpoint_assets'] += json.loads((ROOT/'5tb_assets.json').read_text())
        alltemplates=[json.loads(p.read_text()) for p in (TEMPLATE/'tasks').glob('*.json')]
        tag='adaptive_continue_v2_gpu2_ck15615'
        templates=[t for t in alltemplates if t['arm']==tag and (t['dataset'] in ('rxrx3-core','ctc','xray','cryo','conic-cell-count','livecell-cell-count','lc25000','nct-crc-he-100') or t['family']=='detection_proxy')]
        assert len(templates)==12,len(templates)
        source_ck=next(t for t in alltemplates if t['arm']==tag and t['dataset']=='ctc')['cmd']
        oldck=source_ck[source_ck.index('--checkpoint')+1];oldconfig=source_ck[source_ck.index('--config')+1]
        adapter=(TEMPLATE/'runtime/run_rxrx3.py').read_text().replace(str(TEMPLATE),str(ROOT))
        dest=ROOT/'runtime/run_rxrx3.py'
        if not dest.exists():dest.write_text(adapter)
        assert dest.read_text()==adapter
        det=ROOT/'runtime/run_detection_single_rank.py'
        if not det.exists():det.write_bytes((TEMPLATE/'runtime/run_detection_single_rank.py').read_bytes())
        for p in (TEMPLATE/'provenance').glob('source_*.json'):
            dst=ROOT/'provenance'/p.name
            if not dst.exists():dst.write_bytes(p.read_bytes())
        monu=json.loads((MONU/'campaign_manifest.json').read_text());ds=next(d for d in monu['datasets'] if d['dataset']=='monuseg')
        assert ds['counts']==dict(train=30,val=7,test=14)
        pinned=json.loads((MONUSRC/'source_snapshot.json').read_text())
        assert pinned==monu['source_snapshot']
        for path,digest in pinned['files'].items():assert sha(MONUSRC/path)==digest,path
        for path,digest in ds['source_hashes'].items():assert sha(path)==digest,path
        for a in ref['checkpoint_assets']:
            arm=f"{a['arm']}_ck{a['checkpoint_id']}";marker=ROOT/'registered'/f'{arm}.json'
            if marker.exists():continue
            def replace(v):
                if isinstance(v,str):return v.replace(oldck,a['path']).replace(oldconfig,a['config']).replace(str(TEMPLATE),str(ROOT)).replace(tag,arm)
                if isinstance(v,list):return [replace(x) for x in v]
                if isinstance(v,dict):return {k:replace(x) for k,x in v.items()}
                return v
            tasks=[]
            selected=([t for t in templates if t['dataset'] in ('rxrx3-core','ctc','xray','cryo')]
                      if a.get('missing_only') else templates)
            for template in selected:
                t=replace(template);cmd=t['cmd'];t['asset']=a
                for flag in ['--step','--ckpt-iter','--checkpoint-id']:
                    if flag in cmd:cmd[cmd.index(flag)+1]=a['checkpoint_id']
                for stale in ['checkpoint_sha256','train_config_sha256','runtime_adapter_sha256','template_sha256']:t.pop(stale,None)
                t['template_sha256']=sha(TEMPLATE/'tasks'/f"{template['id']}.json")
                if t['family']=='ood':
                    adapterroot=ROOT/'ood_adapters'/arm;link=adapterroot/a['checkpoint_id']/'checkpoint.pth';link.parent.mkdir(parents=True,exist_ok=True)
                    if not link.exists():link.symlink_to(a['path'])
                    # The template's path encodes its old step independently of arm.
                    t['done']['path']=t['done']['path'].replace('/15615/',f"/{a['checkpoint_id']}/")
                if t['family']=='detection_proxy':
                    assert '--batch-size' in cmd and cmd[cmd.index('--batch-size')+1]=='8'
                    if '--conic-split-protocol' not in cmd:cmd+=['--conic-split-protocol','official-baseline-fold0-nested-v1']
                t['priority']={'cell_tracking':0,'ood':1,'detection_proxy':3}.get(t['family'],2)
                t['created_unix']=time.time();tasks.append(t)
            key=f'{arm}__segmentation__monuseg__primary-last__formal-static-v1'
            if not a.get('missing_only'):
                tasks.append(dict(id=key,arm=arm,family='segmentation',dataset='monuseg',priority=4,order=0,
                    cwd=str(MONUSRC),pythonpath=str(MONUSRC),asset=a,monuseg_spec=ds,
                    monuseg_source=monu['source_snapshot'],expected_memory_mib=18000))
            assert len({t['id'] for t in tasks})==(4 if a.get('missing_only') else 13)
            for t in tasks:save(ROOT/'tasks'/f"{t['id']}.json",t)
            save(marker,dict(asset=a,task_ids=[t['id'] for t in tasks]))
        save(ROOT/'campaign_manifest.json',dict(protocol_id='bio-eval-union-v4',authorization='2026-10-07 user explicitly requests all v4 tasks and practical optimization on all four hosts',
            source_component_campaign=str(PAIRED),admission_reference=str(TEMPLATE/'FULL_V4_ADMISSION.json'),
            expected_counts=json.loads((REPO/'Evaluation Rules/protocol_v4.json').read_text())['expected_unique_dataset_counts'],
            monuseg_counts=ds['counts'],lc25000_classification='PROVISIONAL_LEGACY_ONLY',
            strict_aggregate_allowed=False,assets=len(list((ROOT/'registered').glob('*.json'))),
            extension_executions=len(list((ROOT/'tasks').glob('*.json'))),
            runtime_hashes={str(p):sha(p) for p in [Path(__file__).resolve(),dest,det]},created_unix=time.time()))
    inventory()

def inventory():
    ref=json.loads((PAIRED/'campaign_manifest.json').read_text());proto=json.loads((REPO/'Evaluation Rules/protocol_v4.json').read_text())
    if (ROOT/'5tb_assets.json').exists():ref['checkpoint_assets'] += json.loads((ROOT/'5tb_assets.json').read_text())
    tasks=[json.loads(p.read_text()) for p in (ROOT/'tasks').glob('*.json')];rows=[]
    families={f:sum([proto.get(s,{}).get(f,[]) for s in ['tier_a','tier_b','union_extension']],[]) for f in ['classification','regression','retrieval','segmentation','cell_tracking','ood']}
    families['clustering']=families['retrieval'];families['detection_proxy']=proto['union_extension']['detection_proxy']
    for a in ref['checkpoint_assets']:
        arm=f"{a['arm']}_ck{a['checkpoint_id']}"
        for family,datasets in families.items():
            for ds in datasets:
                tf='retrieval' if family=='clustering' else 'classification' if family=='regression' and ds.endswith('cell-count') else family
                ext=[t for t in tasks if t['arm']==arm and t['family']==tf and t['dataset']==ds]
                if ext:
                    p=ROOT/'claims'/ext[0]['id']/'status.json';status=json.loads(p.read_text())['state'] if p.exists() else 'QUEUED'
                    evidence=str(p)
                elif a.get('missing_only'):
                    status='REMOTE_COMPONENT_VALIDATION_PENDING'
                    evidence=a['resident_component_root']
                else:
                    matched=[t for t in ref['tasks'] if t['asset']['path']==a['path'] and t['dataset']['dataset']==ds and t['dataset']['task']==('retrieval' if family=='clustering' else family)]
                    paths=[PAIRED/'_state/done'/f"{t['key']}.json" for t in matched]
                    status='VALID_COMPLETE' if paths and all(p.exists() and json.loads(p.read_text()).get('status')=='VALID_COMPLETE' for p in paths) else 'QUEUED' if matched else 'MISSING_NOT_SCHEDULED'
                    evidence=';'.join(map(str,paths))
                rows.append(dict(arm=arm,family=family,dataset=ds,state=status,evidence=evidence,provisional=family=='classification' and ds=='lc25000'))
    assert len(rows)==56*len(ref['checkpoint_assets'])
    save(ROOT/'FULL_V4_INVENTORY.json',dict(updated_unix=time.time(),expected_cells_per_checkpoint=56,cells=rows,
        missing=sum(r['state']=='MISSING_NOT_SCHEDULED' for r in rows),strict_aggregate_allowed=False))

def worker(args):
    os.environ.update(THREADS)
    sys.path.insert(0,str(REPO/'scripts'))
    import run_method_v4_fleet_20260928 as validator
    import run_deepcad_method_20260927 as resource
    fill=load_module('full_v4_fill',REPO/'scripts/run_v2_progress_fleet_fill_20261006.py')
    monufleet=load_module('full_v4_monufleet',MONUSRC/'scripts/run_retest_fleet_20260918.py')
    m=json.loads((ROOT/'campaign_manifest.json').read_text())
    for p,h in m['runtime_hashes'].items():assert sha(p)==h,p
    import importlib.metadata
    expected=json.loads((PAIRED/'campaign_manifest.json').read_text())['numerical_environment']
    for package,version in expected.items():
        assert importlib.metadata.version(package).split('+')[0]==version,package
    for p in (ROOT/'provenance').glob('*.json'):
        d=json.loads(p.read_text());assert resource.hash_files(Path(d['source']))==d['files'],d['source']
    lock=(ROOT/'workers'/f'{args.host}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    active={};last_inventory=0
    while not stopping:
        for key,j in list(active.items()):
            rc=j['proc'].poll()
            if rc is None:continue
            j['log'].close();t=j['task']
            try:
                if rc:raise RuntimeError(f'exit {rc}')
                if 'monuseg_spec' in t:
                    result=monufleet.validate(j['ft'],ROOT/'cells'/key,j['inv']);ok=result['status']=='VALID_COMPLETE';why='six amended MoNuSeg fits validated'
                else:ok,why=validator.validate(t)
                if not ok:raise RuntimeError(why)
                status='VALID_COMPLETE'
            except Exception as e:status='FAILED';why=str(e)
            save(ROOT/'claims'/key/'status.json',dict(j['status'],state=status,reason=why,returncode=rc,end=time.time()))
            print(status,key,why,flush=True);del active[key]
        cards=resource.gpu_info()
        ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024**2
        cg=Path(f'/sys/fs/cgroup/memory/user.slice/user-{os.getuid()}.slice')
        if (cg/'memory.limit_in_bytes').exists():
            stats=dict((k,int(v)) for k,v in (line.split() for line in (cg/'memory.stat').read_text().splitlines()))
            clean=max(0,stats.get('total_inactive_file',0)-stats.get('total_dirty',0)-stats.get('total_writeback',0))
            ram=min(ram,(int((cg/'memory.limit_in_bytes').read_text())-int((cg/'memory.usage_in_bytes').read_text())+clean)/1024**3)
        app=resource.gpu_processes();ready={p['pid'] for p in app}
        waiting=[j for j in active.values() if not (resource.descendants(j['proc'].pid)|{j['proc'].pid})&ready]
        future_ram=sum(24 if j['task']['family']=='cell_tracking' else 8 for j in waiting)
        save(ROOT/'workers'/f'{args.host}.json',dict(pid=os.getpid(),updated_unix=time.time(),active=[j['status'] for j in active.values()],gpu_memory=cards,ram_headroom_gib=ram,unloaded_jobs=len(waiting),target_fraction=.75))
        pending=[json.loads(p.read_text()) for p in (ROOT/'tasks').glob('*.json') if not (ROOT/'claims'/p.stem).exists()]
        pending.sort(key=lambda t:(int(t['asset']['checkpoint_id']),t['priority'],t['id']))
        launched=False
        for g in sorted(args.gpus,key=lambda g:cards[g]['used']/cards[g]['total']):
            if len(active)>=args.max_jobs or sum(j['status']['gpu']==g for j in active.values())>=5 or cards[g]['used']/cards[g]['total']>=.75:continue
            reserve=sum(j['peak'] for j in waiting if j['status']['gpu']==g)
            for t in pending:
                if args.family!='all' and t['family']!=args.family:continue
                peak=18000 if t['family']=='segmentation' else 10000 if t['family']=='cell_tracking' else 6000 if t['family']=='detection_proxy' else 4200
                hostpeak=24 if t['family']=='cell_tracking' else 8
                if cards[g]['total']-cards[g]['used']-reserve<peak+2048 or ram-future_ram-hostpeak<args.ram_reserve:continue
                claim=ROOT/'claims'/t['id']
                try:claim.mkdir()
                except FileExistsError:continue
                st=dict(state='STARTING',host=args.host,gpu=g,task=t['id'],time=time.time());save(claim/'status.json',st)
                try:
                    record=fill.checkpoint_record(PAIRED,t['asset'])
                    env=dict(os.environ,**THREADS);env.update(CUDA_VISIBLE_DEVICES=str(g),PYTHONPATH=t['pythonpath'],DINOV3_ROOT=t['cwd'],DINOV3_CODE_ROOT=t['cwd'])
                    inv=dict(checkpoint=record,git_commit=t.get('monuseg_source',{}).get('git_commit'),source_snapshot_sha256=t.get('monuseg_source',{}).get('sha256'),
                        numerical_environment=json.loads((PAIRED/'campaign_manifest.json').read_text())['numerical_environment'],
                        host=args.host,gpu=g,started_unix=time.time(),batch_size=32 if t['family']=='segmentation' else 64)
                    ft=None
                    if 'monuseg_spec' in t:
                        ft=dict(key=t['id'],asset=t['asset'],dataset=t['monuseg_spec']);cmd=monufleet.command(ft,ROOT/'cells'/t['id'],ref_benchmark())
                    else:cmd=[str(x).replace('{GPU}',str(g)) for x in t['cmd']]
                    inv.update(command=cmd,task=t);save(claim/'invocation.json',inv)
                    log=(ROOT/'logs'/f"{t['id']}.log").open('a');proc=subprocess.Popen(cmd,cwd=t['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    st.update(state='RUNNING',pid=proc.pid);save(claim/'status.json',st)
                    active[t['id']]=dict(proc=proc,log=log,task=t,status=st,peak=peak,ft=ft,inv=inv)
                    print('START',args.host,g,proc.pid,t['id'],flush=True)
                except Exception as e:save(claim/'status.json',dict(st,state='FAILED',reason=str(e)))
                launched=True;break
            if launched:break
        if time.time()-last_inventory>120:inventory();last_inventory=time.time()
        time.sleep(5)
    for j in active.values():
        try:os.killpg(j['proc'].pid,signal.SIGTERM)
        except ProcessLookupError:pass
        save(ROOT/'claims'/j['task']['id']/'status.json',dict(j['status'],state='INTERRUPTED'))

def ref_benchmark():return '/mnt/huawei_deepcad/benchmark'

if __name__=='__main__':
    os.umask(0)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','watch','worker','inventory']);p.add_argument('--host',default=os.uname().nodename);p.add_argument('--gpus',nargs='+',type=int,default=list(range(8)));p.add_argument('--max-jobs',type=int,default=24);p.add_argument('--ram-reserve',type=float,default=96);p.add_argument('--family',default='all');a=p.parse_args()
    if a.mode=='worker':worker(a)
    elif a.mode=='inventory':inventory()
    elif a.mode=='prepare':prepare()
    else:
        lock=(ROOT/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            prepare()
            subprocess.run([sys.executable,str(REPO/'scripts/summarize_v2_full_v4_20261007.py')],check=True)
            time.sleep(180)
