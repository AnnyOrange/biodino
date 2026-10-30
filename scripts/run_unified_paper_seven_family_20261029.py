"""Scoped 100TB/1PB runner; shared admission, unchanged scientific runners."""
import hashlib,importlib,json,os,signal,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/02_eval_runs/seven_family_100tb_1pb_20261029'
QUEUE=OUT/'queue.json'
CPU_HOSTS={'cpu1','cpu2','cpu5','cpu9','cpu10','cpu11','cpu12','cpu19'}
ALLOWED={'5090-zxr','3090-qi'}|CPU_HOSTS
RUNNERS={'run_unified_paper_'+s for s in ['global','native_detection','detection','retrieval','hpa_retrieval','rxrx1','rxrx3','lc25000_grouped','biodino_segmentation_cache','segmentation']}

def scoped_policy():
    import unified_paper_gpu_policy as policy
    previous=policy.excluded
    def excluded(host,gpu):
        if host not in ALLOWED:return True
        if host in CPU_HOSTS:
            # Only this explicitly authorized campaign process sees this exception.
            # Global policy and all old single-3090 queues remain disabled.
            return int(gpu)!=0 or int(gpu) in policy.cached_policy('gpu_exclusions.json').get('hosts',{}).get(host,{}).get('gpus',[])
        return previous(host,gpu)
    policy.excluded=excluded

def main():
    scoped_policy()
    if '--worker' in sys.argv:
        sys.argv.remove('--worker')
        assert Path(sys.argv[sys.argv.index('--queue-file')+1]).resolve()==QUEUE
        assert sys.argv[sys.argv.index('--host-label')+1] in ALLOWED
        import unified_paper_queue as queue
        # Generic source reload would lose scoped entrypoint; watchdog owns this wrapper.
        queue.reload_if_changed=lambda lock:None
        queue.main();return
    assert len(sys.argv)==3 and sys.argv[1]=='--job-key'
    # Graceful migration releases shared guards/reservations through atexit.
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(75))
    job=next(j for j in json.loads(QUEUE.read_text())['tasks'] if j['key']==sys.argv[2])
    assert job['runner'] in RUNNERS and Path(job['result']).is_relative_to(OUT)
    for filename,expected in json.loads((OUT/'scientific_source_hashes.json').read_text()).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
            raise ValueError('Scientific source changed; audit before retry: '+filename)
    import unified_paper_gpu_admission as admission
    admission.matching_job=lambda:job
    admission.reserve_for_current_command()
    admission.reserve_for_current_command=lambda:None
    module=importlib.import_module(job['runner'])
    module.OUT=OUT
    # MoNuSeg's approved 30/7/14 membership must also apply during head source verification.
    if job.get('dataset')=='monuseg':
        cells=json.loads((OUT/'biodino_segmentation_inventory.json').read_text())['cells']
        cell=cells[int(job['runner_args'][job['runner_args'].index('--cell-index')+1])]
        recipe=json.loads(Path(cell['invocation']).read_text())
        template=json.loads(Path(recipe['template_invocation']).read_text())
        hashes=template.get('template_source_hashes',template.get('dataset',{}).get('source_hashes',{}))
        manifest=next(k for k in hashes if k.endswith('monuseg2018_train30_extra7val_test14_manifest.json'))
        os.environ.update(DINOV3_MONUSEG_MANIFEST_FILE=Path(manifest).name,DINOV3_MONUSEG_MANIFEST_SHA256=hashes[manifest],DINOV3_MONUSEG_COUNTS=json.dumps(recipe['expected_counts']))
    sys.argv=[str(ROOT/'scripts'/f"{job['runner']}.py"),*job['runner_args']]
    try:
        module.main()
    except ValueError as error:
        if '--recover-existing' not in sys.argv or 'Retained head failed exact score reproduction' not in str(error):
            raise
        print('RETAINED HEAD REJECTED; fresh same-protocol E20 fit:', error, flush=True)
        sys.argv.remove('--recover-existing')
        module.main()

if __name__=='__main__':main()
