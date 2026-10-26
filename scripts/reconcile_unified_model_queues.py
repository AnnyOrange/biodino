"""Keep newly added local BioDINO anchors in subsequently prepared common tasks."""
import fcntl, json, time, hashlib
from pathlib import Path
from admit_unified_paper_20tb_best import OUT, read, write

def release_selected_native_fit_claims():
    """Selection stays serialized; completed validation selection is immutable.

    Distinct folds/seeds write distinct fit directories. Keep active legacy
    claim owners unchanged; new children verify the current per-fit claim
    before loading weights, rejecting any stale worker queue snapshot.
    """
    changed_jobs=[];invalid=[]
    for name in ['nucls_queue.json','native_detection_queue.json','rsna_queue.json']:
        path=OUT/name
        if not path.exists():continue
        queue=read(path);changed=False;checked={}
        for job in queue.get('tasks',[]):
            if job.get('blocked_reason')or Path(job['result']).exists()or job.get('claim_schema')=='per_fit_after_locked_selection_v1':continue
            base=OUT/'results'/job['model']
            if name=='nucls_queue.json':selection=base/'nucls_qc/selected_recipe.json';config=OUT/'nucls_native_config.json'
            elif name=='native_detection_queue.json':selection=base/'native_box'/job['dataset']/'selected_recipe.json';config=OUT/'native_detection_config.json'
            else:selection=base/'rsna_pneumonia/selected_recipe.json';config=None
            if not selection.exists():continue
            if selection not in checked:
                raw=read(selection)
                try:
                    assert Path(raw['selected']['state']).is_file()
                    if config:
                        assert raw['config_sha256']==hashlib.sha256(config.read_bytes()).hexdigest()
                        assert raw['selected']['status']=='complete'
                    if name=='nucls_queue.json':assert raw['selection_fold']==1 and raw['fold2_5_scores_used']is False
                    else:assert raw['test_used']is False
                    if name=='rsna_queue.json':assert raw['manifest_sha256']==hashlib.sha256((OUT.parents[1]/'02_eval_inputs/formal_v4/rsna_pneumonia_manifest.json').read_bytes()).hexdigest()
                    checked[selection]=True
                except (AssertionError,KeyError,OSError)as error:checked[selection]=False;invalid.append(dict(selection=str(selection),error=repr(error)))
            if not checked[selection]:continue
            legacy=job.get('claim_key',job['model']);claim=OUT/'locks'/f'{legacy}.claimdir'
            try:owner=read(claim/'owner.json')
            except FileNotFoundError:
                # mkdir precedes owner publication; preserve this live launch
                # window rather than migrating its shared claim concurrently.
                if claim.exists():continue
                owner={}
            except (OSError,json.JSONDecodeError):
                # An unreadable NFS owner is not evidence of an inactive job.
                continue
            if owner.get('task')==job['key']:continue
            job.update(claim_key=job['key'],claim_schema='per_fit_after_locked_selection_v1',selection_locked=str(selection),selection_locked_sha256=hashlib.sha256(selection.read_bytes()).hexdigest())
            job['requires']=sorted(set(job.get('requires',[]))|{str(selection)});changed=True;changed_jobs.append(job['key'])
        if changed:write(path,queue)
    record=dict(updated=time.time(),jobs_released_for_independent_fit_claims=changed_jobs,selection_validation_errors=invalid,active_legacy_claim_owners_preserved=True,training_protocol_changed=False)
    write(OUT/'states/native_independent_fit_admission.json',record)
    if changed_jobs:
        with(OUT/'logs/native_independent_fit_admission.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')
    return record

def run():
    added={}
    with (OUT/'locks/model_admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cfg=read(OUT/'config.json');base=next(m for m in cfg['models'] if m['model']=='biodino_20tb_l')
        models=[m for m in cfg['models'] if m['kind']=='biodino' and m['model'].startswith('biodino_20tb_') and not m.get('execution_host')]
        # Pilot-gated CTC and inventory-indexed segmentation have separate admission.
        for name in ['film_queue.json','rsna_queue.json','bone_age_queue.json','ood_repair_queue.json']:
            path=OUT/name
            if not path.exists():continue
            queue=read(path);templates=[j for j in queue.get('tasks',[]) if j['model']==base['model']];known={j['key'] for j in queue.get('tasks',[])};new=[]
            for spec in models:
                for template in templates:
                    raw=json.dumps(template).replace(base['checkpoint'],spec['checkpoint']).replace(base['train_config'],spec['train_config']).replace(base['model'],spec['model']);job=json.loads(raw)
                    if job['key'] in known:continue
                    job['min_free_mib']=max(job.get('min_free_mib',0),spec['min_free_mib']);new.append(job);known.add(job['key'])
            if new:
                queue['tasks'].extend(new);write(path,queue);added[name]=[j['key'] for j in new]
        resident={m['model']:m['execution_host'] for m in cfg['models'] if m.get('execution_host')};held=[]
        for path in OUT.glob('*queue.json'):
            if path.name=='remote_queue.json':continue
            queue=read(path);changed=False
            for job in queue.get('tasks',[]):
                if job['model'] not in resident:continue
                reason='Main-cluster placeholder; execution is restricted to resident host '+resident[job['model']]+'. Actual admitted jobs and states are indexed in remote_queue.json; no cross-building weight transfer.'
                if job.get('dataset')=='xray_bone':
                    preparation=read(OUT/'states/ai_bone_age_preparation.json')
                    reason='Resident bone-age preparation: '+preparation.get('status','not_started')+'; complete lossless input archive SHA verification and shared-runner equivalence are required. See states/ai_bone_age_preparation.json.'
                elif job.get('dataset')=='xray_chest' and job.get('entrypoint')=='run_unified_paper_rsna.py':
                    preparation=read(OUT/'states/ai_native_rsna_pneumonia_preparation.json')
                    reason='Resident RSNA preparation: '+preparation.get('status','not_started')+'; all20403 canonical images must pass SHA verification. See states/ai_native_rsna_pneumonia_preparation.json.'
                if job.get('blocked_reason')!=reason or job.get('allowed_hosts')!=[resident[job['model']]]:
                    job.update(blocked_reason=reason,allowed_hosts=[resident[job['model']]]);changed=True
                held.append(job['key'])
            if changed:write(path,queue)
        release_selected_native_fit_claims()
        record=dict(updated=time.time(),added=added,resident_main_placeholders=held,models_checked=len(models),excluded='AI-only weights require resident-host adapters; no cross-building checkpoint copy. CTC waits for native pilot, segmentation uses explicit cache inventory.')
        write(OUT/'states/model_queue_reconciliation.json',record)
        if added:
            with (OUT/'logs/model_queue_reconciliation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    return added

if __name__=='__main__':print(json.dumps(run(),indent=2))
