#!/usr/bin/env python3
"""Explicit 56-cell v4 inventory; missing cells never enter an aggregate as zero."""
import json,sys,time
from collections import Counter
import selective_retention_eval_queue_20260923 as q
import hs6_l5_weightspace_campaign_20260922 as v4

def once():
    protocol=json.loads((q.REPO/'Evaluation Rules/protocol_v4.json').read_text())
    cells={}
    for tier in ('tier_a','tier_b','union_extension'):
        for family,datasets in protocol[tier].items():
            for ds in datasets:cells[(family,ds)]=True
    for family,ds in list(cells):
        if family=='retrieval':cells[('clustering',ds)]=True
    tasks={p.stem:json.loads(p.read_text()) for p in (q.ROOT/'tasks').glob('*.json')}
    arms=sorted({t['arm'] for t in tasks.values()});rows=[]
    for arm in arms:
        for family,ds in sorted(cells):
            prefix={'classification':'cls','regression':'cls','retrieval':'ret','clustering':'ret',
                    'detection_proxy':'det','segmentation':'seg'}.get(family,family)
            if family=='segmentation' and ds in ('conic','livecell','pannuke','monuseg'):prefix='seg_primary'
            tid=f'{prefix}__{ds}__{arm}';task=tasks.get(tid)
            row={'arm':arm,'family':family,'dataset':ds,'admission':protocol['admission_gates'].get(f'{family}/{ds}','STANDARD')}
            if task:
                st=q.ROOT/'claims'/tid/'status.json'
                row.update(state=json.loads(st.read_text())['state'] if st.exists() else 'QUEUED',task=tid,
                           evidence=task['done'],source=task['cwd'],readout=task.get('readout','v4 canonical'))
            elif arm.startswith('baseline_') and family not in ('ood','cell_tracking'):
                role=arm[-1];v4.ROOT=v4.COEX;v4.SEG_CKPT_ID.update(v4.BASE_ROLES)
                if prefix=='cls':old=v4.cls_task(ds,role,0)
                elif prefix=='ret':old=v4.ret_task(ds,role,0)
                elif prefix=='det':old=v4.det_task(ds,role,0)
                else:old=v4.seg_task(ds,role,0)
                if family=='segmentation' and ds in ('tissuenet','multimodal_cellseg'):
                    old['done']['root']=str(v4.COEX/'segmentation/remaining_results')
                if family=='detection_proxy' and ds in ('conic','livecell'):
                    old['done']['path']=str(q.REPO/'outputs/02_eval_runs/hs6_l5_weightspace_baseline_v4_20260922/detection'/ds/role/'results_bio_detection.json')
                try:ok,reason=v4.task_done(old['done'])
                except Exception as e:ok,reason=False,str(e)
                if family in ('retrieval','clustering') and ds=='lc25000':
                    paired=v4.COEX/'fusion/retrieval_lc25000.json'
                    data=json.loads(paired.read_text());bank=data['feature_files'][role]
                    ok=(data['protocol']=='within-set-leave-one-out' and data['n']==25000 and
                        'nmi' in data['results'][role] and 'recall_at_1' in data['results'][role] and
                        q.sha(bank['path'])==bank['sha256'])
                    reason='paired result with verified feature identity';old['done']={'type':'paired_json','path':str(paired),'role':role}
                row.update(state='REUSED_VALIDATED' if ok else 'NOT_SCHEDULED',evidence=old['done'],validation=reason)
            else:
                row.update(state='ADMISSION_PENDING' if family=='cell_tracking' else 'NOT_SCHEDULED',
                           reason='native fixed head/linker admission' if family=='cell_tracking' else 'Separate OOD evaluator not in overnight queue')
            rows.append(row)
    counts={arm:dict(Counter(r['state'] for r in rows if r['arm']==arm)) for arm in arms}
    q.atomic(q.ROOT/'V4_INVENTORY.json',{'time':q.now(),'protocol_sha256':q.sha(q.REPO/'Evaluation Rules/protocol_v4.json'),
             'expected_per_arm':protocol['expected_unique_dataset_counts'],'cells_per_arm':len(cells),'counts':counts,'cells':rows,
             'aggregate_status':'NOT_ADMITTED: requires all matched valid cells; provisional and proxy columns remain separate'})
    lines=['# v4 coverage ledger','',q.now(),'','Each model has 56 expected task-dataset cells. Retrieval and clustering share extraction; dense primary and secondary readouts are distinct.','',
           '| Model | Cell states |','|---|---|']
    lines += [f'| {arm} | {state} |' for arm,state in counts.items()]
    lines += ['','No complete v4 aggregate is claimed. CTC native admission and OOD remain explicit gaps; LC25000 classification is provisional. See V4_INVENTORY.json for each output and source.']
    (q.ROOT/'V4_INVENTORY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    if '--watch' in sys.argv:
        end=time.time()+24*3600
        while time.time()<end:once();time.sleep(60)
    else:once()
