"""Publish branch-level checkpoint scope and actual queue/result counts."""
import csv, hashlib, json, time
from collections import Counter
from pathlib import Path
from admit_unified_paper_20tb_best import OUT, REPORT, read, write

def main():
    cfg=read(OUT/'config.json');jobs=[]
    for path in OUT.glob('*queue.json'):jobs.extend(read(path).get('tasks',[]))
    rows=[]
    for spec in cfg['models']:
        if not spec['model'].startswith('biodino_20tb_'):continue
        model=spec['model'];source=OUT/'anchor_specs'/f'{model}.json'
        provenance=read(source) if source.exists() else spec
        selected=[j for j in jobs if j['model']==model]
        rows.append(dict(model=model,display_name=spec.get('display_name',model),branch=spec.get('method_branch',spec.get('method',model)),step=spec['step'],checkpoint=spec['checkpoint'],checkpoint_sha256=spec.get('checkpoint_sha256',''),selection_source=provenance.get('selection_source',''),selection_source_snapshot=provenance.get('selection_source_snapshot',''),selection_note=provenance.get('selection_note',''),registered_jobs=len(selected),completed_job_outputs=sum(Path(j['result']).exists() for j in selected),status='已接入持久队列；完整任务与统计状态见 execution_status.csv',execution_host=spec.get('execution_host','shared main-building fleet')))
    pending=[dict(branch='lw2 + rebalanced mix',step=79543,status='AI-only checkpoint located; common retrieval adapter pilot being prepared',execution_host='5090-lyx-xr'),dict(branch='residual complement Bio5TB',step=50263,status='AI-only checkpoint located; common retrieval adapter pilot being prepared',execution_host='5090-lyx-xr'),dict(branch='boundary sampling',step=None,status='Historical selected29 has no fully validated point across all three PanNuke rounds. No imputed score used to select best; branch representative unresolved',evidence=str(OUT/'selection_evidence/boundary_historical_complete_points.json'))]
    pending=[p for p in pending if not any(s.get('method_branch')==p['branch'] for s in cfg['models'])]
    record=dict(updated=time.time(),scope='no-GRAM plus one best per method branch; retain pre-existing no-GRAM74663 late anchor separately',selection_policy='Use already completed historical branch evidence, freeze representative before new unified results. Historical test-selected choices are descriptive/exploratory; never describe them as independent validation-selected choices. No imputation and no new all-checkpoint sweep.',models=rows,pending_branches=pending,total_models=len(cfg['models']),fm_models=sum(s['kind']=='fm' for s in cfg['models']),holm_family_size=cfg['statistics']['planned_comparisons_per_family'],excluded=['withdrawn survey24','7B scale experiment as a new L20TB method branch','untrained proposals','all historical checkpoint sweep'],checkpoint_transfers='No main-building / AI-building weight transfers performed by this scope expansion')
    for path in [OUT/'scope_20tb_best.json',REPORT/'20tb_checkpoint_scope.json']:
        if path.exists():
            digest=hashlib.sha256(path.read_bytes()).hexdigest();archive=OUT/'corrections'/f'20tb_scope_before_{digest}.json'
            if not archive.exists():archive.write_bytes(path.read_bytes())
        write(path,record)
    with (REPORT/'20tb_checkpoint_scope.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('SCOPE',len(rows),'20TB anchors;',len(cfg['models']),'all models; pending branches',len(pending))

if __name__=='__main__':main()
