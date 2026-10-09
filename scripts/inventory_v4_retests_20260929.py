#!/usr/bin/env python3
"""List protocol-matched retests across FM, 1TB and 5TB trajectories."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924/v4_all_models.json'
OUT = ROOT / 'outputs/00_reports/v4_retest_amendment_20260929'


def matched_conic(observation):
    if observation.get('provenance') == 'legacy_accepted':
        return False
    if observation.get('provenance') in ('v4_completion_validated', 'v4_list_raw'):
        return True
    source = observation.get('source','')
    if ';' in source or not source.endswith('.json'):
        return False
    path = Path(source)
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (result.get('batch_size') == 8 and result.get('epochs') == 5
            and result.get('conic_split_protocol') == 'official-baseline-fold0-nested-v1'
            and 'test_patch_f1' in result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(SOURCE.read_text())
    rows = []
    for name, model in sorted(data.items()):
        if model['group'] not in ('fm14', 'hs0_1tb', 'hs6_1tb', 'hs6_5tb'):
            continue
        for step, checkpoint in sorted(model['checkpoints'].items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1):
            if name == 'hs6_l_5tb_gram12687' and step == '12687':
                continue
            cells = checkpoint.get('cells', {})
            training = checkpoint.get('training', {})
            if name == 'hs6_l_5tb_no_gram' and int(step) >= 29767:
                host = '5090-hxw-xzj'
            else:
                host = training.get('evidence_host') or 'shared/resolve_checkpoint'
            base = dict(model=name, group=model['group'], checkpoint=step, evidence_host=host,
                        checkpoint_path=training.get('checkpoint',''))
            observations = cells.get('segmentation', {}).get('monuseg', {}).get('observations', [])
            old37 = [o for o in observations if o.get('provenance') == 'legacy_accepted'
                     and o.get('budget_epochs') == 20 and 'three_seed' in o.get('note','')]
            rows.append(dict(base, family='segmentation', dataset='monuseg', target='37-pool E20 and E50 / three seeds',
                             status='OLD_37_E20_OBSERVATION_REVALIDATE' if old37 else 'RETEST_REQUIRED',
                             existing_source=old37[0]['source'] if old37 else ''))
            observed = cells.get('detection_proxy', {}).get('conic', {}).get('observations', [])
            candidates = [o for o in observed if matched_conic(o)]
            rows.append(dict(base, family='detection_proxy', dataset='conic',
                             target='official-baseline-fold0-nested-v1 / B8 / 224 / 5 epochs',
                             status='FINGERPRINT_VALIDATE' if candidates else 'RETEST_REQUIRED',
                             existing_source=candidates[0]['source'] if candidates else ''))
            if name == 'hs6_l_5tb_gram12687':
                for dataset in ('rxrx3-core', 'lc25000', 'nct-crc-he-100'):
                    r = cells.get('retrieval', {}).get(dataset, {})
                    c = cells.get('clustering', {}).get(dataset, {})
                    if not (r.get('observations') and c.get('observations')):
                        rows.append(dict(base,family='retrieval+clustering',dataset=dataset,
                                         target='fixed v4 manifest / Recall@1 and NMI',status='MISSING_RETEST',existing_source=''))
                for dataset in ('bbbc038','conic','livecell'):
                    c = cells.get('detection_proxy', {}).get(dataset, {})
                    matched = [o for o in c.get('observations',[]) if o.get('provenance') not in ('legacy_accepted',)]
                    if not matched:
                        rows.append(dict(base,family='detection_proxy',dataset=dataset,
                                         target='matched B8',status='MISSING_OR_B4_RETEST',existing_source=''))
    with (OUT/'TARGETS.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    counts = Counter((r['group'],r['family'],r['status']) for r in rows)
    summary = dict(models=len(data), checkpoints=sum(len(m['checkpoints']) for m in data.values()),
                   targets=len(rows), counts=[dict(group=g,family=f,status=s,n=n) for (g,f,s),n in sorted(counts.items())],
                   note='Inventory only; every reused observation needs exact split/probe fingerprint validation.')
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
