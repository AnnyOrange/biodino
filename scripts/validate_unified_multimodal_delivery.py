"""Verify the corrected author-public52 fits and index their retained outputs."""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/02_eval_runs/unified_paper'
REPORT = ROOT / 'outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924'
PROTOCOL = 'neurips22-author-public52-v2'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    models = read(OUT / 'config.json')['models']
    cells = []
    for name in ['biodino_segmentation_inventory.json', 'fm_segmentation_inventory.json']:
        cells.extend(c for c in read(OUT / name)['cells']
                     if c['dataset'] == 'multimodal_cellseg' and not c.get('superseded_protocol'))
    assert {c['model'] for c in cells} == {m['model'] for m in models}
    records, identity = [], None
    for cell in cells:
        assert cell['split'] == PROTOCOL
        model = cell['model']
        for budget in [20, 50]:
            family = cell.get('result_family', 'segmentation')
            ci_path = OUT / 'statistics' / f'{model}__{family}__multimodal_cellseg__{PROTOCOL}__E{budget}__ci.json'
            ci = read(ci_path)
            assert ci['training_seeds'] == list(range(5)) and ci['n_test'] == 52
            scores, hashes = [], []
            for seed in range(5):
                p = OUT / 'results' / model / family / 'multimodal_cellseg' / PROTOCOL / f'E{budget}' / f'seed{seed}' / 'component_result.json'
                raw = read(p)
                assert (raw['model'], raw['split'], raw['budget'], raw['seed']) == (model, PROTOCOL, budget, seed)
                meta = raw['_meta']
                assert meta['probe_epochs'] == budget and meta['probe_batch_size'] == 32
                assert meta['used_train_samples'] == meta['full_train_samples'] == 1000
                assert len(meta['validation_history']) == budget
                assert meta['best_val_miou'] == max(r['mIoU'] for r in meta['validation_history'])
                assert meta['validation_history'][meta['best_epoch'] - 1]['mIoU'] == meta['best_val_miou']
                manifest_path = Path(raw['sample_manifest'])
                assert raw['sample_manifest_status'] == 'verified'
                assert sha(manifest_path) == raw['sample_manifest_sha256']
                manifest = read(manifest_path)
                members = [(r['sample_id'], r['group_id'], r['label_sha256']) for r in manifest['samples']]
                assert len(members) == 52 and len(set(x[0] for x in members)) == 52
                if identity is None:
                    identity = members
                assert identity == members, (model, budget, seed, 'different test identity')
                prediction = Path(raw['prediction_path'])
                assert prediction.is_file() and prediction.stat().st_size > 0
                digest = sha(p)
                source = next(s for s in ci['source'] if s['result_sha256'] == digest)
                score = raw['test']['mDice']
                assert abs(source['prediction_metric'] - score) < 1e-6
                scores.append(score)
                hashes.append(digest)
                records.append(dict(model=model, dataset='multimodal_cellseg', protocol=PROTOCOL,
                                    budget=budget, seed=seed, mDice=score, test_samples=52,
                                    selected_epoch=meta['best_epoch'], result=str(p), result_sha256=digest,
                                    prediction=str(prediction), sample_manifest=str(manifest_path),
                                    statistics=str(ci_path), inference_scope=manifest['group_limitation']))
            assert sorted(hashes) == sorted(s['result_sha256'] for s in ci['source'])
            assert np.allclose(scores, ci['per_seed_values'], rtol=0, atol=1e-12)
            assert abs(np.mean(scores) - ci['metrics']['mDice']['value']) < 1e-12
            assert abs(np.std(scores, ddof=1) - ci['seed_sd']) < 1e-12
        print('VERIFIED', model, '10 fits; 52 identical test members; 2 current CI files', flush=True)
    target = REPORT / 'multimodal_corrected_run_index.csv'
    tmp = target.with_suffix('.tmp.csv')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    tmp.replace(target)
    proof = dict(updated=time.time(), status='PASS', protocol=PROTOCOL, models=len(cells),
                 fits=len(records), budgets=[20, 50], seeds=list(range(5)), n_test=52,
                 same_test_members_and_label_hashes=True, prediction_files_retained=True,
                 current_statistics_sources_verified=True, run_index=str(target),
                 limitation='CI remains conditional on source images; donor/acquisition identities unavailable. This is not the hidden official challenge test or primary biological-group inference.')
    (OUT / 'validation/multimodal_corrected_delivery.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
