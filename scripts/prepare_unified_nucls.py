"""Materialize the full shared NuCLS geometry adapter without altering publisher folds."""
import json,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dinov3.eval.bio_detection.nucls import prepare_manifest,NuCLSDataset
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def main():
    source=ROOT/'outputs/02_eval_inputs/formal_v4/nucls_qc_manifest.json';manifest=prepare_manifest(source)
    target=source.with_name('nucls_native_manifest.json');temp=target.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(target)
    counts=collections.Counter(a['source_type'] for r in manifest['records'] for a in r['objects'])
    folds=[]
    for fold in range(1,6):
        tr=NuCLSDataset(manifest,fold,'train');te=NuCLSDataset(manifest,fold,'test')
        assert not {r['group_id'] for r in tr.records}&{r['group_id'] for r in te.records}
        folds.append({'fold':fold,'train':len(tr),'test_or_selection':len(te)})
    result={'status':'prepared_pending_target_validation','images':len(manifest['records']),'retained_annotation_types':dict(counts),
        'genuine_segmentation_objects':sum(a['genuine_mask'] for r in manifest['records'] for a in r['objects']),
        'quarantined_by_author_crop_rule':len(manifest['quarantined_annotations']),'folds':folds,'manifest':str(target)}
    (OUT/'nucls_native_geometry.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':main()
