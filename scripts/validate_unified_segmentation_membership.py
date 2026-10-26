"""Exercise source identity adapters against retained real prediction labels."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from unified_paper_segmentation_artifacts import attach_source_manifest
def main():
    cells=json.loads((OUT/'biodino_segmentation_inventory.json').read_text())['cells'];results=[]
    for dataset in ['livecell','pannuke','tissuenet']:
        cell=next(c for c in cells if c['model']=='biodino_1tb_l' and c['dataset']==dataset)
        directory=OUT/'results'/cell['model']/'segmentation'/dataset/cell['split']/'E20/seed0'
        result=attach_source_manifest(cell,directory);manifest=json.loads(Path(result['sample_manifest']).read_text())
        results.append(dict(dataset=dataset,n_samples=len(manifest['samples']),status=result,unique_ids=len({r['sample_id'] for r in manifest['samples']})))
        print('SOURCE IDENTITIES VERIFIED',dataset,len(manifest['samples']),flush=True)
    path=OUT/'validation/segmentation_source_identity_adapters.json';path.write_text(json.dumps(dict(updated=time.time(),results=results),indent=2)+'\n')
if __name__=='__main__':main()
