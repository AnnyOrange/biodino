from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dinov3.eval.bio_detection.spheroid import prepare,centroid_metrics
r=prepare('/mnt/huawei_deepcad/benchmark/ood/ood_detection/datasets/Spheroid_LSFM/extracted/dataset_upload/SH-SY5Y')
p=ROOT/'outputs/02_eval_runs/unified_paper/spheroid_spatial_audit.json';p.write_text(json.dumps(r,indent=2));print(r['status'],len(r['records']),len(r['overlap_pairs']),sum(x['identical_overlap_pixels'] for x in r['overlap_pairs']))
assert centroid_metrics([[10,100,100]],[[10,100,100]])['f1']==1
assert centroid_metrics([[10,100,100]],[])['f1']==0
assert centroid_metrics([[10,100,100]],[[20,100,100]])['f1']==0
