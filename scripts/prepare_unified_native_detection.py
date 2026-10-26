"""Prepare the complete released native box tasks and a shared primary-last recipe."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dinov3.eval.bio_detection.native import build_bccd_manifest,build_bbbc041_manifest,sha256
OUT=ROOT/'outputs/02_eval_runs/unified_paper';INPUT=ROOT/'outputs/02_eval_inputs/formal_v4'

def main():
 base=Path('/mnt/huawei_deepcad/benchmark/ood/ood_detection/datasets')
 paths={}
 for name,build,path in [('bccd',build_bccd_manifest,base/'BCCD/extracted/BCCD_Dataset-master'),('bbbc041',build_bbbc041_manifest,base/'BBBC041/extracted/malaria')]:
  manifest=build(path);target=INPUT/f'{name}_native_manifest.json'
  target.write_text(json.dumps(manifest,indent=2)+'\n');paths[name]={'path':str(target),'sha256':sha256(target),'counts':manifest['split_counts']}
  print(name,manifest['split_counts'],flush=True)
 registry=ROOT/'Evaluation Rules/native_detection_candidates.json'
 recipe=json.loads(registry.read_text())
 recipe.update(protocol_id='unified-native-box-primary-last-v1',source_registry=str(registry),source_registry_sha256=sha256(registry),batch_size=2,training_seeds=list(range(5)),
   feature_scope='Primary-last final spatial map for every encoder, matching the established FM comparison view; unfrozen exploratory4-even feature search is separate and has no valid historical score in this comparison',
   selection='Per model/dataset, seed0 trains both prescribed learning rates on fixed training members; compare final-epoch validation AP; tie chooses lower learning rate. Selected seed0 head evaluated once on test, seeds1-4 freshly trained with the frozen selected recipe.',
   seed_semantics='training_randomness: head initialization, training shuffle, RPN/RoI sampling; fixed membership',manifests=paths)
 recipe['invalid_candidate_rule']='Across every model/dataset, nonfinite training candidates are ineligible; archive failure and select completed prescribed candidates using validation AP only; all-invalid tasks remain failed'
 recipe['search']['features']=['last'];recipe['selection_model_budget']='fixed declared model checkpoint'
 (OUT/'native_detection_config.json').write_text(json.dumps(recipe,indent=2)+'\n')

if __name__=='__main__':main()
