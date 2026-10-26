"""Lock every released scored image/instance map; no partial scoring cohort."""
import csv,hashlib,json,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from dinov3.eval.bio_segmentation.datasets.bbbc038 import BBBC038Dataset
from dinov3.eval.bio_segmentation.datasets.bbbc038_official import decode_rle
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
SOURCE=Path('/mnt/huawei_deepcad/benchmark/segmentation/bbbc038')
MANIFEST=ROOT/'outputs/02_eval_inputs/formal_v4/bbbc038_official_manifest.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    target=OUT/'prepared_inputs/bbbc038_official';target.mkdir(parents=True,exist_ok=True)
    splits={};source_hashes={};all_ids=set();pixel_roles=collections.defaultdict(set)
    trainroot=SOURCE/'extracted/stage1_train';dirs=sorted(p for p in trainroot.iterdir() if p.is_dir())
    assert len(dirs)==670,len(dirs)
    loader=BBBC038Dataset([str(p/'images'/f'{p.name}.png')for p in dirs],[str(p/'masks')for p in dirs],size=None,augment=False)
    entries=[('train',p.name,p/'images'/f'{p.name}.png',loader.get_instance_map(i),None)for i,p in enumerate(dirs)]
    for stage,filename in [('stage1','stage1_solution.csv'),('stage2','stage2_solution_final.csv')]:
        solution=SOURCE/'official_labels'/filename;source_hashes[str(solution)]=sha(solution);grouped=collections.defaultdict(list)
        for row in csv.DictReader(solution.open()):
            if row['Usage'].lower()!='ignored':grouped[row['ImageId']].append(row)
        assert len(grouped)==(65 if stage=='stage1' else 106),(stage,len(grouped))
        for image_id,rows in sorted(grouped.items()):
            h,w=int(rows[0]['Height']),int(rows[0]['Width']);instances=np.zeros((h,w),dtype=np.int32)
            for inst_id,row in enumerate(rows,1):
                assert (int(row['Height']),int(row['Width']))==(h,w)
                mask=decode_rle(row['EncodedPixels'],h,w)>0
                assert not np.any(instances[mask]),'Overlapping published instance masks'
                instances[mask]=inst_id
            image=SOURCE/'extracted'/('stage1_test'if stage=='stage1'else'stage2_test_final')/image_id/'images'/f'{image_id}.png'
            entries.append((stage,image_id,image,instances,rows[0]['Usage']))
    for split,image_id,image,instances,usage in entries:
        assert image_id not in all_ids;all_ids.add(image_id)
        with Image.open(image)as im:array=np.asarray(im.convert('RGB'));assert array.shape[:2]==instances.shape
        digest=hashlib.sha256(array.tobytes()+str(array.shape).encode()).hexdigest();pixel_roles[digest].add(split)
        path=target/f'{image_id}.npy'
        if path.exists():assert np.array_equal(np.load(path),instances)
        else:np.save(path,instances.astype(np.int32))
        row=dict(sample_id=image_id,group_id=image_id,image=str(image),image_sha256=sha(image),instances=str(path),instances_sha256=sha(path),shape=list(instances.shape),n_instances=int(instances.max()),usage=usage)
        splits.setdefault(split,[]).append(row)
        if split=='train':
            for p in sorted((image.parent.parent/'masks').glob('*.png')):source_hashes[str(p)]=sha(p)
    assert not any('train'in roles and len(roles)>1 for roles in pixel_roles.values()),'Exact train/test image overlap'
    manifest=dict(dataset='bbbc038-official',protocol='dsb2018-released-scoring-sets-frozen-hovernet-fixed50-v1',splits=splits,source_hashes=source_hashes,
        source='https://bbbc.broadinstitute.org/BBBC038',training='All670 official training images; fixed50epoch frozen-encoder decoder training; no validation search or test selection',
        test='All65 stage1 and106 non-Ignored stage2 scoring images, reported separately; same trained head',
        grouping='Released ImageId; parent source-image/patient mapping not supplied by released aggregate metadata.xlsx. Conditional image inference only.',
        primary_metric='dsb_mean_precision',metric_definition='Per-image TP/(TP+FP+FN), strict IoU>.50:.05:.95, then mean images. Official released test objects; local scorer, not leaderboard submission.')
    encoded=json.dumps(manifest,indent=2,sort_keys=True)+'\n'
    if MANIFEST.exists():assert json.loads(MANIFEST.read_text())==manifest
    else:MANIFEST.write_text(encoded)
    print(json.dumps(dict(manifest=str(MANIFEST),sha256=sha(MANIFEST),counts={k:len(v)for k,v in splits.items()},source_files=len(source_hashes))))
if __name__=='__main__':main()
