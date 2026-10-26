"""Certify complete author train/tuning/public-test members without cross-set joins."""
import csv,hashlib,json,zipfile
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
DATA=Path('/mnt/huawei_deepcad/benchmark/segmentation/Multimodal_CellSeg/neurips22_cellseg')
PROTOCOL='neurips22-author-public52-v2'
def image_shape(path):
    if path.suffix.lower() in {'.tif','.tiff'}:
        import tifffile
        with tifffile.TiffFile(path) as f:
            s=f.series[0];return (s.shape[s.axes.index('X')],s.shape[s.axes.index('Y')])
    with Image.open(path) as f:return f.size
def main():
    Image.MAX_IMAGE_PIXELS=None
    evidence=json.loads((DATA/'metadata/zenodo_record_10719375.json').read_text())
    checks={f['key']:f['checksum'] for f in evidence['files']}
    hashes={}
    for archive in ['Training-labeled.zip','Tuning.zip','Testing.zip']:
        p=DATA/'archives'/archive;digest=hashlib.file_digest(p.open('rb'),'md5').hexdigest()
        assert 'md5:'+digest==checks[archive],archive;hashes[archive]=digest
    groups={'train':('Training-labeled/Training-labeled',1000),'val':('Tuning/Tuning',101),'test':('Testing/Testing/Public',52)}
    rows={}
    for split,(relative,count) in groups.items():
        base=DATA/'extracted'/relative;items=[]
        for image_dir,label_dir in [('images','labels')]+([('WSI','WSI-labels')] if split=='test' else []):
            for image in sorted((base/image_dir).iterdir()):
                if not image.is_file():continue
                label=base/label_dir/(image.stem+'_label.tiff')
                assert label.is_file(),label
                shape=image_shape(image);label_shape=image_shape(label)
                assert shape==label_shape,(image,label,shape,label_shape)
                items.append(dict(image_path=str(image.relative_to(DATA)),mask_path=str(label.relative_to(DATA)),sample_id=f'{split}/{image.stem}',group_id=f'{split}/{image.stem}',width=shape[0],height=shape[1],source_dataset='author_'+split,has_mask='1'))
        assert len(items)==count,(split,len(items),count);rows[split]=items
    folder=OUT/'manifests'/PROTOCOL;folder.mkdir(parents=True,exist_ok=True)
    for split,items in rows.items():
        p=folder/f'{split}.csv'
        with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    record=dict(protocol=PROTOCOL,root=str(DATA),counts={k:len(v) for k,v in rows.items()},archive_md5=hashes,source='https://zenodo.org/records/10719375',split_origin='Author labeled training1000, tuning101, public testing50microscopy+2WSI; hidden400images lack released labels and are not scored',task='Existing binary semantic frozen probe, not official instance-segmentation leaderboard',group_limitation='Original image identity retained; biological donor and acquisition mapping not supplied',manifests={s:str(folder/f'{s}.csv') for s in rows},manifest_sha256={s:hashlib.sha256((folder/f'{s}.csv').read_bytes()).hexdigest() for s in rows},correction='Old global basename join paired Tuning images to Training masks; all old formal-static-v1 scores rejected; all models require fresh common fits')
    p=OUT/'multimodal_official_readiness.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(p);print(json.dumps(record),flush=True)
if __name__=='__main__':main()
