"""Losslessly stage DINOv3's existing deterministic RGB spatial transform."""
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
import torch
from numpy.lib.format import write_array
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
def main():
    from torchvision.transforms import v2
    from dinov3.data.transforms import make_classification_eval_transform
    from run_unified_paper_ood import inputs,_collate,sha
    from verify_unified_ood_inputs import fingerprint
    from unified_paper_ood_packed import PreparedRGB
    from ood_input_fix import REVISION
    torch.set_num_threads(2);locked,_,dataset,_,records=inputs('cryo')
    source=json.loads(Path(locked['source_records']).read_text());fp=OUT/'validation/ood_input_fingerprints/cryo.json';expected=json.loads(fp.read_text())
    transform=make_classification_eval_transform(resize_size=256,crop_size=224);spatial=v2.Compose(transform.transforms[:-1]);base=transform.transforms[-1]
    folder=OUT/'prepared_inputs/ood_fixed_cryo_spatial';folder.mkdir(parents=True,exist_ok=True);target=folder/'cryo_inputs.npz';labels=[];kinds=[];hashes=[];i=0;raw_examples={}
    with zipfile.ZipFile(target.with_suffix('.tmp'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for images,ys,_ in torch.utils.data.DataLoader(dataset,batch_size=32,num_workers=4,shuffle=False,collate_fn=_collate):
            for image,label in zip(images,ys):
                assert fingerprint(image)==expected['fingerprints'][i]
                tensor=spatial(image);assert tensor.dtype==torch.uint8 and tuple(tensor.shape)==(3,224,224)
                assert torch.equal(transform(image),base(tensor)),f'Staged preprocessing differs at record{i}'
                # Cryo RGB repeats a gray plane. Preserve all channels if that is ever false.
                array=tensor.numpy();stored=array[0] if np.array_equal(array[0],array[1]) and np.array_equal(array[0],array[2]) else array
                restored=PreparedRGB(stored);assert torch.equal(tensor,restored.prepared_rgb)
                hashes.append(fingerprint(restored));labels.append(int(label));kinds.append('dinov3_spatial_uint8_v1')
                with archive.open(f'image_{i:06d}.npy','w',force_zip64=True) as stream:write_array(stream,stored,allow_pickle=False)
                if i<2:raw_examples[f'image_{i:06d}']=np.asarray(image)
                i+=1
            if i%3200==0:print('EXACT SPATIAL INPUTS',i,'/',len(dataset),flush=True)
    assert i==80000;target.with_suffix('.tmp').replace(target);np.savez(folder/'raw_validation_examples.npz',**raw_examples)
    with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    payload=dict(transport_stage='dinov3_spatial_uint8_v1',input_revision=REVISION,source_records_sha256=locked['source_records_sha256'],selected_sources=source['cryo'],identities=records,identity_sha256=sha(records),kinds=kinds,labels=labels,archive_file=target.name,archive_sha256={target.name:digest},input_fingerprints_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),transport_fingerprints=hashes,all_normalized_inputs_elementwise_equal=True,raw_validation_examples='raw_validation_examples.npz',preprocessing=str(transform),provenance='All80000 raw inputs match canonical fingerprints; exact existing ToImage/Resize256bicubic/CenterCrop224 prefix; every full normalized tensor equals unstaged pipeline; gray-plane repetition and zip compression are lossless')
    (folder/'manifest.json').write_text(json.dumps(payload)+'\n');print('EXACT SPATIAL TRANSPORT COMPLETE',i,target.stat().st_size,flush=True)
if __name__=='__main__':main()
