"""Fingerprint exact shared pre-encoder samples for resident-host equivalence."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
def fingerprint(image):
    if hasattr(image,'prepared_rgb'):
        a=image.prepared_rgb.numpy();h=hashlib.sha256(json.dumps(['dinov3_spatial_uint8_v1',str(a.dtype),list(a.shape)],separators=(',',':')).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()
    kind='tensor_CHW' if torch.is_tensor(image) else 'PIL_HWC' if hasattr(image,'mode') else 'array'
    a=image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    h=hashlib.sha256(json.dumps([kind,str(a.dtype),list(a.shape)],separators=(',',':')).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()
def main():
    from run_unified_paper_ood import inputs,_collate,sha
    from ood_input_fix import CorrectedIDDataset,REVISION
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['xray','cryo'],required=True);p.add_argument('--export-id',action='store_true');p.add_argument('--export-ood',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    locked,ids,ood,identity,other=inputs(a.task);folder=OUT/'validation/ood_input_fingerprints';folder.mkdir(parents=True,exist_ok=True)
    if a.export_id or a.export_ood:
        import zipfile
        from numpy.lib.format import write_array
        part='id' if a.export_id else a.task;records=identity if a.export_id else other;dataset=CorrectedIDDataset(ids) if a.export_id else ood
        packed=OUT/'prepared_inputs'/f'ood_fixed_{part}';packed.mkdir(parents=True,exist_ok=True);target=packed/f'{part}_inputs.npz'
        expected=json.loads((folder/f'{part}.json').read_text());labels=[];kinds=[];hashes=[];i=0
        with zipfile.ZipFile(target.with_suffix('.tmp'),'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for images,batch_labels,_ in torch.utils.data.DataLoader(dataset,batch_size=32,num_workers=4,shuffle=False,collate_fn=_collate):
                for image,label in zip(images,batch_labels):
                    digest=fingerprint(image);assert digest==expected['fingerprints'][i];hashes.append(digest);labels.append(int(label))
                    kinds.append('tensor_CHW' if torch.is_tensor(image) else 'PIL_HWC' if hasattr(image,'mode') else 'array');array=image.numpy() if torch.is_tensor(image) else np.asarray(image)
                    with z.open(f'image_{i:06d}.npy','w',force_zip64=True) as stream:write_array(stream,array,allow_pickle=False)
                    i+=1
                if i%3200==0:print('PACK',part,i,'/',len(dataset),flush=True)
        target.with_suffix('.tmp').replace(target)
        with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        source=json.loads(Path(locked['source_records']).read_text())
        payload=dict(input_revision=REVISION,source_records_sha256=locked['source_records_sha256'],selected_sources=source['id'] if a.export_id else source[a.task],identities=records,identity_sha256=sha(records),kinds=kinds,labels=labels,archive_file=target.name,archive_sha256={target.name:digest},input_fingerprints_sha256=hashlib.sha256((folder/f'{part}.json').read_bytes()).hexdigest(),provenance='Lossless arrays emitted by exact shared dataset; every array fingerprint checked before packing; no resize, quantization or membership change')
        (packed/'manifest.json').write_text(json.dumps(payload)+'\n');print('PACKED FIXED INPUTS COMPLETE',part,len(records),target.stat().st_size,flush=True);return
    for name,dataset,records in [('id',CorrectedIDDataset(ids),identity),(a.task,ood,other)]:
        path=folder/f'{name}.json'
        if path.exists():
            d=json.loads(path.read_text());assert d['identity_sha256']==sha(records) and d['input_revision']==REVISION;continue
        hashes=[]
        for images,_,_ in torch.utils.data.DataLoader(dataset,batch_size=32,num_workers=4,shuffle=False,collate_fn=_collate):
            hashes.extend(fingerprint(image) for image in images)
            if len(hashes)%3200==0:print('FINGERPRINT',name,len(hashes),'/',len(dataset),flush=True)
        assert len(hashes)==len(records)
        d=dict(part=name,n_samples=len(hashes),identity_sha256=sha(records),input_revision=REVISION,source_records_sha256=locked['source_records_sha256'],fingerprints=hashes,definition='SHA256 of type/layout,dtype,shape and exact contiguous pre-encoder array bytes; original shared loader and corrected CyclOps normalization')
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d)+'\n');tmp.replace(path);print('INPUT FINGERPRINTS COMPLETE',name,len(hashes),flush=True)
if __name__=='__main__':main()
