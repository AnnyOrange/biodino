"""Lossless tensor-prefix transport for the existing official bone-age cohort."""
import concurrent.futures,fcntl,hashlib,io,json,os,subprocess,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from manage_unified_ai_branches import ROOT,OUT,HOST,REMOTE_ROOT,REMOTE_OUT,PY,MODELS
sys.path.insert(0,str(ROOT))
from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder

PACK=OUT/'prepared_inputs/bone_age_spatial'
SOURCE=ROOT/'outputs/02_eval_inputs/formal_v4/rsna_bone_age_manifest.json'

def write(path,data):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def prepare():
    PACK.mkdir(parents=True,exist_ok=True);raw=SOURCE.read_bytes();digest=hashlib.sha256(raw).hexdigest();cfg=json.loads(raw)
    if (PACK/'manifest.json').exists():
        meta=json.loads((PACK/'manifest.json').read_text());assert meta['canonical_manifest_sha256']==digest
        for name,expected in meta['archive_sha256'].items():
            with (PACK/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
        return
    torch.set_num_threads(1)
    encoder=object.__new__(Dinov3CkptEncoder);encoder.resize_size=256;encoder.image_size=224
    def one(pair):
        index,row=pair;path=Path(row['path'])
        with path.open('rb') as stream:source_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        with Image.open(path) as im:image=np.asarray(im).copy()
        assert image.ndim==2 and image.dtype==np.uint8
        tensor=torch.from_numpy(image.astype('float32')/255).expand(3,-1,-1).contiguous()
        spatial=encoder._resize_center_crop_tensor(tensor)
        mono=spatial[0].numpy().copy()
        assert torch.equal(torch.from_numpy(mono).expand(3,-1,-1),spatial)
        return index,mono,image if index<2 else None,source_sha
    archive=PACK/'spatial_inputs.npz';temp=PACK/f'spatial_inputs.{os.getpid()}.tmp';sources={};examples={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool,zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as target:
        for start in range(0,len(cfg['records']),32):
            batch=list(enumerate(cfg['records']))[start:start+32]
            for index,array,example,source_sha in pool.map(one,batch):
                buf=io.BytesIO();np.save(buf,array,allow_pickle=False);target.writestr(f'image_{index:06d}.npy',buf.getvalue())
                sources[cfg['records'][index]['path']]=source_sha
                if example is not None:examples[f'raw_{index}']=example
            if start%320==0:
                write(OUT/'states/ai_bone_age_preparation.json',dict(pid=os.getpid(),updated=time.time(),status='preparing_lossless_inputs',completed_inputs=min(start+32,len(cfg['records'])),required_inputs=len(cfg['records'])))
                print('PREPARED BONE INPUTS',min(start+32,len(cfg['records'])),len(cfg['records']),flush=True)
    temp.replace(archive);np.savez_compressed(PACK/'raw_validation_examples.npz',**examples)
    hashes={}
    for name in ['spatial_inputs.npz','raw_validation_examples.npz']:
        with (PACK/name).open('rb') as stream:hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    write(PACK/'manifest.json',dict(canonical_manifest_sha256=digest,n_records=len(cfg['records']),archive_sha256=hashes,source_sha256=sources,all_spatial_channels_exact=True,dtype='float32',shape_per_record=[224,224],transport='Lossless first channel of exactly equal three-channel outputs from shared tensor bilinear-antialias resize256/center224. Reexpand all three channels before identical normalization, channel routing and encoder.',independent_encoder_equivalence_required=True))

def main():
    with (OUT/'locks/ai_bone_age_preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare()
        state=OUT/'states/ai_bone_age_preparation.json'
        write(state,dict(pid=os.getpid(),updated=time.time(),status='transferring_verified_input_pack',checkpoint_transfer=False))
        files=['scripts/run_unified_paper_bone_age.py','scripts/run_unified_paper_global.py','scripts/verify_unified_global_inputs.py','dinov3/eval/bio_frozen_eval/encoder.py',str(SOURCE.relative_to(ROOT)),str(PACK.relative_to(ROOT))+'/']
        subprocess.run(['rsync','-aR','--partial','--partial-dir=.rsync-partial',*files,HOST+':'+REMOTE_ROOT+'/'],cwd=ROOT,check=True)
        code='''import fcntl,json,pathlib,hashlib
root=pathlib.Path(__ROOT__);out=root/'outputs/02_eval_runs/unified_paper';pack=out/'prepared_inputs/bone_age_spatial';meta=json.loads((pack/'manifest.json').read_text())
source=root/'outputs/02_eval_inputs/formal_v4/rsna_bone_age_manifest.json';assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['canonical_manifest_sha256']
for name,digest in meta['archive_sha256'].items():
 with (pack/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest
with (out/'locks/model_admission.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=out/'bone_age_queue.json';q=json.loads(path.read_text()) if path.exists() else {'tasks':[]};known={j['key'] for j in q['tasks']}
 for model in __MODELS__:
  key=model+'__bone_age__official_validation'
  if key in known:continue
  q['tasks'].append(dict(key=key,model=model,dataset='xray_bone',task='bone_age_regression',entrypoint='run_unified_paper_bone_age.py',args=['--model',model],claim_key=key,min_free_mib=10000,expected_peak_mib=5500,result=str(out/'results'/model/'bone_age/component_result.json'),log=str(out/'logs'/f'{key}.log'),requires=[str(pack/'manifest.json'),str(source)]))
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2)+'\\n');tmp.replace(path)
 print('VERIFIED RESIDENT BONE AGE QUEUE',len(q['tasks']))
'''.replace('__ROOT__',repr(REMOTE_ROOT)).replace('__MODELS__',repr(MODELS))
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,PY+' -u -'],input=code,text=True,check=True)
        write(state,dict(pid=os.getpid(),updated=time.time(),status='resident_queue_admitted',models=MODELS,queue=REMOTE_OUT+'/bone_age_queue.json',source_manifest=str(SOURCE),input_pack=str(PACK),checkpoint_transfer=False))

if __name__=='__main__':
    if '--watch' not in sys.argv:main()
    else:
        while True:
            state=OUT/'states/ai_bone_age_preparation.json'
            if not state.exists() or json.loads(state.read_text()).get('status')!='resident_queue_admitted':
                try:main()
                except BlockingIOError:pass
                except Exception:traceback.print_exc()
            time.sleep(60)
