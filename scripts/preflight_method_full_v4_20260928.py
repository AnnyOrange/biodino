"""Freeze OOD population identities and admit the existing full CTC runner."""
import hashlib,json,subprocess,sys
from pathlib import Path
from collections import Counter
from dinov3.eval.eval_ood.datasets import XrayTomogramSliceDataset,CryoParticleDataset,build_id_reference_dataset

REPO=Path('/mnt/huawei_deepcad/dinov3')
ROOT=REPO/'outputs/02_eval_runs/hs6_l5_deepcad_method_v4_20260927'
SOURCE=Path(__file__).resolve().parents[1]
BENCH=Path('/mnt/huawei_deepcad/benchmark')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    x=XrayTomogramSliceDataset(BENCH/'ood',slices_per_volume=8,input_mode='three_slices',percentiles=(.5,99.5))
    xr=[(r.volume_id,r.tomo_id,r.variant,r.z_index,r.raw_shape_xyz,r.raw_path.stat().st_size) for r in x.records]
    assert len(xr)==992 and len({r.volume_id for r in x.records})==124
    c=CryoParticleDataset(BENCH/'ood',max_projects=None,max_particles_per_project=20000,
                         max_per_class=None,percentiles=(.5,99.5),invert=False,seed=0)
    cr=[(r.project_id,r.cs_path.name,r.mrc_path.name,r.particle_index,r.class_id) for r in c.records]
    counts=Counter(r.project_id for r in c.records)
    assert dict(counts)=={'10535':20000,'11043':20000,'11387':20000,'11388':20000},counts
    ids=build_id_reference_dataset(BENCH,transform=None,max_samples=3000,dataset_names=('bloodmnist','bbbc048','cyclops'),seed=0)
    selected=[dict(source=d.source,source_size=len(d.dataset.dataset),indices=[int(i) for i in d.dataset.indices]) for d in ids.datasets]
    assert len(ids)==3000 and [len(d) for d in ids.datasets]==[1000]*3
    protocol=json.loads((SOURCE/'Evaluation Rules/protocol_v4.json').read_text())
    split=REPO/'outputs/02_eval_inputs/formal_v3/ctc/split_manifest.jsonl'
    assert hashlib.sha256(split.read_bytes()).hexdigest()==protocol['cell_tracking_splits']['ctc']['manifest_sha256']
    vendor=REPO/'outputs/02_eval_runtime/py-ctcmetrics'
    commit=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
    assert commit=='59481c48a62d4376fe34bed3e3606b4ec4d60972'
    for ndim in [2,3]:
        out=ROOT/'provenance'/f'ctc_oracle_{ndim}d.json'
        subprocess.run([sys.executable,str(SOURCE/'scripts/smoke_ctc_native_evaluator.py'),'--ndim',str(ndim),'--output',str(out)],check=True,cwd=SOURCE)
        assert json.loads(out.read_text())['status']=='PASS'
    admission=dict(status='READY_FOR_FULL_PROTOCOL_EXECUTION',full_suite_complete=False,source=str(SOURCE),
        ood=dict(xray=dict(count=len(xr),volumes=124,records_sha256=digest(xr)),cryo=dict(count=len(cr),projects=dict(counts),records_sha256=digest(cr)),
                 id=dict(count=3000,selection_sha256=digest(selected)),settings=dict(batch=64,seed=0,dtype='bf16',resize=256,crop=224,percentiles=[.5,99.5],k=10,id_train_fraction=.7)),
        ctc=dict(oracle_2d='PASS',oracle_3d='PASS',domains=20,folds=5,epochs=50,batch=8,commit=commit,split_sha256=hashlib.sha256(split.read_bytes()).hexdigest()))
    (ROOT/'provenance/ood_selected_records.json').write_text(json.dumps(dict(xray=xr,cryo=cr,id=selected)))
    (ROOT/'FULL_V4_ADMISSION.json').write_text(json.dumps(admission,indent=2))
    print(json.dumps(admission,indent=2))

if __name__=='__main__':main()
