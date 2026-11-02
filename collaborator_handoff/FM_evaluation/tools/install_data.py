#!/usr/bin/env python3
"""Install the supplied archives into an isolated checkout and benchmark tree."""
import argparse,json,shutil,stat,subprocess,sys,tarfile,zipfile,os
from pathlib import Path,PurePosixPath
P=Path(__file__).resolve().parents[1]
def safe_target(root,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError(f'Unsafe member: {name}')
 out=root.joinpath(*p.parts)
 if not out.resolve().is_relative_to(root.resolve()):raise ValueError(f'Escaping member: {name}')
 return out
def write_member(root,name,source):
 target=safe_target(root,name)
 if target.exists():return # Never overwrite; subsequent full SHA audit verifies it.
 target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.installing')
 with tmp.open('wb') as dest:shutil.copyfileobj(source,dest,4*1024*1024)
 tmp.replace(target)
def extract(archive,root,split_roots=None):
 print('EXTRACT',archive.name,flush=True)
 def destination(name):
  if split_roots:
   first,sep,rest=name.partition('/')
   return (split_roots[first],rest) if first in split_roots else (root,name)
  return root,name
 if zipfile.is_zipfile(archive):
  with zipfile.ZipFile(archive) as z:
   for i in z.infolist():
    if i.is_dir() or i.filename=='PACKAGE_FILES.json' or i.filename.startswith('__MACOSX/'):continue
    if stat.S_ISLNK(i.external_attr>>16):raise ValueError('Archive symlink rejected')
    base,name=destination(i.filename)
    with z.open(i) as f:write_member(base,name,f)
 else:
  with tarfile.open(archive,'r|*') as t:
   for i in t:
    if i.isdir():continue
    if not i.isfile():raise ValueError(f'Non-regular tar member rejected: {i.name}')
    base,name=destination(i.name)
    with t.extractfile(i) as f:write_member(base,name,f)
def main():
 a=argparse.ArgumentParser();a.add_argument('--archives',type=Path,required=True);a.add_argument('--repo',type=Path,default=Path('/mnt/huawei_deepcad/dinov3'));a.add_argument('--benchmark',type=Path,default=Path('/mnt/huawei_deepcad/benchmark'));a.add_argument('--prepare-ctc',action='store_true');x=a.parse_args();r=x.repo.resolve();b=x.benchmark.resolve();b.mkdir(parents=True,exist_ok=True)
 if r!=Path('/mnt/huawei_deepcad/dinov3'):raise ValueError('Use the canonical container mount for installation')
 names={p.name:p for p in x.archives.rglob('*') if p.is_file()}
 specs={n:b for n in ['deepcad_current_run_datasets_20260608.zip','deepcad_missing_datasets_20260616.zip','ood.zip']}
 specs.update({'CHAMMI.tar.gz':b/'Classification','retrieval_hpa-subcellular.tar':b,'retrieval_rxrx3-core.tar':r,'BCCD_Dataset-master.zip':b/'ood/ood_detection/datasets/BCCD/extracted','malaria.zip':b/'ood/ood_detection/datasets/BBBC041/extracted','dataset_upload.zip':b/'ood/ood_detection/datasets/Spheroid_LSFM/extracted','FILM_data.zip':b/'ood/ood_classification/datasets/FILM/extracted'})
 for n in ['Training-labeled','Tuning','Testing']:specs[n+'.zip']=b/'segmentation/Multimodal_CellSeg/neurips22_cellseg/extracted'/n
 new=['pcam_full.zip','nct_crc_he_full_classification.zip','nucls_corrected_qc.zip','bbbc038_official_supplement.zip','rsna_bone_age_complete.zip','current_input_supplement.zip','nucls_qc_labels.zip','rsna_pneumonia_eval_tiff.zip']
 canonical=json.loads((r/'outputs/02_eval_runs/unified_paper/validation/ctc_canonical_manifest.json').read_text())
 required=set(specs)|set(new)|{'rxrx1_v4_complete_core.zip'}|{d+'.zip' for d in canonical['domains']}
 missing=sorted(required-set(names))
 if missing:raise ValueError('Missing archives: '+', '.join(missing))
 for n,dest in specs.items():extract(names[n],dest)
 for n in new:extract(names[n],r,{'benchmark':b,'outputs':r/'outputs','dinov3':r/'dinov3'})
 for sub in ['train','test']:
  parent=b/'ood/ood_detection/datasets/Spheroid_LSFM/extracted/dataset_upload/SH-SY5Y';extract(parent/(sub+'.zip'),parent/sub)
 for family,dest in [('CoNIC_Cell_Count','segmentation/conic/extracted'),('LIVECell_Cell_Count','segmentation/LIVECell/LIVECell_dataset_2021')]:
  link=b/'Regression'/family/'data';link.parent.mkdir(parents=True,exist_ok=True)
  if not link.exists():link.symlink_to(b/dest,target_is_directory=True)
 for src in (P/'benchmark_overlay').rglob('*'):
  if src.is_file():
   with src.open('rb') as f:write_member(b,str(src.relative_to(P/'benchmark_overlay')),f)
 prepared=r/'outputs/02_eval_runs/unified_paper/prepared_inputs';prepared.mkdir(parents=True,exist_ok=True)
 links={prepared/'rxrx1_v4_complete_core.zip':names['rxrx1_v4_complete_core.zip']}
 for d in canonical['domains']:links[prepared/'ctc_official_archives'/f'{d}.zip']=names[d+'.zip']
 for target,source in links.items():
  target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():target.symlink_to(source.resolve())
 if x.prepare_ctc:
  env=os.environ.copy();env['UNIFIED_PAPER_PYTHON']=sys.executable
  subprocess.run([sys.executable,str(r/'scripts/prepare_unified_ctc_resident_data.py')],cwd=r,env=env,check=True)
 print('INSTALL COMPLETE; run accept.py data --full-hash before evaluations')
if __name__=='__main__':main()
