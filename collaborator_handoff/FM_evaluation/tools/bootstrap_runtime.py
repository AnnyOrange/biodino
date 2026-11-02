#!/usr/bin/env python3
"""Install and byte-verify pinned CTC runtime dependencies, without running evaluations."""
import argparse,hashlib,json,shutil,subprocess,sys,urllib.request,zipfile
from pathlib import Path
P=Path(__file__).resolve().parents[1]
COMMIT='59481c48a62d4376fe34bed3e3606b4ec4d60972'
WHEEL='imagecodecs-2026.3.6-cp311-abi3-manylinux_2_28_x86_64.whl'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--repo',type=Path,required=True);x=a.parse_args();repo=x.repo.resolve()
 if sys.version_info<(3,11):raise RuntimeError('Python 3.11 or later required')
 locks=json.loads((P/'RUNTIME_SHA256.json').read_text());rt=repo/'outputs/02_eval_runtime';rt.mkdir(parents=True,exist_ok=True)
 ctc=rt/'py-ctcmetrics'
 if not ctc.exists():
  subprocess.run(['git','clone','--no-checkout','https://github.com/CellTrackingChallenge/py-ctcmetrics.git',str(ctc)],check=True)
  subprocess.run(['git','-C',str(ctc),'checkout','--detach',COMMIT],check=True)
 if subprocess.check_output(['git','-C',str(ctc),'rev-parse','HEAD'],text=True).strip()!=COMMIT:raise ValueError('CTC source commit mismatch')
 wheel=rt/'imagecodecs_wheel'/WHEEL;wheel.parent.mkdir(exist_ok=True)
 key='checkout/outputs/02_eval_runtime/imagecodecs_wheel/'+WHEEL;expected=locks[key]
 if not wheel.exists():
  with urllib.request.urlopen('https://pypi.org/pypi/imagecodecs/2026.3.6/json',timeout=60) as f:meta=json.load(f)
  entry=next(e for e in meta['urls'] if e['filename']==WHEEL)
  if entry['digests']['sha256']!=expected:raise ValueError('PyPI wheel digest differs from locked artifact')
  tmp=wheel.with_suffix('.partial')
  with urllib.request.urlopen(entry['url'],timeout=120) as f,tmp.open('wb') as out:shutil.copyfileobj(f,out)
  if sha(tmp)!=expected:raise ValueError('Downloaded wheel hash mismatch')
  tmp.replace(wheel)
 if sha(wheel)!=expected:raise ValueError('Local wheel hash mismatch')
 vendor=rt/'imagecodecs_py311_wheel_2026.3.6';vendor.mkdir(exist_ok=True)
 with zipfile.ZipFile(wheel) as z:
  for info in z.infolist():
   target=vendor/info.filename
   if not target.resolve().is_relative_to(vendor.resolve()):raise ValueError('Unsafe wheel member')
   if info.is_dir():continue
   if not target.exists():
    target.parent.mkdir(parents=True,exist_ok=True)
    with z.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
 for name,h in locks.items():
  target=repo/name[9:]
  if not target.is_file() or sha(target)!=h:raise ValueError('Runtime file mismatch: '+name)
 print('PINNED RUNTIME PASS',len(locks),'files')
if __name__=='__main__':main()
