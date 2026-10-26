#!/usr/bin/env python3
"""Resumable publisher QC download through the approved process-local proxy."""
import concurrent.futures,csv,hashlib,json,os,time,re
from pathlib import Path
import gdown,requests
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';DEST=Path('/mnt/huawei_deepcad/benchmark/ood/ood_detection/datasets/NuCLS/raw/QC')
PROXY='http://127.0.0.1:17891'
def save(path,d):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(path)
def valid(path):
 if not path.is_file() or path.stat().st_size==0:return False
 try:
  if path.suffix.lower()=='.png':
   from PIL import Image
   with Image.open(path) as im:im.verify()
  elif path.suffix.lower()=='.csv':
   with path.open() as f:
    text=f.read(200)
    if '<html' in text.lower() or not ',' in text:return False
  return True
 except Exception:return False

def fetch(row):
 p=Path(row['local_path']);p.parent.mkdir(parents=True,exist_ok=True)
 if valid(p):return {'id':row['id'],'path':str(p),'status':'complete','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 error=''
 for attempt in range(4):
  try:
   u=f"https://drive.usercontent.google.com/download?id={row['id']}&export=download&authuser=0&confirm=t"
   r=requests.get(u,proxies={'http':PROXY,'https':PROXY},timeout=(20,120));r.raise_for_status()
   if 'text/html' in r.headers.get('Content-Type',''):raise ValueError('Google returned HTML instead of dataset file')
   temp=p.with_name(p.stem+'.partial'+p.suffix);temp.write_bytes(r.content)
   if not valid(temp):raise ValueError('Downloaded file failed PNG/CSV validation')
   temp.replace(p)
   return {'id':row['id'],'path':str(p),'status':'complete','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  except Exception as e:error=repr(e);time.sleep(min(2**attempt,8))
 return {'id':row['id'],'path':str(p),'status':'failed','error':error}

def main():
 os.environ.update(HTTP_PROXY=PROXY,HTTPS_PROXY=PROXY)
 manifest=OUT/'nucls_qc_download_manifest.json'
 if not manifest.exists():
  from bs4 import BeautifulSoup
  def enumerate_folder(folder, target):
   response=requests.get('https://drive.google.com/embeddedfolderview',params={'id':folder},proxies={'http':PROXY,'https':PROXY},timeout=(20,120));response.raise_for_status()
   listing=BeautifulSoup(response.text,'html.parser');records=[]
   for a in listing.select('a[href]'):
    url=a['href'];name=a.get_text().strip()
    child=re.search(r'/folders/([A-Za-z0-9_-]+)',url)
    file=re.search(r'/file/d/([A-Za-z0-9_-]+)',url)
    if not name or Path(name).name!=name:continue
    if child:records.extend(enumerate_folder(child.group(1),target/name))
    elif file:records.append({'id':file.group(1),'path':str(target/name),'local_path':str(target/name)})
   return records
  records=enumerate_folder('1eGlF9Dgu3WMEik4fqj0wJ13LKVufsfZ0',DEST/'QC')
  csv_count=sum(Path(x['local_path']).suffix=='.csv' and Path(x['local_path']).parent.name=='csv' for x in records)
  if csv_count!=1744:raise RuntimeError(f'Publisher QC CSV inventory incomplete: {csv_count}/1744')
  if len({x['local_path'] for x in records})!=len(records):raise RuntimeError('Duplicate published paths')
  save(manifest,records)
 rows=json.loads(manifest.read_text());print('Publisher files',len(rows),flush=True)
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for result in pool.map(fetch,rows):
   results.append(result)
   if len(results)%25==0:
    save(OUT/'nucls_qc_download_progress.json',{'files':len(rows),'processed':len(results),'complete':sum(x['status']=='complete' for x in results),'failures':[x for x in results if x['status']=='failed']});print('download',len(results),'/',len(rows),flush=True)
 save(OUT/'nucls_qc_file_validation.json',results)
 counts={name:sum(Path(x['path']).parent.name==name and x['status']=='complete' for x in results) for name in ['csv','rgb','mask']}
 save(OUT/'states/nucls_download.json',{'pid':os.getpid(),'host':'server','status':'download_complete_pending_cohort_validation' if all(x['status']=='complete' for x in results) else 'partial_download','expected_fovs':1744,'counts':counts,'manifest':str(manifest),'validation':str(OUT/'nucls_qc_file_validation.json'),'log':str(OUT/'logs/nucls_qc_download.log')})
 print('FINISHED',counts,flush=True)
if __name__=='__main__':main()
