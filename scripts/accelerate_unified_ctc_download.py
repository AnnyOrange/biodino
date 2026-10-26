"""Resume a copy of the sequential CTC archive with persistent aria2 segments.

Only the duplicate sequential download is retired after a complete aria2 exit,
exact archive length, canonical directory signature, and member metadata checks.
The existing data preparer still verifies every extracted source SHA256.
"""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper'
DOMAIN='Fluo-N3DL-TRIF'

def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def validate(path,canonical):
 expected=canonical['domains'][DOMAIN]['archive'];assert path.stat().st_size==expected['bytes']
 digest=hashlib.sha256()
 with zipfile.ZipFile(path) as z:
  for info in sorted(z.infolist(),key=lambda x:x.filename):digest.update(f'{info.filename}\t{info.file_size}\t{info.CRC}\n'.encode())
  assert digest.hexdigest()==expected['central_directory_sha256']
  for row in canonical['files']:
   if row['source_member'].startswith(DOMAIN+'/'):
    info=z.getinfo(row['source_member']);assert info.file_size==row['bytes'] and f'{info.CRC:08x}'==row['crc32']
 return dict(bytes=path.stat().st_size,central_directory_sha256=digest.hexdigest())

def run():
 canonical=json.loads((OUT/'validation/ctc_canonical_manifest.json').read_text())
 archive_dir=OUT/'prepared_inputs/ctc_official_archives';archive=archive_dir/f'{DOMAIN}.zip';partial=archive.with_suffix('.zip.part')
 state_path=OUT/'states/ctc_download_acceleration.json';proof_path=OUT/'validation/ctc_accelerated_archive.json'
 if proof_path.exists() and archive.exists():return
 if archive.exists():
  proof=validate(archive,canonical);write(proof_path,dict(status='existing_complete_archive',**proof));return
 workdir=archive_dir/'segmented';workdir.mkdir(exist_ok=True);work=workdir/archive.name
 state=dict(pid=os.getpid(),updated=time.time(),status='copying_verified_length_prefix',domain=DOMAIN,connections=16,original_partial=str(partial),archive=str(archive),work=str(work),expected_bytes=canonical['domains'][DOMAIN]['archive']['bytes'])
 write(state_path,state)
 if not work.exists():
  prefix=partial.stat().st_size if partial.exists() else 0;tmp=work.with_suffix('.prefix.tmp')
  with tmp.open('wb') as dest:
   if prefix:
    with partial.open('rb') as src:
     copied=0;last=time.monotonic()
     while copied<prefix:
      chunk=src.read(min(16*1024**2,prefix-copied))
      if not chunk:raise RuntimeError('Source partial truncated while copying')
      dest.write(chunk);copied+=len(chunk)
      if time.monotonic()-last>10:state.update(updated=time.time(),prefix_copied=copied,prefix_bytes=prefix);write(state_path,state);last=time.monotonic()
  tmp.replace(work);write(workdir/'prefix_provenance.json',dict(source=str(partial),bytes=prefix,created=time.time(),method='Copy fixed contiguous prefix while original sequential download continues'))
 log=OUT/'logs/ctc_segmented_download.log';url='https://data.celltrackingchallenge.net/training-datasets/'+DOMAIN+'.zip'
 runtime=ROOT/'outputs/02_eval_runtime/aria2_native';binary=runtime/'bin/aria2c';environment=os.environ.copy()
 if binary.is_file():environment['LD_LIBRARY_PATH']=str(runtime/'lib')
 args=[str(binary) if binary.is_file() else 'aria2c','--continue=true','--always-resume=true','--auto-file-renaming=false','--file-allocation=none','--max-connection-per-server=16','--split=16','--min-split-size=64M','--max-tries=0','--retry-wait=10','--connect-timeout=20','--timeout=120','--lowest-speed-limit=1024','--auto-save-interval=30','--summary-interval=30','--console-log-level=notice','--download-result=full','--dir='+str(workdir),'--out='+work.name,url]
 adopted=None
 for path in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   argv=path.read_bytes().split(b'\0')
   if ('--dir='+str(workdir)).encode() in argv and ('--out='+work.name).encode() in argv and url.encode() in argv:adopted=int(path.parent.name);break
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 with log.open('a') as f:
  proc=None if adopted else subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
  pid=adopted or proc.pid
  state.update(status='segmented_download_running',download_pid=pid,adopted=bool(adopted),log=str(log),started=time.time());write(state_path,state)
  while (Path(f'/proc/{pid}').exists() if adopted else proc.poll() is None):
   state.update(updated=time.time(),aria2_control_exists=work.with_name(work.name+'.aria2').exists());write(state_path,state);time.sleep(30)
 if proc is not None and proc.returncode:raise RuntimeError(f'aria2 exit {proc.returncode}; durable .aria2 state retained')
 if work.with_name(work.name+'.aria2').exists():raise RuntimeError('Incomplete aria2 control state retained; resume next pass')
 proof=validate(work,canonical)
 # Atomic hard-link publication does not copy a 343GB payload or alter its bytes.
 try:os.link(work,archive)
 except FileExistsError:validate(archive,canonical)
 retired=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   argv=p.read_bytes().split(b'\0')
   if argv and Path(os.fsdecode(argv[0])).name=='curl' and os.fsencode(str(partial)) in argv and url.encode() in argv:
    pid=int(p.parent.name);os.kill(pid,signal.SIGTERM);retired.append(pid)
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 proof.update(status='archive_metadata_verified_pending_source_extraction_sha256',canonical_manifest_sha256=hashlib.sha256((OUT/'validation/ctc_canonical_manifest.json').read_bytes()).hexdigest(),retired_redundant_sequential_download_pids=retired,archive=str(archive),completed=time.time())
 write(proof_path,proof);state.update(status='complete_archive_published',updated=time.time(),proof=str(proof_path),retired_redundant_sequential_download_pids=retired);write(state_path,state)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
 with (OUT/'locks/ctc_download_acceleration.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(0)
  while True:
   try:run()
   except Exception as error:
    write(OUT/'states/ctc_download_acceleration_error.json',dict(pid=os.getpid(),updated=time.time(),error=repr(error)));traceback.print_exc()
   if not args.watch:break
   time.sleep(60)
