"""Resume official CTC downloads and reproduce the exact canonical native cache."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,shutil,subprocess,time,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
CANONICAL_ROOT='/mnt/huawei_deepcad/dinov3'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def signature(path):
    st=path.stat();return dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,inode=st.st_ino)

def relocate(value):
    if isinstance(value,str) and value.startswith(CANONICAL_ROOT+'/'):return str(ROOT)+value[len(CANONICAL_ROOT):]
    if isinstance(value,list):return [relocate(x) for x in value]
    if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
    return value

def prepare_domain(domain,canonical):
    state=OUT/'states'/f'ai_ctc_data_{domain}.json'
    proof_path=OUT/'validation'/f'ai_ctc_data_{domain}.json'
    files=[r for r in canonical['files'] if r['source_member'].startswith(domain+'/')]
    assert files,domain
    old=json.loads(proof_path.read_text()) if proof_path.exists() else {}
    if old.get('canonical_files_sha256')==hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest():
        if all(Path(p).is_file() and signature(Path(p))==s for p,s in old.get('source_stats',{}).items()) and len(old.get('source_stats',{}))==len(files):return old
    archives=OUT/'prepared_inputs/ctc_official_archives';archives.mkdir(parents=True,exist_ok=True)
    archive=archives/f'{domain}.zip';partial=archive.with_suffix('.zip.part')
    url='https://data.celltrackingchallenge.net/training-datasets/'+domain+'.zip'
    if not archive.exists():
        write(state,dict(pid=os.getpid(),updated=time.time(),status='downloading_official_archive',url=url,partial=str(partial),files_required=len(files)))
        log=OUT/'logs'/f'ai_ctc_download_{domain}.log'
        with log.open('a') as stream:
            subprocess.run(['curl','--fail','--location','--continue-at','-','--retry','5','--retry-delay','10','--connect-timeout','20','--speed-limit','1024','--speed-time','120','--output',str(partial),url],stdout=stream,stderr=subprocess.STDOUT,check=True)
        with zipfile.ZipFile(partial) as z:
            for row in files:
                info=z.getinfo(row['source_member']);assert info.file_size==row['bytes'] and f'{info.CRC:08x}'==row['crc32'],row['source_member']
        partial.replace(archive)
    write(state,dict(pid=os.getpid(),updated=time.time(),status='extracting_and_hash_verifying',archive=str(archive),files_required=len(files)))
    stats={};fingerprints={}
    def extract_verified(row):
        target=Path(relocate(row['path']));target.parent.mkdir(parents=True,exist_ok=True)
        expected=row['sha256'];valid=False
        # Separate readers allow the large final domain to use multiple disk requests.
        # Atomic rename publishes only a fully hash-verified image, never a prefix.
        with target.with_name(target.name+'.extraction.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if target.is_file() and target.stat().st_size==row['bytes']:
                with target.open('rb') as stream:valid=hashlib.file_digest(stream,'sha256').hexdigest()==expected
            if not valid:
                tmp=target.with_name(target.name+'.extracting');digest=hashlib.sha256()
                with zipfile.ZipFile(archive) as z,z.open(row['source_member']) as source,tmp.open('wb') as dest:
                    while chunk:=source.read(8*1024*1024):digest.update(chunk);dest.write(chunk)
                assert tmp.stat().st_size==row['bytes'] and digest.hexdigest()==expected,row['source_member']
                tmp.replace(target)
        return str(target),signature(target),expected
    workers=4 if sum(r['bytes']for r in files)>100_000_000_000 else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(extract_verified,row)for row in files]
        for number,future in enumerate(concurrent.futures.as_completed(futures),1):
            target,stat,digest=future.result();stats[target]=stat;fingerprints[target]=digest
            write(state,dict(pid=os.getpid(),updated=time.time(),status='extracting_and_hash_verifying',files_verified=number,files_required=len(files),archive=str(archive),extraction_workers=workers,bytes_verified=sum(v['size']for v in stats.values())))
    proof=dict(domain=domain,canonical_files_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),source_stats=stats,source_fingerprints=fingerprints,source_url=url)
    write(proof_path,proof);write(state,dict(pid=os.getpid(),updated=time.time(),status='complete_verified',files_verified=len(files),proof=str(proof_path)));return proof

def run():
    path=OUT/'validation/ctc_canonical_manifest.json';canonical=json.loads(path.read_text())
    assert len(canonical['domains'])==20 and len(canonical['folds'])==5
    required=sum(r['bytes'] for r in canonical['files'])
    if shutil.disk_usage(OUT).free<required*1.2 and not (OUT/'validation/ai_ctc_inputs_complete.json').exists():raise RuntimeError('Insufficient resident free disk for complete native CTC cache')
    domains=sorted(canonical['domains'],key=lambda d:canonical['domains'][d]['archive']['bytes']);errors=[];proofs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(prepare_domain,d,canonical):d for d in domains}
        for future in concurrent.futures.as_completed(futures):
            domain=futures[future]
            try:proofs.append(future.result());print('CTC DOMAIN COMPLETE',domain,flush=True)
            except Exception as error:errors.append(dict(domain=domain,error=repr(error)));print('CTC DOMAIN FAILED',domain,repr(error),flush=True)
            write(OUT/'states/ai_ctc_data_preparation.json',dict(pid=os.getpid(),updated=time.time(),complete_domains=len(proofs),required_domains=20,errors=errors))
    if errors:raise RuntimeError(f'{len(errors)} domains failed; complete domains retained')
    manifest=relocate(canonical);destination=ROOT/'outputs/02_eval_inputs/formal_v3/ctc_native_full/data_manifest.json'
    split=ROOT/'outputs/02_eval_inputs/formal_v3/ctc/split_manifest.jsonl'
    assert hashlib.sha256(split.read_bytes()).hexdigest()==canonical['source_split_manifest_sha256']
    write(destination,manifest)
    write(OUT/'validation/ai_ctc_inputs_complete.json',dict(status='PASS',domains=20,files=len(canonical['files']),canonical_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),resident_manifest_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),source_split_manifest_sha256=canonical['source_split_manifest_sha256'],domain_proofs=[str(OUT/'validation'/f'ai_ctc_data_{d}.json') for d in domains]))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    with (OUT/'locks/ai_ctc_data_preparation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(0)
        while True:
            try:run()
            except Exception:
                traceback.print_exc()
                if not args.watch:raise
            if not args.watch:break
            time.sleep(180)
