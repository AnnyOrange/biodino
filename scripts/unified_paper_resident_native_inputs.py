"""Require byte-verified complete sources for relocated native image loaders."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
def verify_resident_inputs(dataset,manifest_path,manifest):
    root=os.environ.get('UNIFIED_PAPER_BENCHMARK_ROOT','/mnt/huawei_deepcad/benchmark')
    if root=='/mnt/huawei_deepcad/benchmark':return
    proof= json.loads((OUT/'validation'/f'ai_native_{dataset}_resident.json').read_text())
    assert proof['manifest_sha256']==hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert proof['n_records']==len(manifest['records'])
    from dinov3.eval.bio_detection.native import resolve_source_image
    paths={str(resolve_source_image(r['path'])) for r in manifest['records']}
    assert paths==set(proof['source_fingerprints'])
    for path,expected in proof['source_fingerprints'].items():
        st=Path(path).stat();sig=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,inode=st.st_ino)
        if sig==proof.get('source_stats',{}).get(path):continue
        with Path(path).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,path
