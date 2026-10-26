"""Verify complete shared global-task members, labels, groups and source bytes."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/02_eval_runs/unified_paper';sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_shared_frozen_stage1_20260917 import identities as base_identities
from dinov3.eval.bio_frozen_eval.registry import build_dataset,NATIVE_TEST_SPLIT_DATASETS
from dinov3.eval.bio_frozen_eval.group_keys import group_keys
from dinov3.eval.bio_frozen_eval.make_group_splits import group_split_indices,split_path

def align_source_order(ds, expected, benchmark):
    """Match canonical tuple-backed sample order without changing labels or members."""
    records,_=base_identities(ds,'train')
    def key(record):
        path,label=record;prefix=str(benchmark)
        if path.startswith(prefix+'/'):path='/mnt/huawei_deepcad/benchmark'+path[len(prefix):]
        return json.dumps([path,label],sort_keys=True)
    observed=[key(r) for r in records];wanted=[json.dumps(r,sort_keys=True) for r in expected]
    if observed==wanted:return
    assert hasattr(ds,'samples') and all(isinstance(s,tuple) for s in ds.samples),'Order adaptation requires tuple-backed samples'
    assert len(set(observed))==len(observed) and len(set(wanted))==len(wanted)
    assert set(observed)==set(wanted),'Canonical source membership or labels differ'
    mapping=dict(zip(observed,ds.samples));ds.samples=[mapping[k] for k in wanted]

def build(name,benchmark,expected=None):
    benchmark=Path(benchmark);files=set();cache_path=OUT/'validation/global_source_hash_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    def canonical(path):
        path=str(path)
        if path.startswith(str(benchmark)+'/'):return '/mnt/huawei_deepcad/benchmark'+path[len(str(benchmark)):]
        if path.startswith(str(ROOT)+'/'):return '/mnt/huawei_deepcad/dinov3'+path[len(str(ROOT)):]
        return path
    def describe(ds,part):
        records,_=base_identities(ds,part)
        if hasattr(ds,'coco_image_ids'):records=[(r[0]+'#image_id='+i,r[1]) for r,i in zip(records,ds.coco_image_ids)]
        for attr in ['path','meta_path','csv_path','images_path']:
            if hasattr(ds,attr):files.add(Path(getattr(ds,attr)))
        files.update(Path(p) for p in getattr(ds,'files',[]))
        for sample in getattr(ds,'samples',[]):
            raw=sample.image_path if hasattr(sample,'image_path') else sample[0]
            if isinstance(raw,(str,Path)) and Path(raw).is_file():files.add(Path(raw))
        if name=='bbbc013':files.add(ds.root/'BBBC013_v1_platemap_all.txt')
        if name in ['conic-cell-count','livecell-cell-count']:files.add(ds.root/('conic_cell_count.csv' if name=='conic-cell-count' else 'livecell_cell_count.csv'))
        records=[(canonical(path),label) for path,label in records]
        if name=='livecell-cell-count':
            # The canonical data symlink is absent on exFAT mirrors. Byte-level
            # fingerprints below still compare the actual resolved image files.
            records=[(path.replace('/mnt/huawei_deepcad/benchmark/segmentation/LIVECell/LIVECell_dataset_2021/','/mnt/huawei_deepcad/benchmark/Regression/LIVECell_Cell_Count/data/'),label) for path,label in records]
        return records
    ds,task=build_dataset(name,'train',None,None,benchmark_root=benchmark)
    if expected and expected.get('source_order',{}).get('train'):align_source_order(ds,expected['source_order']['train'],benchmark)
    records=describe(ds,'train');groups={};source_order={}
    if hasattr(ds,'samples') and all(isinstance(s,tuple) for s in ds.samples):source_order['train']=records
    if name in NATIVE_TEST_SPLIT_DATASETS:
        test,_=build_dataset(name,'test',None,None,benchmark_root=benchmark)
        if expected and expected.get('source_order',{}).get('test'):align_source_order(test,expected['source_order']['test'],benchmark)
        test_records=describe(test,'test');members=dict(train=records,test=test_records)
        if hasattr(test,'samples') and all(isinstance(s,tuple) for s in test.samples):source_order['test']=test_records
    elif name=='bbbc013':
        assert len(records)==96;members=dict(oof_all=records);groups['oof_all']=group_keys(name,ds,benchmark)
    else:
        tr,te=group_split_indices(name,ds,benchmark);all_groups=group_keys(name,ds,benchmark);files.add(split_path(name));members=dict(train=[records[i] for i in tr],test=[records[i] for i in te]);groups=dict(train=[all_groups[i] for i in tr],test=[all_groups[i] for i in te]);assert not set(groups['train']) & set(groups['test'])
    fingerprints={};canonical_files={};source_stats={}
    for path in sorted(files):
        stat=path.stat();key=str(path);old=cache.get(key,{})
        signature=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,inode=stat.st_ino)
        if all(old.get(k)==v for k,v in signature.items()):digest=old['sha256']
        else:
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            digest=h.hexdigest();cache[key]=dict(**signature,sha256=digest)
        fingerprints[str(path)]=digest;canonical_files[canonical(path.resolve())]=digest
        source_stats[str(path)]=signature
    assert fingerprints
    tmp=cache_path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(cache)+'\n');tmp.replace(cache_path)
    membership=dict(dataset=name,task=task,members=members,groups=groups)
    return dict(dataset=name,task=task,counts={k:len(v) for k,v in members.items()},membership_sha256=hashlib.sha256(json.dumps(membership,sort_keys=True).encode()).hexdigest(),canonical_source_files=canonical_files,source_fingerprints=fingerprints,source_stats=source_stats,source_order=source_order,group_split_preserved=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--benchmark-root',default='/mnt/huawei_deepcad/benchmark');p.add_argument('--output',type=Path,required=True);p.add_argument('--expected',type=Path);a=p.parse_args();expected=json.loads(a.expected.read_text()) if a.expected else None;result=build(a.dataset,a.benchmark_root,expected)
    if a.expected:
        expected=json.loads(a.expected.read_text())
        for key in ['dataset','task','counts','membership_sha256','canonical_source_files']:
            if result[key]!=expected[key]:raise ValueError('Resident inputs differ from common protocol: '+a.dataset+' / '+key)
        result['canonical_proof_sha256']=hashlib.sha256(a.expected.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(a.output)
    print('GLOBAL INPUTS VERIFIED',a.dataset,result['counts'],len(result['source_fingerprints']),flush=True)
