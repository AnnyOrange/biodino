"""Lock the existing complete HPA v4 query/gallery and location cohorts."""
import concurrent.futures,hashlib,json,re,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dinov3.eval.bio_frozen_eval.retrieval_clustering import ManifestImageDataset
BENCH=Path(os.environ.get('UNIFIED_PAPER_BENCHMARK_ROOT','/mnt/huawei_deepcad/benchmark'));SOURCE=BENCH/'Retrieval_Clustering/HPA_Subcellular';PROTO=BENCH/'Retrieval_Clustering/protocols/v1'
MANIFEST=ROOT/'outputs/02_eval_inputs/formal_v4/hpa-subcellular_retrieval_manifest.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def datasets():
    return {role:ManifestImageDataset(SOURCE,PROTO/('hpa_single_location_clustering.csv'if role=='cluster'else'hpa_same_gene_query_gallery.csv'),role=None if role=='cluster'else role)for role in ['gallery','query','cluster']}
def main():
    sources=datasets();expected=dict(gallery=1786,query=1786,cluster=1458);records={}
    for role,ds in sources.items():
        assert len(ds)==expected[role],(role,len(ds))
        def row(r):
            path=SOURCE/r['image_path'];match=re.search(r'_((?:si)?\d+)_([A-Z]+\d+)_(\d+)_(?:cr[0-9a-f]+_)?(?:blue_)?red_green\.jpg$',path.name)
            assert match,'Unknown HPA original image naming: '+str(path)
            return dict(sample_id=str(path),group_id='HPA-well:'+match[1]+'_'+match[2],label=int(r['label']),image_sha256=sha(path),gene=r['gene'],robust_ge10=r.get('robust_ge10'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:records[role]=list(pool.map(row,ds.rows))
        assert len({r['sample_id']for r in records[role]})==len(ds)
    assert not(set(r['sample_id']for r in records['gallery'])&set(r['sample_id']for r in records['query']))
    shared_wells=set(r['group_id']for r in records['gallery'])&set(r['group_id']for r in records['query'])
    data=dict(dataset='hpa-subcellular',protocol='existing-v4-HPA-same-gene-query-gallery-and-single-location-all41',records=records,source_hashes={str(PROTO/n):sha(PROTO/n)for n in ['hpa_same_gene_query_gallery.csv','hpa_single_location_clustering.csv']},
        grouping='Inferred HPA plate/well tokens from source image names; all matching fields grouped together; biological independence not verified',primary_grouping_verified=False,
        group_limitation='Existing within-cohort same-gene image retrieval, conditional on fixed gallery. Gallery/query share '+str(len(shared_wells))+' wells; this is not heldout-well or unseen-gene generalization. Bootstrap all query fields within each well together. Clustering inference conditional on fitted partitions.',
        sizes=expected,shared_gallery_query_wells=len(shared_wells),selection='No new members, caps, label changes or random partition; exact existing v4 CSVs')
    if MANIFEST.exists():assert json.loads(MANIFEST.read_text())==data
    else:MANIFEST.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(manifest=str(MANIFEST),sha256=sha(MANIFEST),sizes=expected,groups={s:len({r['group_id']for r in rows})for s,rows in records.items()},shared_gallery_query_wells=len(shared_wells))))
if __name__=='__main__':main()
