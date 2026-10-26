"""Author-documented well hierarchy for inference; immutable training manifests."""
import hashlib,json,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/02_eval_runs/unified_paper'
PROOF=OUT/'validation/livecell_well_groups.json'
VERSION='livecell-author-well-inference-v1'
LIMITATION='Nine held-out culture wells from eight fixed cell types; inference is conditional on this acquisition/plate context and these five fitted heads, not unseen plates, donors or cell types. Small cluster count limits precision.'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def filename(row):return Path(row['sample_id'].split('#',1)[0]).name

def parse(name):
    match=re.fullmatch(r'(.+)_Phase_([A-H]\d+)_(\d+)_(\d+d\d+h\d+m)_(\d+)\.tif',name)
    if match is None:raise ValueError('Unrecognized official LIVECell filename: '+name)
    cell,well,location,timestamp,crop=match.groups()
    return dict(group_id=cell+'::'+well,cell_type=cell,well=well,location=location,timestamp=timestamp,crop=crop)

def prepare():
    source=OUT/'provenance/livecell/source.json';s=json.loads(source.read_text())
    assert digest(s['local_path'])==s['sha256']
    text=Path(s['local_path']).read_text()
    assert '<Cell Type>_Phase_<Well>_<Location>_<Timestamp>_<Crop>.tif'in text
    assert '96-well plate'in text and 'crop of the original larger image'in text
    manifest=ROOT/'outputs/02_eval_inputs/formal_v4/livecell_center_probe_manifest.json'
    if PROOF.exists():
        existing=json.loads(PROOF.read_text())
        if existing.get('version')==VERSION and existing.get('source')==s and existing.get('canonical_manifest_sha256')==digest(manifest):return existing
    raw=json.loads(manifest.read_text());splits={};rows=[]
    for split,records in raw['records'].items():
        for row in records:rows.append(dict(split=split,sample_id=row['sample_id'],file_name=filename(row),**parse(filename(row))))
        names={filename(r)for r in records};groups={parse(n)['group_id']for n in names}
        splits[split]=dict(n_records=len(records),n_images=len(names),groups=sorted(groups))
    for split in ['train','val']:assert not set(splits[split]['groups'])&set(splits['test']['groups'])
    assert len(splits['test']['groups'])==9
    proof=dict(version=VERSION,created=time.time(),source=s,canonical_manifest=str(manifest),canonical_manifest_sha256=digest(manifest),splits=splits,records=rows,
        primary_grouping_verified=True,grouping='Author-documented cell-type/culture-well clusters; all locations, time points, crops and repeated COCO annotation records retained together',limitation=LIMITATION,
        training_membership_changed=False,point_estimator_changed=False,bootstrap='Unstratified resampling of nine whole wells; 2000 replicates, equal well draw probability, pooled counts retain unequal well sizes; same draw for all five individual fits',paired_test='Two-sided whole-well model swaps; all five fits swap together; 10000 Monte Carlo permutations; Holm family as in unified config')
    tmp=PROOF.with_suffix('.tmp');tmp.write_text(json.dumps(proof,indent=2)+'\n');tmp.replace(PROOF)
    return proof

def regroup(rows):
    proof=json.loads(PROOF.read_text());assert proof['version']==VERSION
    assert digest(proof['canonical_manifest'])==proof['canonical_manifest_sha256']
    allowed={r['file_name']:r['group_id']for r in proof['records']if r['split']=='test'}
    result=[]
    for row in rows:
        name=filename(row)
        if name not in allowed:raise ValueError('LIVECell test image absent from canonical author split: '+name)
        result.append(dict(row,original_image_group_id=row['group_id'],group_id=allowed[name],stratum='all'))
    metadata=dict(grouping_version=VERSION,grouping_proof=str(PROOF),grouping_proof_sha256=digest(PROOF),primary_grouping_verified=True,grouping=proof['grouping'],group_limitation=LIMITATION)
    return result,metadata

def archive_previous(path):
    if not path.exists():return
    target=OUT/'statistics/history/grouped_inference_revisions';target.mkdir(parents=True,exist_ok=True)
    saved=target/(digest(path)+'.'+path.name)
    if not saved.exists():saved.write_bytes(path.read_bytes())

if __name__=='__main__':
    proof=prepare();print(json.dumps(dict(version=VERSION,splits=proof['splits'],proof=str(PROOF))))
