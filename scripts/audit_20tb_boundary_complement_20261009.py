#!/usr/bin/env python3
"""Read-only source inventory comparison; writes aggregate diagnostic artifacts."""
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923')
OUT = ROOT / 'plot/fig2/route2_20tb_v4_20261008/dataset_diagnosis_20261008/boundary_complement_20261009'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_file = DATA / 'final_selection/selected_100tb_sources.parquet'
    boundary_file = DATA / 'phase_repartition_no_old5_20260923/phase5_sources.parquet'
    old_file = DATA / 'original_5tb_oid_index_20260923/all_5tb_oids.parquet'
    df = pq.read_table(source_file, columns=['source_key', 'source_dataset', 'imaging_family',
                       'qualified_patches', 'qualified_bytes', 'nearest_target_center']).to_pandas()
    boundary = set(pq.read_table(boundary_file).column(0).to_pylist())
    old = set(pq.read_table(old_file, columns=['oid']).column(0).to_pylist())
    assert df.source_key.is_unique
    # These are source-run:item/OID keys. The separate `source_id` column is
    # another identifier and must not be joined to the original 5TB OID index.
    pieces = df.source_key.str.split(':', expand=True)
    assert set(pieces[0]) == {'1', '5'}
    df['oid'] = pieces[1].astype('int64')
    df['phase'] = df.source_key.isin(boundary).map({True: 'boundary5', False: 'new15_micro'})
    df['overlap_old5_oid'] = df.oid.isin(old)
    old_audit = json.loads((DATA / 'original_5tb_oid_index_20260923/aggregate_report.json').read_text())
    expected_sources = sum(old_audit[k]['sources'] for k in ['phase15_overlap', 'phase5_overlap'])
    expected_patches = sum(old_audit[k]['patches'] for k in ['phase15_overlap', 'phase5_overlap'])
    q = df[df.phase == 'boundary5']
    assert not set(q.oid).intersection(df[df.phase == 'new15_micro'].oid)
    overlap = q[q.overlap_old5_oid]
    assert len(overlap) == expected_sources == 77011
    assert int(overlap.qualified_patches.sum()) == expected_patches == 1724753
    assert not df[(df.phase == 'new15_micro') & df.overlap_old5_oid].shape[0]
    aggregate = dict(sources=('source_key', 'size'), patches=('qualified_patches', 'sum'),
                     byte_count=('qualified_bytes', 'sum'))
    groups = df.groupby(['phase', 'overlap_old5_oid']).agg(**aggregate).reset_index()
    groups.to_csv(OUT / 'source_overlap.csv', index=False)
    for field in ['source_dataset', 'imaging_family']:
        result = q.groupby([field, 'overlap_old5_oid']).agg(**aggregate).reset_index()
        result.to_csv(OUT / f'boundary_by_{field}.csv', index=False)
    centers15 = set(df[df.phase == 'new15_micro'].nearest_target_center)
    centers5 = set(q.nearest_target_center)
    summary = {
        'sources': {k: str(p) for k, p in [('selected', source_file), ('boundary', boundary_file), ('old5_oid', old_file)]},
        'identity': 'source_key suffix is OID, audited against historical overlap counts; source_id is not OID',
        'boundary_sources': len(q), 'boundary_patches': int(q.qualified_patches.sum()),
        'old5_overlapping_sources': len(overlap),
        'old5_overlapping_patches': int(overlap.qualified_patches.sum()),
        'old5_overlapping_patch_fraction': float(overlap.qualified_patches.sum()/q.qualified_patches.sum()),
        'old5_overlapping_byte_fraction': float(overlap.qualified_bytes.sum()/q.qualified_bytes.sum()),
        'new_source_count_vs_main_corpus': len(q)-len(overlap),
        'new_source_patches_vs_main_corpus': int(q.qualified_patches.sum()-overlap.qualified_patches.sum()),
        'new_source_definition': 'not in new15 source-key set or historical old5 OID set; not a patient/slide/pixel dedup guarantee',
        'boundary_centers_not_in_new15_micro': len(centers5-centers15),
        'center_note': 'These centers may already be represented by old5; not established as gaps in main training.',
        'scope': 'Historical selected/packing manifests, not actual online frequency or current-tar pixel audit',
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
