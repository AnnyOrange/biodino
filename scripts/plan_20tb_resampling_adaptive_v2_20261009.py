#!/usr/bin/env python3
"""Offline sampling-policy design from source metadata; never launches training.

Probabilities describe historical manifest pools, not measured DataLoader output.
Actual r0/r9 tar membership and decoder success must be joined before deployment.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923')
OUT = ROOT / 'outputs/00_reports/20tb_resampling_adaptive_v2_plan_20261009'


def bounded_project_weights(raw_target, baseline):
    """Normalize a target while bounding each project to 0.5–2x its old share."""
    lower, upper = .5 * baseline, 2 * baseline
    lo, hi = 0., 1.
    while np.clip(hi * raw_target, lower, upper).sum() < 1:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.clip(mid * raw_target, lower, upper).sum() < 1:
            lo = mid
        else:
            hi = mid
    q = np.clip((lo + hi) / 2 * raw_target, lower, upper)
    assert abs(q.sum()-1) < 1e-10
    assert np.all(q >= lower-1e-12) and np.all(q <= upper+1e-12)
    return q


def policy_for_pool(df, name):
    df = df.copy()
    n = df.qualified_patches.astype(float)
    assert (n > 0).all()
    total = n.sum()
    # Preserve imaging-family exposure. Use an effective source mass that gives
    # diminishing credit to more tiles from the same original image.
    df['source_mass'] = np.sqrt(np.minimum(n, 64))
    groups = ['imaging_family', 'source_dataset']
    g = df.groupby(groups).agg(patches=('qualified_patches','sum'),
                              source_mass=('source_mass','sum'), sources=('source_key','size'))
    g['baseline_probability'] = g.patches / total
    g['target_probability'] = 0.
    for modality in g.index.get_level_values(0).unique():
        part = g.loc[modality]
        quota = part.patches.sum() / total
        baseline = part.patches.to_numpy(float) / part.patches.sum()
        target = bounded_project_weights(np.sqrt(part.source_mass.to_numpy(float)), baseline)
        g.loc[modality, 'target_probability'] = quota * target
        assert abs(g.loc[modality, 'target_probability'].sum()-quota) < 1e-10
    df = df.join(g[['patches', 'source_mass', 'target_probability']], on=groups, rsuffix='_project')
    # Half of each project's old tile distribution remains, avoiding abrupt
    # deletion of large images; half uses the capped source-level mass.
    df['baseline_probability'] = df.qualified_patches / total
    old = df.baseline_probability.to_numpy()
    top_n = max(1, len(df)//100)
    def top_share(v):
        return np.partition(v, -top_n)[-top_n:].sum()
    local = (.5 * df.qualified_patches / df.patches +
             .5 * df.source_mass / df.source_mass_project).to_numpy()
    source_only = (df.patches.to_numpy()/total) * local
    full_proposal = df.target_probability.to_numpy() * local
    assert np.isfinite(old).all() and np.isfinite(full_proposal).all()
    old_top, old_sq, old_max = top_share(old), np.square(old).sum(), old.max()
    def acceptable(v):
        return (top_share(v) <= old_top+1e-12 and
                np.square(v).sum() <= old_sq+1e-12 and v.max() <= old_max+1e-12)
    assert acceptable(source_only)
    # Project equalization can move mass toward projects with fewer independent
    # sources. Limit it with metadata-only concentration guards instead of
    # assuming that a flatter project histogram is always more diverse.
    strength = 1.
    if not acceptable(full_proposal):
        low, high = 0., 1.
        for _ in range(32):
            mid = (low+high)/2
            if acceptable((1-mid)*source_only + mid*full_proposal):
                low = mid
            else:
                high = mid
        strength = low
    new = (1-strength)*source_only + strength*full_proposal
    assert acceptable(new) and abs(new.sum()-1) < 1e-10
    df['target_probability_source'] = new
    g['target_probability'] = ((1-strength)*g.baseline_probability + strength*g.target_probability)
    summary = dict(pool=name, source_records=len(df), patches=int(total),
                   baseline_source_ess=float(1/np.square(old).sum()),
                   target_source_ess=float(1/np.square(new).sum()),
                   baseline_top1pct_source_probability=float(np.sort(old)[-top_n:].sum()),
                   target_top1pct_source_probability=float(np.sort(new)[-top_n:].sum()),
                   project_rebalance_strength=strength,
                   baseline_largest_source_probability=float(old.max()),
                   target_largest_source_probability=float(new.max()),
                   modality_probabilities_preserved=True)
    g = g.reset_index()
    g.insert(0, 'pool', name)
    g['exposure_multiplier'] = g.target_probability / g.baseline_probability
    # Export a concrete source policy without writing millions of CSV lines.
    cols = ['source_key', 'source_dataset', 'imaging_family', 'qualified_patches',
            'baseline_probability', 'target_probability_source']
    df[cols].to_parquet(OUT / f'{name}_source_probabilities.parquet', index=False)
    return g, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = DATA / 'final_selection/selected_100tb_sources.parquet'
    boundary = DATA / 'phase_repartition_no_old5_20260923/phase5_sources.parquet'
    old = DATA / 'original_5tb_oid_index_20260923/all_5tb_oids.parquet'
    df = pq.read_table(source, columns=['source_key','source_dataset','imaging_family','qualified_patches']).to_pandas()
    keys = set(pq.read_table(boundary).column(0).to_pylist())
    oids = set(pq.read_table(old, columns=['oid']).column(0).to_pylist())
    df['is_boundary'] = df.source_key.isin(keys)
    df['overlaps_old5'] = df.source_key.str.split(':').str[-1].astype('int64').isin(oids)
    assert not (df.overlaps_old5 & ~df.is_boundary).any()
    pools = {'new15_micro_manifest': df[~df.is_boundary],
             'boundary_replay_manifest': df[df.is_boundary & df.overlaps_old5],
             'boundary_novel_manifest': df[df.is_boundary & ~df.overlaps_old5]}
    groups, summaries = [], []
    for name, frame in pools.items():
        g, s = policy_for_pool(frame, name)
        groups.append(g); summaries.append(s)
    projects = pd.concat(groups, ignore_index=True)
    projects.to_csv(OUT / 'project_modality_weights.csv', index=False)
    projects.groupby(['pool','source_dataset'])[['baseline_probability','target_probability']].sum().to_csv(OUT / 'project_weights.csv')
    pd.DataFrame(summaries).to_csv(OUT / 'sampling_concentration.csv', index=False)
    plan = {
        'status': 'DESIGN_ONLY_NOT_DEPLOYED',
        'selection_note': 'Exploratory: past test curves informed design; select changes on independent validation.',
        'starting_checkpoint': 38063,
        'primary_anchor_checkpoint': 38063,
        'optional_anchor_ablation': 26351,
        'alternative_starting_checkpoint': 47823,
        'steps_to_run': 5856, 'last_checkpoint': 43919, 'max_updates': 43920,
        'evaluation_checkpoints': [38063,40015,41967,43919],
        'starting_checkpoint_reason': 'Keep dense-task gains before the later Cellpose/BBBC048 decline; not declared globally best.',
        'pool_weights': {'legacy_1tb': .09, 'legacy_4tb': .21, 'new15_r0': .4581,
                        'new15_r9_mixed': .1419, 'boundary_replay': .04, 'boundary_novel': .06},
        'pool_weights_note': 'Candidate, not validated optimum. Boundary replay/novel require sample-level OID routing, not independent full-tar globs.',
        'sampling': {'modality_quota': 'preserve within each deployed pool',
                     'source_mass': 'sqrt(min(valid_unique_tile_count,64))',
                     'project_target': 'sqrt(sum(source_mass)), normalized with probability bounds 0.5x..2x baseline within each modality',
                     'concentration_guard': 'Blend bounded project target with original project share; maximize blend subject to source top-1% share, largest-source share and sum(p^2) not increasing. Determined from metadata only.',
                     'source_distribution': '0.5*original_tile_share + 0.5*normalized_source_mass within each project/modality',
                     'tile_selection': 'one tile per source draw, rotate without replacement within source; exact sample-key duplicates excluded',
                     'same_source_across_pools': 'global OID accounting, preserve distinct crop identities',
                     'buffer': 3000, 'domain_block_size': 1,
                     'bounded_amplification_note': '0.5x..2x applies to project probability within a modality, not to each individual source.'},
        'recovery': {'enabled': True, 'mode': 'fixed', 'global_tokens': 'cls', 'loss_weight': 1.,
                     'global_weight': 1., 'local_weight': 0., 'anchor_momentum': 0.,
                     'ridge': .1, 'warmup': 32},
        'recovery_note': 'warmup=32 forward calls, approximately four optimizer updates with accumulation=8; fixed mode has no adaptive dual gate.',
        'precision': 'FSDP bf16 computation, fp32 masters and optimizer states',
        'layout': {'ranks': 8, 'batch_per_rank': 16, 'accumulation': 8,
                   'global_microbatch': 128, 'effective_batch': 1024},
        'layout_alternative': {'ranks': 4, 'batch_per_rank': 32, 'accumulation': 8,
                               'note': 'Preserves global microbatch; memory/throughput must be verified. 4x16x16 is not identical despite same effective batch.'},
        'training_schedule': 'Continue existing 61x4098 LR/WD/teacher schedules and original decoder normalization; no optimizer restart.',
        'factorial_arms': {'A': {'sampling': 'original', 'recovery': False},
                          'B': {'sampling': 'proposed', 'recovery': False},
                          'C': {'sampling': 'original', 'recovery': True},
                          'D': {'sampling': 'proposed', 'recovery': True}},
        'extra_arms': 'Only after primary checks: w=3 vs w=1 at same sampling; ck26351 teacher vs ck38063 teacher from same full ck38063.',
        'implementation_required': ['Join source policy to actual formal r0/r9/boundary tar member inventories and valid decoded samples.',
                                    'Recompute source masses and project/modal quotas on those exact pools; new15_micro_manifest is not the exact r0 pool.',
                                    'Retain metadata through loader and implement source-aware selection; current mixture loader only weights entire streams.',
                                    'Use sequential shard reads with indexed metadata and bounded source buffers; avoid per-sample remote random reads.',
                                    'Audit accepted-sample frequencies, source/crop overlap, actual rejection/I/O rate, and per-rank seeds before training.'],
        'inputs': [str(source),str(boundary),str(old)],
        'offline_concentration': summaries,
    }
    assert abs(sum(plan['pool_weights'].values())-1) < 1e-12
    (OUT / 'experiment_plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(summaries, indent=2))
    print(projects.groupby(['pool','source_dataset'])[['baseline_probability','target_probability']].sum().sort_values('baseline_probability', ascending=False).head(15).to_string())


if __name__ == '__main__':
    main()
