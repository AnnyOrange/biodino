"""The H100 correction must match unified membership and reject legacy scores."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_lc25000_h100_grouped_20261029 import INPUT, PROTOCOL, install, membership, validate
from dinov3.eval.bio_frozen_eval.make_group_splits import group_split_indices


def test_membership_matches_unified_and_no_augmented_group_crosses():
    records = json.loads((INPUT / 'images.json').read_text())['records']
    dataset = SimpleNamespace(samples=[(Path(r['sample_id']), r['label']) for r in records])
    train, test, groups, _, _ = membership(dataset)
    reference_train, reference_test = group_split_indices('lc25000', dataset)
    np.testing.assert_array_equal(train, reference_train)
    np.testing.assert_array_equal(test, reference_test)
    assert (len(train), len(test)) == (19949, 5051)
    assert set(groups[train]).isdisjoint(groups[test])
    dataset.samples[0], dataset.samples[1] = dataset.samples[1], dataset.samples[0]
    with pytest.raises(AssertionError):
        membership(dataset)


def test_legacy_summary_or_wrong_protocol_cannot_be_adopted(tmp_path):
    (tmp_path / 'summary.csv').write_text('dataset,balanced_accuracy\nlc25000,0.9999\n')
    with pytest.raises(FileNotFoundError):
        validate(tmp_path)
    (tmp_path / 'component_result.json').write_text(json.dumps({'protocol': 'internal-80-20'}))
    with pytest.raises(AssertionError):
        validate(tmp_path)


def test_command_override_is_classification_only():
    m = SimpleNamespace(command=lambda *a: ['original'], validate=lambda *a: {'original': True},
                        ROOT=Path('/campaign'), PYTHON='python', BENCHMARK=Path('/data'), MODEL_TAG='L_seed2')
    install(m)
    command = m.command({'family': 'classification', 'dataset': 'lc25000'}, 3999, Path('/out'))
    assert any('run_lc25000_h100_grouped_20261029.py' in x for x in command)
    assert '/campaign/checkpoints/3999/checkpoint.pth' in command
    assert m.command({'family': 'retrieval_clustering', 'dataset': 'lc25000'}, 3999, Path('/out')) == ['original']
