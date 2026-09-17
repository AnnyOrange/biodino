import unittest

import torch

from dinov3.eval.bio_frozen_eval.external_fm_protocol import (
    MODELS,dense_recipe,even4_indices,resolve_dense_layers,reuse_decision,validate_best_head,
)
from dinov3.eval.bio_frozen_eval.external_fm_features import tokens_to_spatial,interpolate_position


class Tests(unittest.TestCase):
    def test_all_fourteen_registered(self):
        self.assertEqual(len(MODELS),14)

    def test_batch_alone_never_requires_retest(self):
        expected=dict(split='official',feature_batch_size=64,probe_batch_size=32,seed=0)
        actual=dict(split='official',feature_batch_size=8,probe_batch_size=64,seed=0)
        self.assertEqual(reuse_decision(expected,actual)['status'],'REUSE_BATCH_TOLERATED')

    def test_split_difference_requires_retest_even_with_batch_exception(self):
        report=reuse_decision(dict(split='official',feature_batch_size=64),dict(split='legacy',feature_batch_size=8))
        self.assertEqual(report['status'],'RETEST_NON_BATCH_DIFFERENCE')
        self.assertEqual(list(report['differences']),['split'])

    def test_missing_evidence_is_review_not_blind_retest(self):
        self.assertEqual(reuse_decision(dict(dtype='bf16',feature_batch_size=64),dict(feature_batch_size=8))['status'],'REVIEW_EVIDENCE')

    def test_even4_matches_dino_mapping(self):
        self.assertEqual(even4_indices(24),[4,11,17,23])
        self.assertEqual(even4_indices(12),[2,5,8,11])

    def test_even4_falls_back_only_when_unavailable(self):
        self.assertEqual(resolve_dense_layers('even4',12)['resolved_layers'],'even4')
        fallback=resolve_dense_layers('even4',None)
        self.assertEqual(fallback['resolved_layers'],'last')
        self.assertTrue(fallback['fallback_reason'])

    def test_primary_is_last_for_every_architecture(self):
        for name in MODELS:
            self.assertEqual(dense_recipe(name,'conic',depth=None if name in ('cytoself','cytoimagenet') else 12)['resolved_layers'],'last')

    def test_dataset_best_and_pannuke_explicit_protocol(self):
        self.assertEqual(dense_recipe('dinov2','conic',view='dataset-best',depth=12)['resolved_layers'],'even4')
        self.assertEqual(dense_recipe('cytoself','conic',view='dataset-best')['resolved_layers'],'last')
        with self.assertRaises(ValueError):dense_recipe('dinov2','pannuke',depth=12)
        with self.assertRaises(ValueError):dense_recipe('dinov2','bbbc038',depth=12)

    def test_registers_removed_and_channel_patch_grid_averaged(self):
        tokens=torch.cat((torch.full((1,5,1),100.),torch.arange(8.).reshape(1,8,1)),1)
        spatial=tokens_to_spatial(tokens,5,(2,2))
        torch.testing.assert_close(spatial.flatten(),torch.tensor([2.,3.,4.,5.]))
        with self.assertRaises(ValueError):tokens_to_spatial(torch.zeros(1,6,2),1,(2,2))

    def test_position_resize_preserves_prefix(self):
        pos=torch.randn(1,5,3);new=interpolate_position(pos,(3,3))
        self.assertEqual(tuple(new.shape),(1,10,3))
        torch.testing.assert_close(new[:,:1],pos[:,:1])

    def test_empty_spatial_tokens_rejected(self):
        with self.assertRaises(ValueError):tokens_to_spatial(torch.zeros(1,1,2),1,(2,2))

    def test_complete_dense_plan_without_loading_checkpoints(self):
        import contextlib
        import io
        import json
        from dinov3.eval.bio_frozen_eval.run_external_dense_rules import main
        output=io.StringIO()
        with contextlib.redirect_stdout(output):self.assertEqual(main(['--plan-only']),0)
        recipes=json.loads(output.getvalue())
        self.assertEqual(len(recipes),126)
        self.assertEqual({r['model'] for r in recipes},set(MODELS))
        self.assertEqual({r['dataset'] for r in recipes},set(('cellpose','conic','livecell','monuseg','pannuke','tissuenet','multimodal_cellseg')))

    def test_best_head_each_epoch_and_earliest_tie(self):
        history=[dict(epoch=e,mIoU=.9 if e in (3,7) else .1) for e in range(1,21)]
        meta=dict(probe_epochs=20,validation_history=history,best_epoch=3,best_val_miou=.9,test_evaluations=1)
        validate_best_head(meta)
        with self.assertRaises(ValueError):validate_best_head(dict(meta,best_epoch=20))
        with self.assertRaises(ValueError):validate_best_head(dict(meta,validation_history=history[-1:]))


if __name__=='__main__':unittest.main()
