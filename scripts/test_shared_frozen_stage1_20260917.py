"""Focused tests for staged commands, provenance, counts and retained inputs."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('stage1', ROOT/'scripts/run_shared_frozen_stage1_20260917.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def task():
    return dict(key='main_ck1__classification__breastmnist',asset=dict(path='/weights/teacher.pth',config='/weights/config.yaml'),
                dataset=dict(dataset='breastmnist',task='classification',counts=dict(n_train=546,n_test=156),
                             image_size=224,resize_size=256,split='official-test'))


class StageTests(unittest.TestCase):
    def test_commands_are_explicit_and_no_cache(self):
        cell = task()
        for kind in ('classification','regression','retrieval'):
            cell['dataset']['task'] = kind
            command = stage.command(cell,Path('/result'),'/benchmark')
            for flag,value in (('--batch-size','64'),('--autocast-dtype','bf16'),('--n-last-blocks','1'),('--seed','0'),('--num-workers','2')):
                self.assertEqual(command[command.index(flag)+1],value)
            self.assertIn('--no-save-features',command)
            self.assertIn('--overwrite-results',command)
            self.assertNotIn('--no-avgpool',command)
            self.assertNotIn('--max-samples',command)

    def result(self, directory, **changes):
        cell = task()
        row = dict(dataset='breastmnist',checkpoint=cell['asset']['path'],train_config=cell['asset']['config'],
                   n_train=546,n_test=156,batch_size=64,seed=0,image_size=224,resize_size=256,
                   split='official-test',balanced_accuracy=.8)
        row.update(changes)
        (directory/'last_result.json').write_text(json.dumps(row))
        return cell

    def test_valid_component_does_not_enable_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            report = stage.validate_cell(self.result(path),path,dict(git_commit='abc'))
            self.assertEqual(report['status'],'VALID_COMPLETE')
            self.assertFalse(report['full_v3_aggregate_allowed'])

    def test_wrong_counts_protocol_and_nonfinite_rejected(self):
        for changes in (dict(n_test=155),dict(batch_size=32),dict(split='internal-80-20'),dict(balanced_accuracy=float('nan'))):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                with self.assertRaises(ValueError):
                    stage.validate_cell(self.result(path,**changes),path,dict(git_commit='abc'))

    def test_config_change_rejected_even_with_checkpoint_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); (path/'_state/inputs').mkdir(parents=True)
            weights = path/'teacher.pth'; weights.write_bytes(b'retained')
            config = path/'config.yaml'; config.write_text('old')
            asset = dict(path=str(weights),config=str(config))
            stat = weights.stat()
            record = dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,config_sha256=stage.sha256(config))
            stage.save(path/'_state/inputs'/f'{stage.fingerprint(str(weights))}.json',record)
            config.write_text('new')
            with self.assertRaisesRegex(RuntimeError,'Config changed'):
                stage.checkpoint_record(path,asset)

    def test_retrieval_no_feature_flag(self):
        from dinov3.eval.bio_frozen_eval import run_retrieval_clustering as retrieval
        args = retrieval.parse_args(['--checkpoint','/teacher.pth','--train-config','/config.yaml','--output-dir','/results','--no-save-features'])
        self.assertTrue(args.no_save_features)
        import numpy as np
        with patch.object(retrieval,'extract_features',return_value=(np.zeros((2,3)),np.array([0,1]))) as extract:
            retrieval._extract(object(),object(),Path('/features'),args)
            self.assertFalse(extract.call_args.kwargs['save_features'])

    def test_source_registration_avoids_repeated_nfs_hash_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); source = path/'data.npz'; source.write_bytes(b'official')
            digest = stage.sha256(source)
            stage.verify_source(path,source,digest)
            with patch.object(stage,'sha256',side_effect=AssertionError('repeated hash')):
                stage.verify_source(path,source,digest)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'Source changed'):
                stage.verify_source(path,source,digest)

    def test_named_nct_1k_accepts_the_full_999_row_release(self):
        from dinov3.eval.bio_frozen_eval import retrieval_clustering
        class FixedRelease:
            rows = [(b'image',i % 9,str(i)) for i in range(999)]
            def __len__(self): return len(self.rows)
        with patch.object(retrieval_clustering,'build_retrieval_dataset',return_value=(FixedRelease(),[])):
            report = stage.retrieval_preflight('nct-crc-he-1k',Path('/benchmark'))
            self.assertEqual(report['counts']['n_samples'],999)


if __name__=='__main__': unittest.main()
