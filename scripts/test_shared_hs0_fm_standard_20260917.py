import importlib.util
import tempfile
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('queue',ROOT/'scripts/run_shared_frozen_stage1_20260917.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
from dinov3.eval.bio_frozen_eval.run_external_standard import cls_patch_feature


class Tests(unittest.TestCase):
    def test_registers_not_in_patch_mean(self):
        tokens=torch.tensor([[[1.,0.],[100.,100.],[0.,2.],[0.,4.]]])
        feature=cls_patch_feature(tokens,2)
        expected=torch.nn.functional.normalize(torch.tensor([[1.,0.,0.,3.]]),dim=1)
        torch.testing.assert_close(feature,expected)

    def test_convolution_or_no_cls_not_silently_accepted(self):
        for tokens,prefix in [(torch.zeros(1,2,2,2),1),(torch.zeros(1,3,2),0)]:
            with self.assertRaises(ValueError):cls_patch_feature(tokens,prefix)

    def test_external_command_fixed_batch(self):
        task=dict(asset=dict(kind='external',model_id='dinov2'),dataset=dict(task='classification',dataset='breastmnist'))
        command=queue.command(task,Path('/output'),'/benchmark')
        self.assertIn('dinov3.eval.bio_frozen_eval.run_external_standard',command)
        self.assertEqual(command[command.index('--batch-size')+1],'64')

    def test_external_directory_change_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'_state/inputs').mkdir(parents=True)
            assets=root/'assets';assets.mkdir();weights=assets/'model.safetensors';weights.write_bytes(b'published')
            asset=dict(kind='external',model_id='dinov2',path=str(assets),config='')
            record=queue.checkpoint_record(root,asset)
            self.assertEqual(record['teacher_key'],'published_frozen_external')
            weights.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):queue.checkpoint_record(root,asset)


if __name__=='__main__':unittest.main()
