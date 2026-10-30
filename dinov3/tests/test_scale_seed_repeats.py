import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'run_scale_seed_repeats_20261028.py'
SPEC = importlib.util.spec_from_file_location('run_scale_seed_repeats_20261028', SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FiniteAuditBatchChangeTests(unittest.TestCase):
    def test_repack_preserves_every_consumed_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_command = ['train.batch_size_per_gpu=32',
                           'optim.gradient_accumulation_steps=16']
            for rank in range(2):
                audit = root / f'consumed_sample_keys_rank{rank:02d}.jsonl'
                with audit.open('w') as handle:
                    for batch in range(16):
                        keys = [f'r{rank}-s{batch * 32 + index}' for index in range(32)]
                        handle.write(json.dumps(keys) + '\n')
                    handle.write(json.dumps(['uncheckpointed']) + '\n')
            runner.repack_finite_audits(root, checkpoint=0,
                                        old_command=old_command, new_microbatch=64)
            for rank in range(2):
                audit = root / f'consumed_sample_keys_rank{rank:02d}.jsonl'
                batches = [json.loads(line) for line in audit.read_text().splitlines()]
                self.assertEqual(len(batches), 8)
                self.assertTrue(all(len(batch) == 64 for batch in batches))
                self.assertEqual([key for batch in batches for key in batch],
                                 [f'r{rank}-s{index}' for index in range(512)])
                backup = root / f'consumed_sample_keys_rank{rank:02d}_before_batch_change_0.jsonl'
                self.assertEqual(len(backup.read_text().splitlines()), 17)


if __name__ == '__main__':
    unittest.main()
