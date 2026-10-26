"""A stale source root must not prevent a complete, auditable 1PB batch."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from scripts.rolling_1pb_wds import _verify_batch, _weighted_worker, produce_batch


class UnreadableRootTest(unittest.TestCase):
    def test_low_space_does_not_reserve_a_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = Path(temporary) / "queue"
            result = produce_batch(Path(temporary) / "unused.sqlite", queue,
                                   batch_samples=1, max_live_batches=2,
                                   workers=1, samples_per_shard=1,
                                   file_block=1, seed=7,
                                   min_free_gib=1_000_000)
            self.assertEqual(result, "LOW_SPACE")
            self.assertEqual(list((queue / ".building").iterdir()), [])

    def test_fills_from_readable_root_and_verifies_actual_mix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_root = root / "missing"
            readable_root = root / "readable"
            readable_root.mkdir()
            image = readable_root / "image.tif"
            tifffile.imwrite(image, np.ones((32, 32), dtype=np.uint16))
            worker_dir = root / "workers" / "w00"
            worker_dir.mkdir(parents=True)
            weights = {str(missing_root): 1, str(readable_root): 1}
            rows = [
                (str(missing_root / "gone.tif"), 1, 0, "missing", "2"),
                (str(image), 1, 0, "readable", "2"),
            ]
            report = _weighted_worker((0, "batch-000001", rows, 2, worker_dir,
                                       2, 1, 7, weights, {}))
            self.assertEqual(report["samples"], 2)
            self.assertEqual(report["root_sample_counts"],
                             {str(missing_root): 0, str(readable_root): 2})
            self.assertEqual(report["unavailable_roots"], [str(missing_root)])

            shards = root / "shards"
            shards.mkdir()
            (worker_dir / "staged_w00_00000.tar").rename(shards / "shard-w00-00000.tar")
            (worker_dir / "provenance.jsonl").rename(root / "provenance_w00.jsonl")
            summary = _verify_batch(root, "batch-000001", 2, 2, [report],
                                    verify_workers=1, root_weights=weights)
            self.assertEqual(summary["root_sample_counts"],
                             {str(missing_root): 0, str(readable_root): 2})
            self.assertEqual(summary["source_mix_policy"],
                             "weighted_with_unreadable_root_fallback")


if __name__ == "__main__":
    unittest.main()
