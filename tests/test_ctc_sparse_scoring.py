"""Compare bounded mask matching and graph edges to the pinned CTC evaluator."""
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'outputs/02_eval_runtime/py-ctcmetrics'
sys.path[:0] = [str(ROOT), str(VENDOR)]
from ctc_metrics.utils.representations import match as pinned_match, create_edge_mapping
from dinov3.eval.bio_tracking.ctc_mask_matching import match_paths
from dinov3.eval.bio_tracking.ctc_metrics_requested import (
    score_requested_metrics, calculate_requested_metrics, create_edge_mapping_indexed)


class SparseScoringTests(unittest.TestCase):
    def test_masks_equal_pinned_for_fragmentation_merges_and_missing_background(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'reference.tif', Path(tmp) / 'computed.tif'
            for shape in [(11, 13), (3, 11, 13)]:
                for _ in range(12):
                    ref = rng.integers(0, 8, size=shape, dtype=np.uint32)
                    comp = np.where(rng.random(shape) < .75, ref * 101,
                                    rng.integers(0, 12, size=shape)).astype(np.uint32)
                    tifffile.imwrite(a, ref, photometric='minisblack')
                    tifffile.imwrite(b, comp, photometric='minisblack')
                    self.assertEqual(match_paths(str(a), str(b)), pinned_match(str(a), str(b)))
            for ref, comp in [(np.ones((8, 8), np.uint32), np.ones((8, 8), np.uint32)),
                              (np.zeros((8, 8), np.uint32), np.zeros((8, 8), np.uint32))]:
                tifffile.imwrite(a, ref)
                tifffile.imwrite(b, comp)
                self.assertEqual(match_paths(str(a), str(b)), pinned_match(str(a), str(b)))
            self.assertEqual(match_paths(None, str(b)), pinned_match(None, str(b)))
            self.assertEqual(match_paths(str(a), None), pinned_match(str(a), None))

    def test_sparse_segmentation_slices_equal_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            volume = np.zeros((4, 8, 9), np.uint32)
            volume[:, 1:5, 2:6] = 19
            pred = root / 'mask000.tif'
            tifffile.imwrite(pred, volume, photometric='minisblack')
            for z in (1, 3):
                tifffile.imwrite(root / f'man_seg000_{z:03d}.tif', volume[z])
            ref = str(root / 'man_seg000.tif')
            self.assertEqual(match_paths(ref, str(pred)), pinned_match(ref, str(pred)))

    def test_edge_table_equals_pinned_including_division(self):
        tracks = np.array([[1, 0, 0, 0], [2, 1, 2, 1], [3, 1, 2, 1]])
        labels = [[1], [2, 3], [2, 3]]
        cumulative = np.cumsum([0] + list(map(len, labels)))
        truth = np.ones(5, dtype=bool)
        np.testing.assert_array_equal(create_edge_mapping_indexed(tracks, labels, truth, cumulative),
                                      create_edge_mapping(tracks, labels, truth, cumulative))

    def test_complete_scores_equal_pinned_without_pool(self):
        evaluator = importlib.import_module('ctc_metrics.scripts.evaluate')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, gt = root / '02_RES', root / '02_GT'
            result.mkdir()
            (gt / 'TRA').mkdir(parents=True)
            (gt / 'SEG').mkdir()
            np.savetxt(result / 'res_track.txt', [[1, 0, 2, 0]], fmt='%d')
            np.savetxt(gt / 'TRA/man_track.txt', [[1, 0, 2, 0]], fmt='%d')
            for i in range(3):
                mask = np.zeros((16, 16), np.uint32)
                mask[3:10, 4:12] = 1
                for p in [result / f'mask{i:03d}.tif', gt / 'TRA' / f'man_track{i:03d}.tif',
                          gt / 'SEG' / f'man_seg{i:03d}.tif']:
                    tifffile.imwrite(p, mask)
            comp, ref, trajectory, segmentation, _, valid = evaluator.load_data(
                str(result), str(gt), True, True, threads=1)
            expected, _ = calculate_requested_metrics(comp, ref, trajectory, segmentation, valid)
            original = evaluator.match_tracks
            with patch.object(evaluator, 'Pool', side_effect=AssertionError('No pool allowed')):
                actual, info = score_requested_metrics(result, gt, VENDOR, threads=16)
            self.assertEqual(actual, expected)
            self.assertEqual(actual['SEG'], 1.0)
            self.assertEqual(info['threads'], 1)
            self.assertIs(evaluator.match_tracks, original)

    def test_high_label_count_under_small_allocation_budget(self):
        code = f'''import sys,resource
from pathlib import Path
sys.path.insert(0,{str(ROOT)!r})
import numpy as np
from dinov3.eval.bio_tracking.ctc_mask_matching import match_arrays
a=np.arange(100001,dtype=np.uint32).reshape(1,-1)
vm=int(Path('/proc/self/statm').read_text().split()[0])*4096
soft,hard=resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS,(vm+256*1024**2,hard))
r=match_arrays(a,a.copy())
assert len(r[2])==100000 and all(v==1.0 for v in r[4])
print('bounded')
'''
        p = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), 'bounded')


if __name__ == '__main__':
    unittest.main()
