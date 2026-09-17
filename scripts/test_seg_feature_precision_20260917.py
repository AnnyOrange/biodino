"""Precision/cache regression checks; all tests run without a GPU."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import TensorDataset
from dinov3.eval.bio_segmentation import feature_extractor as features
import validate_bio_eval_formal_v3 as preflight


class Backbone(torch.nn.Module):
    in_chans = 3

    def get_intermediate_layers(self, images, **kwargs):
        return (images[:, :1, ::16, ::16],)


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.dataset = TensorDataset(torch.ones(2, 3, 32, 32), torch.zeros(2, 32, 32, dtype=torch.long))

    def test_precisions_and_batches_have_different_cache_keys(self):
        tags = {features.feature_precision_tag(dtype, batch) for dtype in features.AUTOCAST_DTYPES for batch in (8, 32)}
        self.assertEqual(len(tags), 6)
        with self.assertRaises(ValueError):
            features.feature_precision_tag("bf16", 0)

    def test_array_extraction_forwards_requested_dtype(self):
        for name, dtype in features.AUTOCAST_DTYPES.items():
            with patch.object(features.torch, "autocast", wraps=torch.autocast) as autocast:
                array, masks, instances = features.extract_features(
                    Backbone(), self.dataset, batch_size=2, num_workers=0,
                    device=torch.device("cpu"), autocast_dtype=name)
                self.assertEqual(autocast.call_args.kwargs["dtype"], dtype)
                self.assertEqual(array.dtype, np.float16)
                self.assertEqual(array.shape, (2, 1, 2, 2))

    def test_streaming_cache_records_compute_and_storage_dtype(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "cache.npz")
            features.extract_features_to_cache(
                path, Backbone(), self.dataset, 1, 2, 0, torch.device("cpu"),
                "test", 16, 1, 1, autocast_dtype="bf16")
            with np.load(path) as cache:
                self.assertEqual(str(cache["autocast_dtype"]), "bf16")
                self.assertEqual(str(cache["cache_storage_dtype"]), "float16")
                self.assertEqual(int(cache["feature_batch_size"]), 2)

    def test_preflight_collects_failures_and_never_passes_without_commands(self):
        functions = ("_validate_dataset_policy", "_validate_v3_additions", "_validate_registry",
                     "_validate_conic", "_validate_livecell", "_validate_pannuke")
        patches = [patch.object(preflight, name, return_value={}) for name in functions]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        with patch.object(preflight, "_validate_v3_additions", side_effect=AssertionError("pending CTC")):
            report = preflight.build_report({"protocol_id": "test"}, Path("/not-used"), None)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["failed_checks"], ["v3_additions", "commands"])
        self.assertEqual(report["checks"]["registry"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
