import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from dinov3.eval.bio_frozen_eval.expansion_campaign import choose, candidates, verify_data, read_manifest, validate_search


class ExpansionCampaignTests(unittest.TestCase):
    def test_grid_keeps_declared_order_and_all_candidates(self):
        config = {"search": {"hyperparameters": {"k": [10, 20], "temperature": [.07, .2]}}}
        self.assertEqual(candidates(config), [{"k": 10, "temperature": .07}, {"k": 10, "temperature": .2},
                                               {"k": 20, "temperature": .07}, {"k": 20, "temperature": .2}])

    def test_selection_direction_ties_and_fail_closed(self):
        rows = [{"status": "SUCCESS", "metrics": {"m": value}, "index": i} for i, value in enumerate([3, 1, 1])]
        self.assertEqual(choose(rows, "m", "min", 3)["index"], 1)
        self.assertEqual(choose(rows, "m", "max", 3)["index"], 0)
        for changed in (rows[:2], [{"status": "FAILURE", "metrics": {}}] + rows[1:],
                        [{"status": "SUCCESS", "metrics": {"m": np.nan}}] + rows[1:]):
            with self.assertRaises(ValueError):
                choose(changed, "m", "min", 3)

    def test_manifest_payload_mutation_is_detected(self):
        from dinov3.eval.bio_frozen_eval.candidate_datasets import file_digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "external_benchmarks_20260901/OpenCell_projections"
            folder.mkdir(parents=True)
            image = folder / "a.tif"
            image.write_bytes(b"verified")
            manifest = {"records": [{"sample_id": "a", "path": "a.tif", "image_sha256": file_digest(image)}]}
            args = SimpleNamespace(benchmark_root=directory)
            verify_data(args, {"adapter": "opencell"}, manifest)
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "payload changed"):
                verify_data(args, {"adapter": "opencell"}, manifest)

    def test_cross_dataset_manifest_and_seed_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"dataset": "cytoimagenet-source-grouped", "task": "classification", "seed": 0}))
            with self.assertRaisesRegex(ValueError, "different dataset"):
                read_manifest(SimpleNamespace(manifest=path, dataset="AllenCell_Morphology"), {"seed": 0, "adapter": "grouped"})
            path.write_text(json.dumps({"dataset": "opencell-major-localization-protein-heldout", "task": "classification", "seed": 4}))
            with self.assertRaisesRegex(ValueError, "seed"):
                read_manifest(SimpleNamespace(manifest=path, dataset="OpenCell"), {"seed": 0, "adapter": "opencell"})

    def test_repeated_or_changed_feature_candidates_cannot_freeze(self):
        config = {"search": {"features": ["last"], "hyperparameters": {"alpha": [.1, 1.]}}}
        rows = [{"feature": "last", "hyperparameters": {"alpha": a}, "zero_based_layers": [11]} for a in [.1, 1.]]
        validate_search(rows, config, 12)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            validate_search([rows[0], rows[0]], config, 12)
        rows[1]["zero_based_layers"] = [10]
        with self.assertRaisesRegex(ValueError, "layer"):
            validate_search(rows, config, 12)

    def test_candidate_registry_is_explicitly_not_frozen(self):
        root = Path(__file__).resolve().parents[2]
        registry = json.loads((root / "Evaluation Rules/repaired_protocol_candidates.json").read_text())
        self.assertIn("REQUIRE_ACTUAL", registry["status"])
        for config in registry["datasets"].values():
            self.assertIn(config["protocol_source"], ["OFFICIAL", "ESTABLISHED_CONVENTION", "PROPOSED_BY_US"])
            self.assertIn("last", config["search"]["features"])
            self.assertIn("4-even", config["search"]["features"])
            self.assertTrue(candidates(config))


if __name__ == "__main__":
    unittest.main()
