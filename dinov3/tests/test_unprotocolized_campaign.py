"""CPU tests for candidate split validation and irreversible selection guards."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dinov3.eval.bio_frozen_eval.candidate_datasets import digest, validate_records
from dinov3.eval.bio_frozen_eval.protocol_campaign import (
    choose_sweep, feature_layers, validate_frozen, validate_model,
    validate_previous, validate_sync, write_new,
)


class CampaignTests(unittest.TestCase):
    def rows(self):
        return [{"feature": f, "alpha": a, "status": "SUCCESS", "metrics": {"mae": 1.0}}
                for f in ["last", "4-even"] for a in [0.1, 1.0]]

    def frozen(self):
        value = {"git_commit": "abc", "registry_sha256": "registry",
                 "manifest_sha256": digest({"records": []}), "selection_budget": "1TB",
                 "status": "FROZEN_DEVELOPMENT_PROTOCOL", "dataset": "test",
                 "model_family": "hs6-S+", "seed": 0}
        value["protocol_sha256"] = digest(value)
        return value

    def test_existing_even4_definitions(self):
        self.assertEqual(feature_layers(12, "4-even"), [2, 5, 8, 11])
        self.assertEqual(feature_layers(24, "4-even"), [4, 11, 17, 23])
        self.assertEqual(feature_layers(40, "4-even"), [9, 19, 29, 39])
        self.assertNotEqual(feature_layers(12, "4-even"), feature_layers(12, "last4"))
        self.assertEqual(feature_layers(12, "last"), [11])

    def test_shallow_or_unknown_features_fail(self):
        for depth, preset in [(0, "last"), (3, "4-even"), (12, "nonsense")]:
            with self.assertRaises(ValueError):
                feature_layers(depth, preset)

    def test_stable_tie_order(self):
        winner = choose_sweep(list(reversed(self.rows())), ["last", "4-even"], [0.1, 1.0])
        self.assertEqual((winner["feature"], winner["alpha"]), ("last", 0.1))

    def test_failed_incomplete_and_duplicate_sweep(self):
        for rows in [self.rows()[:-1], self.rows() + [self.rows()[0]]]:
            with self.assertRaises(ValueError):
                choose_sweep(rows, ["last", "4-even"], [0.1, 1.0])
        rows = self.rows()
        rows[0]["status"] = "FAILURE"
        with self.assertRaises(ValueError):
            choose_sweep(rows, ["last", "4-even"], [0.1, 1.0])

    def test_nonfinite_selection_fails(self):
        for metric in [None, float("nan"), float("inf")]:
            rows = self.rows()
            rows[0]["metrics"]["mae"] = metric
            with self.assertRaises(ValueError):
                choose_sweep(rows, ["last", "4-even"], [0.1, 1.0])

    def test_frozen_matches_all_budgets(self):
        for budget in ["1TB", "5TB", "20TB"]:
            validate_frozen(self.frozen(), "abc", "registry", {"records": []}, budget)

    def test_frozen_tampering(self):
        value = self.frozen()
        value["seed"] = 1
        with self.assertRaises(ValueError):
            validate_frozen(value, "abc", "registry", {"records": []}, "5TB")

    def test_code_registry_and_sample_mismatches(self):
        for commit, registry, manifest in [("new", "registry", {"records": []}),
                                           ("abc", "new", {"records": []}),
                                           ("abc", "registry", {"records": ["new"]})]:
            with self.assertRaises(ValueError):
                validate_frozen(self.frozen(), commit, registry, manifest, "5TB")

    def test_no_5tb_selection_model(self):
        value = self.frozen()
        value["selection_budget"] = "5TB"
        value["protocol_sha256"] = digest({k: v for k, v in value.items() if k != "protocol_sha256"})
        with self.assertRaises(ValueError):
            validate_frozen(value, "abc", "registry", {"records": []}, "20TB")

    def test_model_identity_binding(self):
        model = {"model_family": "hs6-S+", "model_budget": "1TB", "checkpoint_sha256": "weights",
                 "train_config_sha256": "cfg", "branch": "teacher"}
        registry = {"models": {"splus": model}}
        self.assertEqual(validate_model(registry, "splus", "hs6-S+", "1TB", "weights", "cfg"), model)
        for family, budget, weights, config in [("hs6-H+", "1TB", "weights", "cfg"),
                                               ("hs6-S+", "20TB", "weights", "cfg"),
                                               ("hs6-S+", "1TB", "wrong", "cfg"),
                                               ("hs6-S+", "1TB", "weights", "wrong")]:
            with self.assertRaises(ValueError):
                validate_model(registry, "splus", family, budget, weights, config)

    def test_budget_order_and_result_identity(self):
        frozen = self.frozen()
        prior = {"status": "SUCCESS", "model_budget": "1TB", "protocol_sha256": frozen["protocol_sha256"],
                 "git_commit": "abc", "dataset": "test", "model_family": "hs6-S+", "seed": 0}
        validate_previous(prior, frozen, "5TB")
        with self.assertRaises(ValueError):
            validate_previous(prior, frozen, "20TB")
        with self.assertRaises(ValueError):
            validate_previous(None, frozen, "5TB")
        prior["model_family"] = "hs6-H+"
        with self.assertRaises(ValueError):
            validate_previous(prior, frozen, "5TB")

    def test_sync_all_four_and_duplicates(self):
        rows = [{"host": host, "git_commit": "abc", "registry_sha256": "registry",
                 "code_clean": True, "environment_pass": True}
                for host in ["local", "5090-hxw-xzj", "5090-lyx-xr", "suxin-8H100-1"]]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sync.json"
            path.write_text(json.dumps({"machines": rows}))
            validate_sync(path, "abc", "registry")
            for wrong in [rows[:-1], rows + [rows[0]]]:
                path.write_text(json.dumps({"machines": wrong}))
                with self.assertRaises(ValueError):
                    validate_sync(path, "abc", "registry")
            changed = copy.deepcopy(rows)
            changed[0]["environment_pass"] = False
            path.write_text(json.dumps({"machines": changed}))
            with self.assertRaises(ValueError):
                validate_sync(path, "abc", "registry")

    def test_sample_and_content_leakage(self):
        records = [{"sample_id": str(i), "split": split, "group": str(i), "target": i,
                    "image_sha256": str(i)} for i, split in enumerate(["train", "val", "test"])]
        self.assertEqual(validate_records(records)["test"]["samples"], 1)
        for key in ["sample_id", "group", "image_sha256"]:
            altered = copy.deepcopy(records)
            altered[2][key] = altered[0][key]
            with self.assertRaises(ValueError):
                validate_records(altered)
        with self.assertRaises(ValueError):
            validate_records(records[:-1])

    def test_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            write_new(path, {"result": 1})
            with self.assertRaises(FileExistsError):
                write_new(path, {"result": 2})
            self.assertEqual(json.loads(path.read_text()), {"result": 1})

    def test_explicit_feature_model_indices(self):
        from contextlib import nullcontext
        from unittest.mock import patch
        import torch
        from dinov3.eval.bio_classification.common import LinearFeatureModel

        class Backbone(torch.nn.Module):
            def get_intermediate_layers(self, images, n, **kwargs):
                self.indices = n
                return [(torch.ones(2, 3, 4) * index, torch.ones(2, 4) * index) for index in n]

        backbone = Backbone()
        model = LinearFeatureModel(backbone, [2, 5, 8, 11], True, torch.bfloat16)
        with patch("torch.autocast", return_value=nullcontext()):
            output = model(torch.zeros(2, 3, 8, 8))
        self.assertEqual(backbone.indices, [2, 5, 8, 11])
        self.assertEqual(tuple(output.shape), (2, 20))
        np.testing.assert_equal(output[0].numpy(), np.repeat([2, 5, 8, 11, 11], 4))

    def test_encoder_preserves_legacy_and_validates_explicit_indices(self):
        from unittest.mock import patch
        import torch
        from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])

        with patch("dinov3.eval.bio_frozen_eval.encoder.load_dinov3_backbone", return_value=Backbone()):
            legacy = Dinov3CkptEncoder(Path("unused.pth"), Path("unused.yaml"), "cpu", 4, True,
                                       torch.bfloat16)
            self.assertEqual(legacy.model.n_last_blocks, 4)
            explicit = Dinov3CkptEncoder(Path("unused.pth"), Path("unused.yaml"), "cpu", 1, True,
                                         torch.bfloat16, feature_layers=[2, 5, 8, 11])
            self.assertEqual(explicit.model.n_last_blocks, [2, 5, 8, 11])
            for invalid in [[], [2, 2], [3, 2], [-1], [12]]:
                with self.assertRaises(ValueError):
                    Dinov3CkptEncoder(Path("unused.pth"), Path("unused.yaml"), "cpu", 1, True,
                                       torch.bfloat16, feature_layers=invalid)


if __name__ == "__main__":
    unittest.main()
