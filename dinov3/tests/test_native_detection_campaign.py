from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from dinov3.eval.bio_detection.campaign import (
    configuration, grid, load_data, remap_path, select, train_recipe, validate_freeze, validate_frozen,
)
from dinov3.eval.bio_frozen_eval.candidate_datasets import digest
from dinov3.eval.bio_frozen_eval.protocol_campaign import ROOT


class DetectionCampaignTests(unittest.TestCase):
    def setUp(self):
        registry = json.loads((ROOT / "Evaluation Rules/native_detection_candidates.json").read_text())
        self.config = configuration(registry, "BCCD")

    def rows(self):
        return [{"status": "SUCCESS", "recipe": recipe, "epochs": [{}] * recipe["epochs"],
                 "metrics": {"bbox_ap_50_95": .1 + i * .01}, "zero_based_layers": [11]}
                for i, recipe in enumerate(grid(self.config))]

    def identity(self):
        return {"model_budget": "1TB", "dataset": "BCCD", "git_commit": "abc", "implementation_sha256": "impl",
                "config_sha256": "config", "registry_sha256": "registry", "manifest_sha256": "data", "seed": 0,
                "model_id": "selector", "checkpoint_sha256": "weight", "train_config_sha256": "train",
                "model_family": "hs6-S+", "model_depth": 12, "batch_size": 2, "classes": ["RBC"],
                "environment": {"environment_sha256": "env"}}

    def frozen(self):
        frozen = {**self.identity(), "status": "FROZEN", "selection_budget": "1TB"}
        frozen["protocol_sha256"] = digest(frozen)
        return frozen

    def test_full_grid_four_candidates_final_epoch_selection(self):
        rows = self.rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual(select(rows, self.config), rows[-1])
        self.assertTrue(all(row["recipe"]["epochs"] == 10 for row in rows))

    def test_missing_duplicate_failed_or_partial_candidates_rejected(self):
        rows = self.rows()
        for damaged in (rows[:-1], [rows[0]] * 4):
            with self.assertRaisesRegex(ValueError, "grid"):
                select(damaged, self.config)
        rows[0]["epochs"] = []
        with self.assertRaisesRegex(ValueError, "incomplete"):
            select(rows, self.config)

    def test_model_repointing_or_budget_substitution_cannot_freeze(self):
        identity = self.identity()
        sweep = {**identity, "status": "SUCCESS", "rows": self.rows()}
        sweep["checkpoint_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            validate_freeze(sweep, identity, self.config)
        sweep["checkpoint_sha256"] = identity["checkpoint_sha256"]
        identity["model_budget"] = "5TB"
        with self.assertRaisesRegex(ValueError, "1TB"):
            validate_freeze(sweep, identity, self.config)

    def test_frozen_commit_family_environment_and_extraction_batch_bound(self):
        frozen = self.frozen()
        validate_frozen(frozen, self.identity())
        for key in ("git_commit", "model_family", "batch_size", "manifest_sha256"):
            changed = self.identity()
            changed[key] = "wrong"
            with self.assertRaisesRegex(ValueError, key):
                validate_frozen(frozen, changed)
        changed = self.identity()
        changed["environment"]["environment_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "environment"):
            validate_frozen(frozen, changed)

    def test_budget_progression_no_retuning_or_family_transfer(self):
        frozen = self.frozen()
        identity = self.identity()
        identity["model_budget"] = "5TB"
        previous = {**self.identity(), "status": "SUCCESS", "protocol_sha256": frozen["protocol_sha256"]}
        validate_frozen(frozen, identity, previous)
        previous["dataset"] = "BBBC041"
        with self.assertRaisesRegex(ValueError, "progression"):
            validate_frozen(frozen, identity, previous)

    def test_portable_canonical_paths_and_escape_rejection(self):
        self.assertEqual(remap_path("/mnt/huawei_deepcad/benchmark/x/y.jpg", "/tmp/native-data"),
                         Path("/tmp/native-data/x/y.jpg"))
        self.assertEqual(remap_path("x/y.jpg", "/tmp/native-data"), Path("/tmp/native-data/x/y.jpg"))
        for path in ("../../escape.jpg", "/etc/passwd"):
            with self.assertRaises(ValueError):
                remap_path(path, "/tmp/native-data")

    def test_wrong_dataset_manifest_rejected_before_data_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps({"dataset": "BBBC041"}))
            args = SimpleNamespace(manifest=str(path), dataset="BCCD", benchmark_root=temporary)
            with self.assertRaisesRegex(ValueError, "different dataset"):
                load_data(args)

    def test_training_records_all_epochs_and_frozen_test_only_once(self):
        class TinyDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, images, targets=None):
                return {"loss_classifier": self.head.square()}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            Image.new("RGB", (8, 8)).save(image_path)
            manifest = {"classes": ["RBC"], "records": [
                {"sample_id": split, "split": split, "path": str(image_path),
                 "boxes": [[0, 0, 4, 4]], "labels": [1], "width": 8, "height": 8}
                for split in ("train", "val", "test")]}
            args = SimpleNamespace(batch_size=1, workers=0, device="cpu")
            recipe = {**grid(self.config)[0], "epochs": 2}
            with patch("dinov3.eval.bio_detection.campaign.detector", return_value=(TinyDetector(), [11])), \
                 patch("dinov3.eval.bio_detection.campaign.evaluate", return_value={"bbox_ap_50_95": .25}) as score:
                report = train_recipe(args, self.config, manifest, self.identity(), recipe, root / "heads", "test")
            self.assertEqual(score.call_count, 1)
            self.assertEqual(len(report["epochs"]), 2)
            self.assertIsNone(report["epochs"][0]["metrics"])
            self.assertEqual(report["metrics"]["bbox_ap_50_95"], .25)
            self.assertTrue(all(Path(epoch["head_checkpoint"]).is_file() for epoch in report["epochs"]))
            self.assertTrue(all(len(epoch["step_losses"]) == 1 for epoch in report["epochs"]))


if __name__ == "__main__":
    unittest.main()
