import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dinov3.eval.bio_frozen_eval.expansion_campaign import choose, candidates, verify_data, read_manifest, validate_search, select_protocol, cv_score, pin_tensor_normalization, ensure_published


class ExpansionCampaignTests(unittest.TestCase):
    def test_hest_requires_patient_audit_and_rejects_known_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            args = SimpleNamespace(manifest=str(path), dataset="HEST_Benchmark", tissue="SKCM")
            manifest = {"status": "PASS", "release_status": "PASS", "payload_verified": True,
                        "tasks": {"SKCM": {}}}
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "patient grouping audit"):
                read_manifest(args, {"adapter": "hest", "seed": 1})
            manifest["patient_grouping"] = {"status": "PASS_KNOWN_PATIENTS_ONE_UNRESOLVED",
                                            "unknown_ids": ["TENX111"], "failures": []}
            path.write_text(json.dumps(manifest))
            self.assertEqual(read_manifest(args, {"adapter": "hest", "seed": 1}), manifest)
            manifest["patient_grouping"]["failures"] = [{"patient_overlap": ["Patient1"]}]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unresolved leakage"):
                read_manifest(args, {"adapter": "hest", "seed": 1})

    def test_published_pinned_ancestor_is_allowed_but_unpublished_commit_is_rejected(self):
        with patch("subprocess.check_output", return_value="new refs/heads/main\n"), patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
            self.assertEqual(ensure_published("old"), "new")
        with patch("subprocess.check_output", return_value="new refs/heads/main\n"), patch("subprocess.run", side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)]):
            with self.assertRaisesRegex(ValueError, "published authoritative"):
                ensure_published("unpublished")

    def test_tensor_stats_are_frozen_independently_of_checkpoint_config(self):
        encoder = SimpleNamespace(mc_mean=(9.,)*3, mc_std=(7.,)*3)
        prep = {"tensor_input": True, "tensor_mean": [.1,.2,.3], "tensor_std": [.4,.5,.6]}
        pin_tensor_normalization(encoder, prep)
        self.assertEqual(encoder.mc_mean, (.1,.2,.3))
        self.assertEqual(encoder.mc_std, (.4,.5,.6))
        with self.assertRaises(ValueError):
            pin_tensor_normalization(encoder, {**prep, "tensor_std": [0.,.5,.6]})
    def test_film_selects_independent_inner_winners_not_global_winner(self):
        config = {"adapter": "film", "primary_metric": "balanced_accuracy", "direction": "max",
                  "search": {"features": ["last"], "hyperparameters": {"C": [.1, 1.]}}}
        rows = []
        for C in [.1, 1.]:
            folds = [{"repetition": r, "fold": f, "metrics": {"balanced_accuracy":
                     float(C == (.1 if f == 0 else 1.))}} for r in range(3) for f in range(3)]
            rows.append({"status": "SUCCESS", "feature": "last", "zero_based_layers": [11],
                         "hyperparameters": {"C": C}, "metrics": {"balanced_accuracy": .5}, "folds": folds})
        selected = select_protocol(rows, config, 12)
        self.assertEqual(len(selected["fold_protocols"]), 9)
        for entry in selected["fold_protocols"]:
            self.assertEqual(entry["hyperparameters"]["C"], .1 if entry["fold"] == 0 else 1.)
        rows[0]["folds"].pop()
        with self.assertRaisesRegex(ValueError, "every unique"):
            select_protocol(rows, config, 12)

    def test_vgg_development_scoring_does_not_require_test_features(self):
        manifest = {"records": [{"sample_id": str(i), "source_split": "development"} for i in range(8)],
                    "folds": [{"fold": 0, "train": ["0", "1", "2", "3"],
                               "val": ["4", "5", "6", "7"], "test": ["hidden"]}]}
        x = np.arange(8, dtype=float)[:, None]
        metric, folds = cv_score({"adapter": "vgg"}, {"alpha": 1.}, {"development": (x, x[:, 0])}, manifest)
        self.assertTrue(np.isfinite(metric["mae"]))
        self.assertEqual(len(folds), 1)

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
