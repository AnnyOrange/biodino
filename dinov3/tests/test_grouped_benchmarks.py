import csv
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from dinov3.eval.bio_frozen_eval.grouped_benchmarks import (
    ALLEN_ROOT, CYTO_ROOT, GroupedBenchmarkDataset, build_allen_morphology_manifest,
    build_cytoimagenet_manifest, validate_grouped_manifest,
)


def write_metadata(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class GroupedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_cyto(self):
        root = self.root / CYTO_ROOT
        rows = []
        for label in ("a", "b", "rare"):
            directory = root / "extracted" / label
            directory.mkdir(parents=True)
            for index in range(2 if label == "rare" else 12):
                image = np.arange(64, dtype=np.uint8).reshape(8, 8)
                image = np.roll(image, index % 8, axis=0) + (index // 8) * 3 + {"a": 0, "b": 15, "rare": 30}[label]
                filename = f"{index}.png"
                Image.fromarray(image).save(directory / filename)
                rows.append({"idx": f"{label}-{index // 2}", "label": label,
                             "path": "/cytoimagenet/" + label, "filename": filename,
                             "dset": "train" if index % 2 else "val", "dir_name": "source"})
        write_metadata(root / "metadata/metadata.csv", rows)

    def test_cyto_source_groups_stratified_and_rare_class_explicit(self):
        self.make_cyto()
        manifest = build_cytoimagenet_manifest(self.root)
        self.assertEqual(manifest, build_cytoimagenet_manifest(self.root))
        self.assertEqual(manifest["classes"], ["a", "b"])
        self.assertEqual(manifest["excluded_class_source_group_counts"], {"rare": 1})
        self.assertTrue(manifest["content_audit"]["completed"])
        self.assertEqual(manifest["official_source_idx_split_leakage"], 13)
        image, target, identity = GroupedBenchmarkDataset(self.root, manifest, "test")[0]
        self.assertEqual(image.mode, "RGB")
        self.assertIsInstance(target, int)

    def test_source_group_leakage_rejected(self):
        self.make_cyto()
        manifest = build_cytoimagenet_manifest(self.root)
        damaged = deepcopy(manifest)
        test = next(r for r in damaged["records"] if r["split"] == "test")
        train = next(r for r in damaged["records"] if r["split"] == "train")
        train["group"] = test["group"]
        with self.assertRaisesRegex(ValueError, "Source group leakage"):
            validate_grouped_manifest(damaged)

    def make_allen(self):
        rows = []
        root = self.root / ALLEN_ROOT
        for i in range(24):
            relative = f"Allen/crops/{i}.tiff"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.roll(np.arange(256, dtype=np.uint8).reshape(8, 32), i % 8, axis=0) + i).save(path)
            rows.append({"train_test_split": "Task_two" if i >= 20 else "Train" if i % 2 else "Task_one",
                         "cell_volume": str(100 + i), "CellId": str(i), "WellId": str(i // 2),
                         "FOVId": str(i // 2), "PlateId": "1", "file_path": relative, "channel_width": "8"})
        write_metadata(root / "Allen/enriched_meta.csv", rows)

    def test_allen_well_grouping_preserves_task_two_test(self):
        self.make_allen()
        manifest = build_allen_morphology_manifest(self.root)
        self.assertEqual(manifest["official_source_fov_split_leakage"], 10)
        self.assertEqual({r["sample_id"] for r in manifest["records"] if r["split"] == "test"},
                         {"20", "21", "22", "23"})
        self.assertEqual(manifest["stats"]["val"], {"samples": 2, "groups": 1})
        image, target, sample = GroupedBenchmarkDataset(self.root, manifest, "test")[0]
        self.assertEqual(tuple(image.shape), (4, 8, 8))
        self.assertAlmostEqual(target, np.log1p(120))


if __name__ == "__main__":
    unittest.main()
