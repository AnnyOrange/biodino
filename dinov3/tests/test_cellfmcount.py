import csv
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from dinov3.eval.bio_frozen_eval.cellfmcount import (
    CELLFM_RELATIVE_ROOT, CellFMCountDataset, build_cellfm_manifest,
)


def make_data(tmp_path):
    root = tmp_path / CELLFM_RELATIVE_ROOT
    (root / "img").mkdir(parents=True)
    (root / "ground_truth").mkdir()
    rows = []
    for i in range(13):
        staining = "Cy3" if i == 12 else "DAPI"
        rows.append({"id": str(i), "cell_count": 0 if i == 0 else 1,
                     "cell_type": "AHPC", "staining": staining, "objective": "20x",
                     "markers": "DAPI" if staining == "DAPI" else "PI",
                     "set": "trainval" if i < 10 else "test"})
        Image.fromarray(np.arange(48, dtype=np.uint16).reshape(6, 8) + i).save(root / "img" / f"{i}.tiff")
        with (root / "ground_truth" / f"{i}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["X", "Y"])
            if i:
                writer.writerow([2.5, 3])
    with (root / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


class CellFMCountTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.benchmark_root = Path(self.temporary_directory.name)
        self.root = make_data(self.benchmark_root)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_dapi_only_fixed_test_and_deterministic_validation(self):
        manifest = build_cellfm_manifest(self.benchmark_root)
        self.assertEqual(manifest, build_cellfm_manifest(self.benchmark_root))
        self.assertEqual(manifest["stats"], {"train": {"samples": 9, "groups": 9},
                                             "val": {"samples": 1, "groups": 1},
                                             "test": {"samples": 2, "groups": 2}})
        self.assertEqual({r["sample_id"] for r in manifest["records"] if r["split"] == "test"}, {"10", "11"})
        self.assertNotIn("12", {r["sample_id"] for r in manifest["records"]})
        image, target, sample_id = CellFMCountDataset(self.benchmark_root, manifest, "test")[0]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (8, 8))
        self.assertEqual((target, sample_id), (1.0, "10"))
        self.assertIn("biological grouping unverified", manifest["grouping"])

    def test_no_dapi_rejected(self):
        metadata = self.root / "metadata.csv"
        metadata.write_text(metadata.read_text().replace("DAPI", "Cy3"))
        with self.assertRaisesRegex(ValueError, "DAPI trainval"):
            build_cellfm_manifest(self.benchmark_root)

    def test_count_mismatch_rejected(self):
        (self.root / "ground_truth" / "1.csv").write_text("X,Y\n")
        with self.assertRaisesRegex(ValueError, "Count mismatch"):
            build_cellfm_manifest(self.benchmark_root)

    def test_missing_image_rejected(self):
        (self.root / "img" / "1.tiff").unlink()
        with self.assertRaisesRegex(ValueError, "Missing image"):
            build_cellfm_manifest(self.benchmark_root)

    def test_duplicate_image_content_rejected(self):
        (self.root / "img" / "10.tiff").write_bytes((self.root / "img" / "1.tiff").read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate image content"):
            build_cellfm_manifest(self.benchmark_root)

    def test_group_leakage_rejected(self):
        manifest = deepcopy(build_cellfm_manifest(self.benchmark_root))
        test = next(r for r in manifest["records"] if r["split"] == "test")
        train = next(r for r in manifest["records"] if r["split"] == "train")
        test["group"] = train["group"]
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            CellFMCountDataset(self.benchmark_root, manifest, "test")

    def test_finite_edge_click_preserves_source_count(self):
        (self.root / "ground_truth" / "1.csv").write_text("X,Y\n2,-2\n")
        manifest = build_cellfm_manifest(self.benchmark_root)
        record = next(r for r in manifest["records"] if r["sample_id"] == "1")
        self.assertEqual((record["target"], record["out_of_bounds_coordinates"]), (1, 1))
        self.assertEqual(manifest["out_of_bounds_coordinates"], 1)

    def test_nonfinite_coordinate_rejected(self):
        (self.root / "ground_truth" / "1.csv").write_text("X,Y\nnan,2\n")
        with self.assertRaisesRegex(ValueError, "Nonfinite coordinate"):
            build_cellfm_manifest(self.benchmark_root)


if __name__ == "__main__":
    unittest.main()
