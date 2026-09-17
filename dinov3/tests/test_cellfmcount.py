import csv
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from dinov3.eval.bio_frozen_eval.cellfmcount import (
    CELLFM_RELATIVE_ROOT, CellFMCountDataset, audit_cellfm_source, build_cellfm_manifest,
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
            build_cellfm_manifest(self.benchmark_root, duplicate_policy="error")

    def test_equal_count_duplicates_preserve_released_test_and_remove_train(self):
        (self.root / "img" / "10.tiff").write_bytes((self.root / "img" / "1.tiff").read_bytes())
        manifest = build_cellfm_manifest(self.benchmark_root)
        self.assertNotIn("1", {r["sample_id"] for r in manifest["records"]})
        self.assertEqual(next(r for r in manifest["records"] if r["sample_id"] == "10")["split"], "test")
        self.assertEqual(manifest["excluded_duplicates"][0]["retained_sample_id"], "10")

    def test_conflicting_duplicates_quarantine_all_members(self):
        (self.root / "img" / "10.tiff").write_bytes((self.root / "img" / "0.tiff").read_bytes())
        manifest = build_cellfm_manifest(self.benchmark_root)
        self.assertEqual({r["sample_id"] for r in manifest["excluded_duplicates"]}, {"0", "10"})
        self.assertEqual({r["reason"] for r in manifest["excluded_duplicates"]},
                         {"DUPLICATE_WITH_CONFLICTING_SOURCE_LABELS"})

    def test_different_encoding_same_pixels_detected(self):
        with Image.open(self.root / "img" / "1.tiff") as image:
            image.save(self.root / "img" / "10.tiff", compression="tiff_lzw")
        manifest = build_cellfm_manifest(self.benchmark_root)
        self.assertEqual(manifest["excluded_duplicates"][0]["sample_id"], "1")

    def make_source_zip(self):
        path = self.root.parent / "raw/cellfmcount.zip"
        path.parent.mkdir()
        with zipfile.ZipFile(path, "w") as archive:
            for member in self.root.rglob("*"):
                if member.is_file():
                    archive.write(member, "dataset/" + str(member.relative_to(self.root)))
        return hashlib.md5(path.read_bytes()).hexdigest()

    def test_source_conflict_audit_classifies_source_not_local_conversion(self):
        (self.root / "img/10.tiff").write_bytes((self.root / "img/0.tiff").read_bytes())
        archive_md5 = self.make_source_zip()
        with patch("dinov3.eval.bio_frozen_eval.cellfmcount.OFFICIAL_ARCHIVE_MD5", archive_md5):
            audit = audit_cellfm_source(self.benchmark_root)
        self.assertEqual(audit["source_rows"], 13)
        self.assertEqual(audit["duplicate_groups"][0]["root_cause"], "DUPLICATE_WITH_CONFLICTING_SOURCE_LABELS")
        self.assertTrue(audit["duplicate_groups"][0]["byte_identical"])
        self.assertTrue(audit["duplicate_groups"][0]["extracted_members_equal_official_archive"])
        self.assertEqual([r["source_annotation_count"] for r in audit["duplicate_groups"][0]["members"]], [0, 1])

    def test_source_audit_detects_extraction_corruption(self):
        archive_md5 = self.make_source_zip()
        (self.root / "img/10.tiff").write_bytes((self.root / "img/0.tiff").read_bytes())
        with patch("dinov3.eval.bio_frozen_eval.cellfmcount.OFFICIAL_ARCHIVE_MD5", archive_md5):
            with self.assertRaisesRegex(ValueError, "extracted member differs"):
                audit_cellfm_source(self.benchmark_root)

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
