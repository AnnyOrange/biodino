import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import tifffile

from dinov3.eval.bio_frozen_eval import opencell, transloc


class OpenCellTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.benchmark_root = Path(self.tmp.name)
        self.root = self.benchmark_root / opencell.OPENCELL_RELATIVE_ROOT
        self.root.mkdir(parents=True)
        self.objects = []
        annotations = []
        for i in range(12):
            protein = f"ENSG{i:011d}"
            fov = f"FID{i:08d}"
            path = f"microscopy/raw/G{i}_{protein}/OC-FOV_G{i}_{protein}_CID{i:06d}_{fov}_proj.tif"
            full_path = self.root / path
            full_path.parent.mkdir(parents=True)
            image = np.arange(720000, dtype=np.uint16).reshape(2, 600, 600) + i
            tifffile.imwrite(full_path, image, imagej=True, metadata={"axes": "CYX"})
            self.objects.append({"path": path, "bytes": full_path.stat().st_size,
                                 "protein": protein, "fov": fov, "gene": f"G{i}", "cell_line": f"CID{i:06d}",
                                 "etag": hashlib.md5(full_path.read_bytes()).hexdigest()})
            annotations.append({"ensg_id": protein, "target_name": f"G{i}",
                                "annotation_name": "cytoplasmic" if i < 6 else "nucleoplasm", "annotation_grade": 3})
        (self.root / "official_projection_objects.json").write_text(json.dumps({"expected_fovs": 12, "objects": self.objects}))
        with (self.root / "annotations.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ensg_id", "target_name", "annotation_name", "annotation_grade"])
            writer.writeheader()
            writer.writerows(annotations)
        self.expected_patch = patch.object(opencell, "EXPECTED_PROJECTIONS", 12)
        self.expected_patch.start()

    def tearDown(self):
        self.expected_patch.stop()
        self.tmp.cleanup()

    def manifest(self):
        return opencell.build_opencell_manifest(self.benchmark_root, min_proteins_per_class=3)

    def test_deterministic_protein_disjoint_manifest_and_loader(self):
        manifest = self.manifest()
        self.assertEqual(manifest, self.manifest())
        self.assertEqual(manifest["stats"], {"train": {"samples": 8, "groups": 8}, "val": {"samples": 2, "groups": 2}, "test": {"samples": 2, "groups": 2}})
        image, target, sample_id = opencell.OpenCellDataset(self.benchmark_root, manifest, "test")[0]
        self.assertEqual(image.size, (600, 600))
        self.assertEqual(image.mode, "RGB")
        self.assertIsInstance(target, int)
        self.assertTrue(sample_id.startswith("FID"))
        self.assertTrue((np.asarray(image)[..., 2] == 0).all())

    def test_channel_semantics(self):
        image = np.array([[[0, 100], [50, 25]], [[100, 0], [25, 50]]], dtype=np.uint16)
        rgb = np.asarray(opencell.projection_rgb(image))
        self.assertEqual(rgb[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(rgb[0, 1].tolist(), [0, 255, 0])

    def test_missing_data_even_outside_task_rejected(self):
        (self.root / self.objects[0]["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing/incomplete"):
            self.manifest()

    def test_checksum_mismatch_rejected(self):
        first, second = [self.root / row["path"] for row in self.objects[:2]]
        second.write_bytes(first.read_bytes())
        with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
            self.manifest()

    def test_official_duplicate_pixels_rejected(self):
        first, second = [self.root / row["path"] for row in self.objects[:2]]
        second.write_bytes(first.read_bytes())
        self.objects[1]["etag"] = hashlib.md5(second.read_bytes()).hexdigest()
        (self.root / "official_projection_objects.json").write_text(json.dumps({"expected_fovs": 12, "objects": self.objects}))
        with self.assertRaisesRegex(ValueError, "Duplicate FOV/pixels"):
            self.manifest()

    def test_source_identity_mismatch_rejected(self):
        self.objects[0]["fov"] = "FID99999999"
        (self.root / "official_projection_objects.json").write_text(json.dumps({"expected_fovs": 12, "objects": self.objects}))
        with self.assertRaisesRegex(ValueError, "Inconsistent official"):
            self.manifest()

    def test_explicit_proxy_uses_existing_curl_transport(self):
        url = "https://czb-opencell.s3.us-west-2.amazonaws.com/example.tif"
        with patch.dict(opencell.os.environ, {"https_proxy": "http://localhost:17897"}), patch.object(opencell.shutil, "which", return_value="/usr/bin/curl"), patch.object(opencell.subprocess, "run", return_value=Mock(stdout=b"verified-source-bytes")) as invoke:
            self.assertEqual(opencell._fetch(url), b"verified-source-bytes")
        self.assertEqual(invoke.call_args.args[0][-1], url)
        self.assertIn("--fail", invoke.call_args.args[0])
        self.assertNotIn("--insecure", invoke.call_args.args[0])
        self.assertNotIn("--retry-all-errors", invoke.call_args.args[0])

    def test_bounded_selection_is_deterministic_before_split(self):
        first = opencell.build_opencell_manifest(self.benchmark_root, min_proteins_per_class=3, max_proteins_per_class=3)
        second = opencell.build_opencell_manifest(self.benchmark_root, min_proteins_per_class=3, max_proteins_per_class=3)
        self.assertEqual(first, second)
        self.assertEqual(first["expected_selected_fovs"], 6)
        self.assertEqual(first["full_official_fovs_listed"], 12)
        self.assertEqual({label: len(proteins) for label, proteins in first["selected_proteins_by_class"].items()}, {"cytoplasmic": 3, "nucleoplasm": 3})

    def test_group_leakage_rejected(self):
        manifest = deepcopy(self.manifest())
        manifest["records"][0]["group"] = next(row["group"] for row in manifest["records"] if row["split"] != manifest["records"][0]["split"])
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            opencell.OpenCellDataset(self.benchmark_root, manifest, "test")


class TranslocTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / transloc.TRANSLOC_RELATIVE_ROOT
        metadata = self.root / "analyse_256x256"
        metadata.mkdir(parents=True)
        (metadata / "single_ratio_images_train.csv").write_text("cat,image_name,ratio\n0,170721_I05_3_739_324.png,0.8563\n")
        (metadata / "single_ratio_images_val.csv").write_text("cat,image_name,ratio\n0,170721_I05_3_750_300.png,0.9000\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_ratio_and_well_fov_recovered(self):
        rows = transloc.read_ratio_rows(self.root)
        self.assertEqual(rows[0]["target"], 0.8563)
        self.assertEqual(rows[0]["group"], "170721:I05")
        self.assertEqual(rows[0]["fov"], "170721:I05:3")
        self.assertEqual(rows[0]["path"], "256x256/0/170721_I05_3_739_324.png")
        self.assertEqual(rows[0]["group"], rows[1]["group"])

    def test_renamed_image_identity_rejected(self):
        path = self.root / "analyse_256x256/single_ratio_images_train.csv"
        path.write_text(path.read_text().replace("170721_I05_3_739_324.png", "img00000000.png"))
        with self.assertRaisesRegex(ValueError, "crop identity"):
            transloc.read_ratio_rows(self.root)

    def test_duplicate_across_released_splits_rejected(self):
        path = self.root / "analyse_256x256/single_ratio_images_val.csv"
        path.write_text(path.read_text().replace("170721_I05_3_750_300.png", "170721_I05_3_739_324.png"))
        with self.assertRaisesRegex(ValueError, "Duplicate original"):
            transloc.read_ratio_rows(self.root)

    def test_nonfinite_ratio_rejected(self):
        path = self.root / "analyse_256x256/single_ratio_images_train.csv"
        path.write_text(path.read_text().replace("0.8563", "nan"))
        with self.assertRaisesRegex(ValueError, "Invalid nuclear"):
            transloc.read_ratio_rows(self.root)


if __name__ == "__main__":
    unittest.main()
