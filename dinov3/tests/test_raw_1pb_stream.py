import csv
import importlib.util
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from dinov3.data.raw_1pb_stream import Raw1PBStream


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_raw_1pb_inventory.py"
SPEC = importlib.util.spec_from_file_location("build_raw_1pb_inventory", SCRIPT)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class Raw1PBInventoryTests(unittest.TestCase):
    def test_inventory_rejects_changed_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root_a = Path(directory) / "raw_a"
            root_b = Path(directory) / "raw_b"
            root_a.mkdir()
            root_b.mkdir()
            tifffile.imwrite(root_a / "a.tif", np.ones((32, 32), dtype=np.uint16))
            index_path = Path(directory) / "inventory.sqlite"
            inventory.build([root_a], index_path, world_size=1, workers=1)
            with self.assertRaisesRegex(ValueError, "roots changed"):
                inventory.build([root_b], index_path, world_size=1, workers=1)

    def test_inventory_reconciles_meta_and_streams_cropped_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            root.mkdir()
            tiff_path = root / "movie.ome.tif"
            tifffile.imwrite(tiff_path, np.arange(2 * 3 * 64 * 80, dtype=np.uint16).reshape(2, 3, 64, 80),
                             metadata={"axes": "TCYX"})
            unindexed_path = root / "unindexed.tif"
            tifffile.imwrite(unindexed_path, np.ones((64, 80), dtype=np.uint16))
            missing_path = root / "gone.tif"
            metadata_path = Path(directory) / "metadata.csv"
            with metadata_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file_path", "source_id", "count"])
                writer.writeheader()
                writer.writerow({"file_path": str(tiff_path), "source_id": "1", "count": "2"})
                writer.writerow({"file_path": str(missing_path), "source_id": "2", "count": "1"})

            index_path = Path(directory) / "inventory.sqlite"
            report = inventory.build([root], index_path, world_size=1, workers=2,
                                     metadata_csv=metadata_path, batch_size=2)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["readable_files"], 2)
            self.assertEqual(report["readable_image_or_frame_units"], 3)
            self.assertEqual(report["metadata_paths_missing_or_unreadable"], 1)
            self.assertEqual(report["filesystem_files_absent_from_metadata"], 1)

            samples = list(itertools.islice(
                Raw1PBStream(str(index_path), None, target_channels=3,
                             seed=7, shuffle_buffer=1), 2))
            self.assertEqual(len(samples), 2)
            for sample in samples:
                self.assertEqual(tuple(sample["image"].shape), (3, 64, 80))
                self.assertTrue(sample["__key__"])


if __name__ == "__main__":
    unittest.main()
