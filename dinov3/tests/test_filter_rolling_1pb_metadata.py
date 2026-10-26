import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "filter_rolling_1pb_metadata.py"
SPEC = importlib.util.spec_from_file_location("filter_rolling_1pb_metadata", SCRIPT)
filterer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(filterer)


class FilterRollingMetadataTests(unittest.TestCase):
    def test_keeps_only_readable_root_and_publishes_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "all.csv"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("source_id", "file_path", "frame_count"))
                writer.writerow(("1", "/mnt/deepcad_nfs/a,b.tif", "3"))
                writer.writerow(("2", "/slfm/c.tif", "2"))
            output = root / "cpu2"
            report = filterer.filter_partition(source, output, "/mnt/deepcad_nfs")
            self.assertEqual(report["selected_rows"], 1)
            self.assertTrue((output / "READY.json").is_file())
            with (output / "sources.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["file_path"], "/mnt/deepcad_nfs/a,b.tif")
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
