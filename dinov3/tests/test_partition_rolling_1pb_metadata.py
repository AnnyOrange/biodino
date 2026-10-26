import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "partition_rolling_1pb_metadata.py"
SPEC = importlib.util.spec_from_file_location("partition_rolling_1pb_metadata", SCRIPT)
partitioner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(partitioner)


class PartitionRolling1PBMetadataTests(unittest.TestCase):
    def test_partitions_by_readable_root_and_keeps_same_path_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fleet = root / "fleet.json"
            fleet.write_text(json.dumps({
                "hosts": {"cpu1": {}, "cpu2": {}},
                "root_hosts": {"/a": ["cpu1", "cpu2"], "/b": ["cpu2"]},
            }))
            source = root / "metadata.csv"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("file_path", "source_id", "source_table_code", "frame_count"))
                for index in range(100):
                    writer.writerow((f"/a/img{index}.tif", index, 1, 1))
                writer.writerow(("/a/img0.tif", 100, 1, 1))
                writer.writerow(("/b/image.tif", 101, 2, 4))

            output = root / "partitions"
            summary = partitioner.partition(source, fleet, output)
            self.assertEqual(summary["rows"], 102)
            self.assertEqual(summary["unmatched"], 0)
            self.assertEqual(summary["root_rows"], {"/a": 101, "/b": 1})
            self.assertEqual(summary["root_frame_units"], {"/a": 101, "/b": 4})
            self.assertEqual(json.loads((output / "READY.json").read_text())["status"], "PASS")
            rows = {}
            for host in ("cpu1", "cpu2"):
                with (output / f"{host}.csv").open(newline="") as handle:
                    rows[host] = list(csv.DictReader(handle))
            self.assertEqual(sum(len(item) for item in rows.values()), 102)
            repeated_path_hosts = [host for host, host_rows in rows.items()
                                   for row in host_rows if row["file_path"] == "/a/img0.tif"]
            self.assertEqual(len(repeated_path_hosts), 2)
            self.assertEqual(len(set(repeated_path_hosts)), 1)
            self.assertIn("/b/image.tif", [row["file_path"] for row in rows["cpu2"]])

    def test_all_hosts_receive_each_source_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fleet = root / "fleet.json"
            fleet.write_text(json.dumps({
                "hosts": {"cpu1": {}, "cpu2": {}},
                "root_hosts": {"/a": "all", "/b": "all"},
            }))
            source = root / "metadata.csv"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("file_path", "source_id", "source_table_code", "frame_count"))
                for source_root in ("/a", "/b"):
                    for index in range(100):
                        writer.writerow((f"{source_root}/{index}.tif", index, 1, 2))
            summary = partitioner.partition(source, fleet, root / "out")
            self.assertEqual(summary["root_frame_units"], {"/a": 200, "/b": 200})
            for host in ("cpu1", "cpu2"):
                self.assertGreater(summary["host_root_rows"][host]["/a"], 0)
                self.assertGreater(summary["host_root_rows"][host]["/b"], 0)

    def test_rejects_excess_unmatched_sources_without_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fleet = root / "fleet.json"
            fleet.write_text(json.dumps({
                "hosts": {"cpu1": {}}, "root_hosts": {"/a": ["cpu1"]},
            }))
            source = root / "metadata.csv"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("file_path", "source_id", "source_table_code", "frame_count"))
                writer.writerow(("/a/one.tif", 1, 1, 1))
                writer.writerow(("/unknown/two.tif", 2, 1, 1))
            output = root / "partitions"
            with self.assertRaisesRegex(ValueError, "coverage"):
                partitioner.partition(source, fleet, output)
            self.assertFalse((output / "READY.json").exists())


if __name__ == "__main__":
    unittest.main()
