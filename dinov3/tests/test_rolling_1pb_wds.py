import csv
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import yaml


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rolling_1pb_wds.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("rolling_1pb_wds", SCRIPT)
rolling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rolling
SPEC.loader.exec_module(rolling)

INVENTORY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_raw_1pb_inventory.py"
INVENTORY_SPEC = importlib.util.spec_from_file_location("build_raw_1pb_inventory", INVENTORY_SCRIPT)
inventory = importlib.util.module_from_spec(INVENTORY_SPEC)
sys.modules[INVENTORY_SPEC.name] = inventory
INVENTORY_SPEC.loader.exec_module(inventory)

TRAIN_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_rolling_1pb_train.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("run_rolling_1pb_train", TRAIN_SCRIPT)
trainer = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = trainer
TRAIN_SPEC.loader.exec_module(trainer)


class Rolling1PBQueueTests(unittest.TestCase):
    def test_metadata_candidates_are_available_before_inventory_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = root / "metadata.csv"
            with metadata.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("file_path", "source_id", "source_table_code", "frame_count"))
                writer.writerow(("/data/a.tif", "a", "1", "1"))
                writer.writerow(("/data/b.tif", "b", "2", "12"))
            rows = rolling._metadata_rows(metadata)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][1:], (1, 0, "a", "1"))
            self.assertEqual(rows[1][1:], (12, 0, "b", "2"))

            index = root / "inventory.sqlite"
            connection = inventory.create_database(index, world_size=1)
            connection.execute("INSERT OR REPLACE INTO inventory_meta VALUES ('status', 'SCANNING')")
            connection.commit()
            connection.close()
            self.assertFalse(rolling._inventory_is_ready(index))

            connection = sqlite3.connect(index)
            connection.execute("UPDATE inventory_meta SET value='PASS' WHERE key='status'")
            connection.commit()
            connection.close()
            self.assertTrue(rolling._inventory_is_ready(index))

    def test_materialize_validate_claim_and_reap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            metadata = root / "metadata.csv"
            paths = []
            with metadata.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("file_path", "source_id", "source_table_code", "frame_count"))
                for index in range(8):
                    path = source / f"image_{index}.tif"
                    tifffile.imwrite(path, np.full((48, 64), index + 1, dtype=np.uint16))
                    paths.append(path)
                    writer.writerow((str(path), index + 1, 1, 1))
            index_path = root / "inventory.sqlite"
            inventory.build([source], index_path, world_size=1, workers=1,
                            metadata_csv=metadata)
            queue_root = root / "queue"
            batch_id = rolling.produce_batch(
                index_path, queue_root, batch_samples=16, max_live_batches=2,
                workers=2, samples_per_shard=4, file_block=2, seed=9)
            self.assertEqual(batch_id, "batch-000001")
            ready = queue_root / "ready" / batch_id
            manifest = json.loads((ready / "READY.json").read_text())
            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual(manifest["samples"], 16)
            self.assertEqual(manifest["shard_count"], 4)
            from dinov3.data.wds_pipeline import WdsConfig, build_packed_robust_wds_pipeline
            shard_paths = sorted(str(path.resolve()) for path in (ready / "shards").glob("*.tar"))
            finite_assignment = root / "finite_assignment.json"
            finite_assignment.write_text(json.dumps({
                "status": "PASS", "rank_shards": [shard_paths],
                "rank_sample_counts": [16], "target_per_rank": 16,
            }))
            prior_assignment = os.environ.get("DQ_FINITE_SHARD_ASSIGNMENT")
            os.environ["DQ_FINITE_SHARD_ASSIGNMENT"] = str(finite_assignment)
            try:
                stream = build_packed_robust_wds_pipeline(
                    WdsConfig(shard_urls=shard_paths, target_channels=3,
                              finite_once=True, shuffle_buffer=4),
                )
                consumed = list(stream)
            finally:
                if prior_assignment is None:
                    os.environ.pop("DQ_FINITE_SHARD_ASSIGNMENT", None)
                else:
                    os.environ["DQ_FINITE_SHARD_ASSIGNMENT"] = prior_assignment
            self.assertEqual(len(consumed), 16)
            self.assertEqual(rolling.produce_batch(
                index_path, queue_root, batch_samples=16, max_live_batches=1,
                workers=2, samples_per_shard=4, file_block=2, seed=10), "FULL")

            active = rolling.claim(queue_root)
            self.assertEqual(active.name, batch_id)
            rolling.release(queue_root, batch_id)
            active = rolling.claim(queue_root)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"valid")
            completed = rolling.complete(queue_root, batch_id, checkpoint, 243,
                                         min_checkpoint_bytes=1)
            self.assertTrue((completed / "CONSUMED.json").is_file())
            self.assertFalse((completed / "shards").exists())
            state = rolling.status(queue_root)
            self.assertEqual(state["live_limit_count"], 0)
            self.assertEqual(len(state["groups"]["completed"]), 1)

    def test_finite_rank_assignment_and_segment_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / "batch-000001"
            shards = batch / "shards"
            shards.mkdir(parents=True)
            for index in range(8):
                (shards / f"shard-{index:02d}.tar").touch()
            (batch / "READY.json").write_text(json.dumps({
                "status": "PASS", "samples": 16, "samples_per_shard": 2,
                "shard_count": 8,
            }))
            assignment_path = trainer.build_assignment(batch, world_size=4)
            assignment = json.loads(assignment_path.read_text())
            self.assertEqual(assignment["rank_sample_counts"], [4, 4, 4, 4])
            self.assertEqual([len(group) for group in assignment["rank_shards"]], [2, 2, 2, 2])

            config_path = root / "base.yaml"
            config_path.write_text(yaml.safe_dump({
                "train": {"dataset_path": "old", "output_dir": "old", "num_workers": 4},
                "optim": {"epochs": 1},
                "checkpointing": {}, "evaluation": {}, "student": {},
            }))
            baseline = root / "baseline_teacher.pth"
            full_checkpoint = root / "487" / "checkpoint.pth"
            full_checkpoint.parent.mkdir()
            full_checkpoint.write_bytes(b"model_and_optimizer")
            trainer.seed_initial_checkpoint(root / "run", full_checkpoint)
            seeded = root / "run" / "ckpt" / "487" / "checkpoint.pth"
            self.assertEqual(seeded.stat().st_ino, full_checkpoint.stat().st_ino)
            self.assertEqual(trainer.latest_checkpoint(root / "run")[0], 487)

            segment = trainer.segment_config(
                config_path, root / "run", batch, baseline,
                total_updates=40016, batch_updates=244)
            self.assertTrue(segment["train"]["finite_stream_fresh_segment"])
            self.assertEqual(segment["train"]["num_workers"], 1)
            self.assertEqual(segment["train"]["OFFICIAL_EPOCH_LENGTH"], 244)
            self.assertEqual(segment["optim"]["epochs"], 164)
            self.assertEqual(segment["student"]["resume_from_teacher_chkpt"], "")


if __name__ == "__main__":
    unittest.main()
