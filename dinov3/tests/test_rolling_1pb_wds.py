import csv
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import tarfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
    def test_weighted_tar_source_mix_and_read_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            logical_a, logical_b = "/virtual/source_a", "/virtual/source_b"
            actual_a, actual_b = root / "source_a", root / "source_b"
            actual_a.mkdir()
            actual_b.mkdir()
            rows = []
            for logical, actual in ((logical_a, actual_a), (logical_b, actual_b)):
                for index in range(4):
                    tifffile.imwrite(actual / f"{index}.tif",
                                     np.full((48, 64), index + 1, dtype=np.uint16))
                    rows.append((f"{logical}/{index}.tif", 1, 0,
                                 f"{logical}-{index}", "test"))
            batch_id = rolling.produce_batch(
                root / "unused.sqlite", root / "queue", batch_samples=20,
                max_live_batches=2, workers=1, samples_per_shard=10,
                file_block=2, seed=7, inventory_rows=rows,
                root_weights={logical_a: 7, logical_b: 3},
                root_aliases={logical_a: str(actual_a), logical_b: str(actual_b)})
            ready = root / "queue" / "ready" / batch_id
            manifest = json.loads((ready / "READY.json").read_text())
            self.assertEqual(manifest["root_sample_counts"],
                             {logical_a: 14, logical_b: 6})
            for tar_path in (ready / "shards").glob("*.tar"):
                with tarfile.open(tar_path) as archive:
                    metas = [json.load(archive.extractfile(member))
                             for member in archive.getmembers()
                             if member.name.endswith(".meta.json")]
                self.assertEqual(sum(meta["source_root"] == logical_a for meta in metas), 7)
                self.assertEqual(sum(meta["source_root"] == logical_b for meta in metas), 3)
                self.assertTrue(all(meta["read_path"].startswith(str(root)) for meta in metas))

    def test_unavailable_weighted_source_never_publishes_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = root / "good"
            good.mkdir()
            tifffile.imwrite(good / "image.tif", np.ones((48, 64), dtype=np.uint16))
            rows = [("/good/image.tif", 1, 0, "good", "test"),
                    ("/missing/image.tif", 1, 0, "missing", "test")]
            with self.assertRaisesRegex(RuntimeError, "cannot fill source quota"):
                rolling.produce_batch(
                    root / "unused.sqlite", root / "queue", batch_samples=10,
                    max_live_batches=1, workers=1, samples_per_shard=10,
                    file_block=2, seed=11, inventory_rows=rows,
                    root_weights={"/good": 1, "/missing": 1},
                    root_aliases={"/good": str(good),
                                  "/missing": str(root / "missing")})
            self.assertFalse(list((root / "queue" / "ready").glob("batch-*")))

    def test_producer_claim_skips_existing_batch_when_state_is_stale(self):
        with tempfile.TemporaryDirectory() as temp:
            queue_root = Path(temp) / "queue"
            (queue_root / "completed" / "batch-000010").mkdir(parents=True)
            (queue_root / "queue_state.json").write_text('{"next_batch": 10}')
            with mock.patch.object(rolling, "_assign_files", side_effect=RuntimeError("stop after claim")):
                with self.assertRaisesRegex(RuntimeError, "stop after claim"):
                    rolling.produce_batch(
                        Path(temp) / "unused.sqlite", queue_root, batch_samples=1,
                        max_live_batches=2, workers=1, samples_per_shard=1,
                        file_block=1, seed=1, inventory_rows=[])
            self.assertTrue((queue_root / ".building" / "batch-000011" / "BUILDING.json").is_file())
            self.assertEqual(json.loads((queue_root / "queue_state.json").read_text()),
                             {"next_batch": 12})

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
            self.assertEqual(len(list((completed / "shards").glob("*.tar"))), 4)
            self.assertTrue(json.loads((completed / "CONSUMED.json").read_text())["retained_shards"])
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
                "train": {"dataset_path": "old", "output_dir": "old", "num_workers": 4,
                          "batch_size_per_gpu": 32},
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
                total_updates=40016, batch_updates=488)
            self.assertTrue(segment["train"]["finite_stream_fresh_segment"])
            self.assertEqual(segment["train"]["num_workers"], 1)
            self.assertEqual(segment["train"]["OFFICIAL_EPOCH_LENGTH"], 488)
            self.assertEqual(segment["optim"]["epochs"], 82)
            self.assertEqual(segment["optim"]["gradient_accumulation_steps"], 4)
            self.assertEqual(segment["student"]["resume_from_teacher_chkpt"], "")

            segment_4gpu = trainer.segment_config(
                config_path, root / "run-4gpu", batch, baseline,
                total_updates=40016, batch_updates=488,
                world_size=4, effective_global_batch_size=1024)
            self.assertEqual(segment_4gpu["optim"]["gradient_accumulation_steps"], 8)
            self.assertEqual(488 * 4 * 32 * 8, 499712)

            segment_64 = trainer.segment_config(
                config_path, root / "run-64", batch, baseline,
                total_updates=40016, batch_updates=488,
                batch_size_per_gpu=64)
            self.assertEqual(segment_64["train"]["batch_size_per_gpu"], 64)
            self.assertEqual(segment_64["optim"]["gradient_accumulation_steps"], 2)
            self.assertEqual(488 * 8 * 64 * 2, 499712)

            command = trainer.training_command(
                SimpleNamespace(python="python", world_size=4, master_port=31971),
                root / "segment.yaml", root / "run", start_update=488)
            self.assertEqual(command[-4:], [
                "--output-dir", str(root / "run"),
                "--expected-start-iteration", "488"])

    def test_pending_batch_recovers_an_existing_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            queue_root = Path(temp)
            active = queue_root / "active" / "batch-000001"
            active.mkdir(parents=True)
            (active / "READY.json").write_text('{"status":"PASS"}')
            (active / "CLAIMED.json").write_text('{"status":"ACTIVE"}')
            recovered, was_active = trainer.pending_batch(queue_root)
            self.assertEqual(recovered, active)
            self.assertTrue(was_active)
            self.assertTrue(active.exists())

    def test_failed_training_retries_same_batch_before_reclaiming(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ready = root / "queue" / "ready" / "batch-000001"
            shards = ready / "shards"
            shards.mkdir(parents=True)
            for index in range(8):
                (shards / f"shard-{index:02d}.tar").touch()
            (ready / "READY.json").write_text(json.dumps({
                "status": "PASS", "samples": 16,
                "samples_per_shard": 2, "shard_count": 8,
            }))
            base_config = root / "base.yaml"
            base_config.write_text(yaml.safe_dump({
                "train": {"batch_size_per_gpu": 4}, "optim": {},
                "checkpointing": {}, "evaluation": {}, "student": {},
            }))
            teacher = root / "teacher.pth"
            teacher.write_bytes(b"teacher")
            initial = root / "0" / "checkpoint.pth"
            initial.parent.mkdir()
            initial.write_bytes(b"initial")
            run_dir = root / "run"
            args = SimpleNamespace(
                queue_root=root / "queue", run_dir=run_dir, base_config=base_config,
                baseline_checkpoint=teacher, initial_full_checkpoint=initial,
                world_size=4, cuda_visible_devices="0,1,2,3",
                batch_updates=1, total_updates=2, effective_global_batch_size=16,
                batch_size_per_gpu=None,
                allow_segment_reconfigure=False,
                max_batches=1, poll_seconds=0, omp_threads=1, master_port=31971,
                min_checkpoint_bytes=1, train_retries=1, retry_seconds=0,
                repo=root, python="python",
            )
            attempts = []

            def fake_training(command, **kwargs):
                attempts.append(command)
                if len(attempts) == 2:
                    checkpoint = run_dir / "ckpt" / "1" / "checkpoint.pth"
                    checkpoint.parent.mkdir()
                    checkpoint.write_bytes(b"complete")
                return SimpleNamespace(returncode=1 if len(attempts) == 1 else 0)

            with mock.patch.object(trainer.subprocess, "run", side_effect=fake_training):
                self.assertEqual(trainer.run(args), 0)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0], attempts[1])
            completed = root / "queue" / "completed" / "batch-000001"
            self.assertTrue((completed / "CONSUMED.json").is_file())
            self.assertEqual(len(list((completed / "shards").glob("*.tar"))), 8)
            assignment = json.loads((completed / "finite_assignment.json").read_text())
            self.assertTrue(all(Path(path).is_file() for rank in assignment["rank_shards"]
                                for path in rank))


if __name__ == "__main__":
    unittest.main()
