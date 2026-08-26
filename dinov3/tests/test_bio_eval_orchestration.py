from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import numpy as np
from PIL import Image

from dinov3.eval import bio_benchmark
from dinov3.eval.bio_detection.center_probe import InstanceCenterDataset, PatchFeatureModel
from dinov3.eval.bio_frozen_eval import run_classification
from dinov3.eval.bio_frozen_eval.datasets import (
    CoNICCellCountRegressionDataset,
    LIVECellCountRegressionDataset,
)
from dinov3.eval.bio_frozen_eval.retrieval_clustering import (
    ManifestImageDataset,
    RxRx1ZipDataset,
    query_gallery_metrics,
)


def make_args(tmp_path: Path, **overrides):
    values = dict(
        output_dir=str(tmp_path / "eval"),
        benchmark_root=str(tmp_path / "benchmark"),
        train_config=str(tmp_path / "config.yaml"),
        checkpoints_dir=str(tmp_path / "ckpt"),
        tasks=["segmentation"],
        segmentation_datasets=["bbbc038", "tissuenet"],
        segmentation_datasets_per_job=0,
        segmentation_multichannel=True,
        segmentation_channel_policy="native",
        segmentation_channel_tta_samples=8,
        segmentation_channel_policy_seed=0,
        segmentation_protocol="best",
        run_name="bio_eval",
        layer_preset="last1",
        seg_feature_batch_size=8,
        seg_feature_num_workers=1,
        seg_probe_epochs=1,
        seg_probe_batch_size=8,
        seg_probe_num_workers=1,
        smoke=False,
        smoke_max_samples=8,
        max_samples_per_split=0,
        classification_datasets=[],
        regression_datasets=[],
        retrieval_datasets=[],
        rxrx1_full=False,
        detection_datasets=[],
        frozen_n_last_blocks=1,
        autocast_dtype="bf16",
        frozen_batch_size=32,
        frozen_datasets_per_job=1,
        frozen_split_protocol="current",
        num_workers=1,
        train_fraction=0.8,
        seed=0,
        frozen_channel_policy="auto",
        frozen_channel_tta_samples=8,
        frozen_channel_policy_seed=0,
        classification_resolution_protocol="best",
        classification_image_size=224,
        classification_resize_size=0,
        regression_resolution_protocol="best",
        regression_image_size=224,
        regression_resize_size=0,
        det_epochs=1,
        det_batch_size=2,
        detection_channel_policy="auto",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class BioEvalOrchestrationTests(unittest.TestCase):
    def test_scheduler_enforces_jobs_per_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [
                bio_benchmark.Job(
                    task="classification",
                    dataset=f"dataset-{index}",
                    ckpt_id=1,
                    gpu="0",
                    cmd=["fake-eval"],
                    output_dir=root / str(index),
                )
                for index in range(8)
            ]
            args = SimpleNamespace(
                tasks=["classification"],
                jobs_per_gpu=2,
                max_concurrent_jobs=8,
                max_cpu_jobs=8,
                dry_run=False,
            )
            lock = threading.Lock()
            active = 0
            peak = 0

            def fake_run(_cmd, **_kwargs):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return SimpleNamespace(returncode=0)

            with mock.patch.object(bio_benchmark.subprocess, "run", side_effect=fake_run):
                rows = bio_benchmark._run_jobs(args, jobs, ["0"])

        self.assertEqual(peak, 2)
        self.assertEqual(len(rows["classification"]), len(jobs))

    def test_scheduler_steals_idle_gpus_instead_of_staying_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [
                bio_benchmark.Job(
                    task="classification",
                    dataset=f"dataset-{index}",
                    ckpt_id=1,
                    gpu="0",
                    cmd=["fake-eval"],
                    output_dir=root / str(index),
                )
                for index in range(4)
            ]
            args = SimpleNamespace(
                tasks=["classification"],
                jobs_per_gpu=1,
                max_concurrent_jobs=2,
                max_cpu_jobs=2,
                dry_run=False,
            )
            seen = set()
            lock = threading.Lock()

            def fake_run(_cmd, **kwargs):
                env = kwargs.get("env") or {}
                with lock:
                    seen.add(env.get("CUDA_VISIBLE_DEVICES"))
                time.sleep(0.02)
                return SimpleNamespace(returncode=0)

            with mock.patch.object(bio_benchmark.subprocess, "run", side_effect=fake_run):
                rows = bio_benchmark._run_jobs(args, jobs, ["0", "1"])

        self.assertEqual(seen, {"0", "1"})
        self.assertEqual(len(rows["classification"]), len(jobs))

    def test_added_id_datasets_are_in_default_task_lists(self):
        self.assertNotIn("allen-cell-volume", bio_benchmark.DEFAULT_REGRESSION_DATASETS)
        self.assertIn("conic-cell-count", bio_benchmark.DEFAULT_REGRESSION_DATASETS)
        self.assertIn("livecell-cell-count", bio_benchmark.DEFAULT_REGRESSION_DATASETS)
        self.assertIn("hpa-subcellular", bio_benchmark.DEFAULT_RETRIEVAL_DATASETS)
        self.assertIn("rxrx1-cross", bio_benchmark.DEFAULT_RETRIEVAL_DATASETS)
        self.assertIn("bbbc038", bio_benchmark.DEFAULT_DETECTION_DATASETS)
        self.assertIn("conic", bio_benchmark.DEFAULT_DETECTION_DATASETS)

    def test_new_id_suite_builds_all_dataset_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(
                tmp_path,
                tasks=["regression", "retrieval", "detection"],
                regression_datasets=list(bio_benchmark.DEFAULT_REGRESSION_DATASETS),
                retrieval_datasets=list(bio_benchmark.DEFAULT_RETRIEVAL_DATASETS),
                detection_datasets=list(bio_benchmark.DEFAULT_DETECTION_DATASETS),
                segmentation_datasets=[],
                segmentation_multichannel=False,
            )
            jobs = bio_benchmark.build_jobs(
                args,
                {1: tmp_path / "ckpt" / "1" / "checkpoint.pth"},
                [1],
                ["0", "1"],
            )

        jobs_by_task = {
            task: [job for job in jobs if job.task == task]
            for task in ("regression", "retrieval", "detection")
        }
        self.assertEqual(
            [job.dataset for job in jobs_by_task["regression"]],
            bio_benchmark.DEFAULT_REGRESSION_DATASETS,
        )
        self.assertEqual(
            [job.dataset for job in jobs_by_task["retrieval"]],
            bio_benchmark.DEFAULT_RETRIEVAL_DATASETS,
        )
        self.assertEqual(
            [job.dataset for job in jobs_by_task["detection"]],
            bio_benchmark.DEFAULT_DETECTION_DATASETS,
        )
        self.assertTrue(all("--split-protocol" in job.cmd for job in jobs_by_task["regression"]))
        self.assertTrue(all("--rxrx1-full" not in job.cmd for job in jobs_by_task["retrieval"]))
        self.assertNotIn("allen-cell-volume", [job.dataset for job in jobs_by_task["regression"]])

    def test_rxrx1_full_uses_a_separate_output_and_child_protocol_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(
                tmp_path,
                tasks=["retrieval"],
                retrieval_datasets=["rxrx1-cross"],
                rxrx1_full=True,
                segmentation_datasets=[],
                segmentation_multichannel=False,
            )
            jobs = bio_benchmark.build_jobs(
                args,
                {1: tmp_path / "ckpt" / "1" / "checkpoint.pth"},
                [1],
                ["0"],
            )

        self.assertEqual(len(jobs), 1)
        self.assertIn("--rxrx1-full", jobs[0].cmd)
        self.assertEqual(jobs[0].output_dir.parent.name, "rxrx1-cross_full")

    def test_new_id_smoke_caps_are_forwarded_to_each_task_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(
                tmp_path,
                tasks=["regression", "retrieval", "detection"],
                regression_datasets=["conic-cell-count"],
                retrieval_datasets=["hpa-subcellular", "rxrx1-cross"],
                detection_datasets=["bbbc038", "conic"],
                segmentation_datasets=[],
                segmentation_multichannel=False,
                smoke=True,
                smoke_max_samples=8,
            )
            jobs = bio_benchmark.build_jobs(
                args,
                {1: tmp_path / "ckpt" / "1" / "checkpoint.pth"},
                [1],
                ["0"],
            )

        regression = next(job for job in jobs if job.task == "regression")
        self.assertEqual(regression.cmd[regression.cmd.index("--max-samples") + 1], "8")
        for job in (job for job in jobs if job.task == "retrieval"):
            self.assertEqual(job.cmd[job.cmd.index("--max-samples") + 1], "64")
        for job in (job for job in jobs if job.task == "detection"):
            cap_at = job.cmd.index("--max-samples-per-split")
            self.assertEqual(job.cmd[cap_at + 1], "8")

    def test_hpa_and_rxrx1_manifest_loaders_accept_new_protocol_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hpa_root = root / "hpa"
            (hpa_root / "images").mkdir(parents=True)
            Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8)).save(
                hpa_root / "images/example.png"
            )
            hpa_manifest = root / "hpa.csv"
            hpa_manifest.write_text(
                "image_path,label,role,robust_ge10\n"
                "images/example.png,7,gallery,1\n"
                "images/example.png,7,query,1\n"
            )
            hpa = ManifestImageDataset(hpa_root, hpa_manifest, role="query")
            image, label, _path = hpa[0]
            self.assertEqual(image.size, (5, 4))
            self.assertEqual(label, 7)

            archive = root / "rxrx1.zip"
            channel_paths = [f"rxrx1/c{channel}.png" for channel in range(1, 7)]
            with zipfile.ZipFile(archive, "w") as handle:
                for channel, path in enumerate(channel_paths, 1):
                    buffer = io.BytesIO()
                    Image.fromarray(np.full((4, 5), channel * 10, dtype=np.uint8)).save(
                        buffer,
                        format="PNG",
                    )
                    handle.writestr(path, buffer.getvalue())
            rxrx1_manifest = root / "rxrx1.csv"
            header = "site_id,role,sirna_id,cell_type,experiment,plate,c1,c2,c3,c4,c5,c6\n"
            channel_csv = ",".join(channel_paths)
            rxrx1_manifest.write_text(
                header
                + f"gallery-site,gallery,11,U2OS,U2OS-01,1,{channel_csv}\n"
                + f"query-site,query,11,U2OS,U2OS-02,1,{channel_csv}\n"
            )
            rxrx1 = RxRx1ZipDataset(archive, rxrx1_manifest, role="query")
            image, label, site_id = rxrx1[0]
            self.assertEqual(tuple(image.shape), (6, 4, 5))
            self.assertEqual(label, 11)
            self.assertEqual(site_id, "query-site")

    def test_count_regression_loaders_preserve_the_counted_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conic_root = root / "conic"
            (conic_root / "data").mkdir(parents=True)
            conic_image = np.zeros((1, 256, 256, 3), dtype=np.uint8)
            conic_image[:, 16:240, 16:240] = 127
            np.save(conic_root / "data/images.npy", conic_image)
            (conic_root / "conic_cell_count.csv").write_text(
                "split,image_index,cell_count\ntrain,0,17\n"
            )
            image, target, _path = CoNICCellCountRegressionDataset(conic_root, "train")[0]
            self.assertEqual(image.size, (224, 224))
            self.assertEqual(target, 17.0)

            live_root = root / "livecell"
            (live_root / "data/images").mkdir(parents=True)
            Image.fromarray(np.full((4, 6, 3), 127, dtype=np.uint8)).save(
                live_root / "data/images/example.png"
            )
            (live_root / "livecell_cell_count.csv").write_text(
                "split,image_path,cell_count\ntrain,images/example.png,23\n"
            )
            image, target, _path = LIVECellCountRegressionDataset(live_root, "train")[0]
            self.assertEqual(image.size, (6, 6))
            self.assertEqual(target, 23.0)

        self.assertEqual(
            run_classification.resolve_dataset_resize_size("conic-cell-count", 224, 0),
            224,
        )
        self.assertEqual(
            run_classification.resolve_dataset_resize_size("livecell-cell-count", 224, 0),
            224,
        )
        self.assertEqual(run_classification.resolve_dataset_resize_size("bloodmnist", 224, 0), 256)
        with self.assertRaises(ValueError):
            run_classification.resolve_dataset_resize_size("conic-cell-count", 224, 256)

    def test_query_gallery_metrics_uses_disjoint_gallery(self):
        features = np.eye(2, dtype=np.float32)
        labels = np.asarray([0, 1])
        metrics = query_gallery_metrics(features, labels, features, labels, metric_device="cpu")
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["map_at_1"], 1.0)

    def test_instance_masks_convert_to_patch_center_labels(self):
        class FakeInstanceDataset:
            def __len__(self):
                return 1

            def __getitem__(self, _index):
                image = torch.zeros(3, 32, 32)
                instances = torch.zeros(32, 32, dtype=torch.long)
                instances[2:6, 2:6] = 1
                instances[20:24, 20:24] = 2
                return image, instances.gt(0).long(), instances

        dataset = InstanceCenterDataset(FakeInstanceDataset(), image_size=32, patch_size=16)
        _image, labels = dataset[0]
        self.assertEqual(tuple(labels.shape), (4,))
        self.assertEqual(int(labels.sum()), 2)

    def test_segmentation_native_policy_is_only_used_for_multichannel_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(tmp_path)
            jobs = bio_benchmark.build_jobs(
                args,
                {1: tmp_path / "ckpt" / "1" / "checkpoint.pth"},
                [1],
                ["0", "1"],
            )
        self.assertEqual(len(jobs), 2)
        rgb_job = next(job for job in jobs if job.dataset == "bbbc038")
        mc_job = next(job for job in jobs if job.dataset == "tissuenet")
        self.assertEqual(rgb_job.cmd[rgb_job.cmd.index("--channel-policy") + 1], "auto")
        self.assertNotIn("--multichannel", rgb_job.cmd)
        self.assertEqual(mc_job.cmd[mc_job.cmd.index("--channel-policy") + 1], "native")
        self.assertIn("--multichannel", mc_job.cmd)

    def test_segmentation_datasets_can_run_as_independent_gpu_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(
                tmp_path,
                segmentation_datasets=["bbbc038", "conic", "monuseg"],
                segmentation_multichannel=False,
                segmentation_channel_policy="auto",
                segmentation_datasets_per_job=1,
            )
            jobs = bio_benchmark.build_jobs(
                args,
                {1: tmp_path / "ckpt" / "1" / "checkpoint.pth"},
                [1],
                ["0", "1"],
            )
        self.assertEqual([job.dataset for job in jobs], ["bbbc038", "conic", "monuseg"])
        self.assertEqual([job.gpu for job in jobs], ["0", "1", "0"])
        for job in jobs:
            datasets_at = job.cmd.index("--datasets")
            checkpoints_at = job.cmd.index("--checkpoints-dir")
            self.assertEqual(job.cmd[datasets_at + 1 : checkpoints_at], [job.dataset])

    def test_run_job_bounds_blas_threads_and_reuses_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_out = tmp_path / "env.json"
            code = (
                "import json, os, pathlib; "
                f"pathlib.Path({str(env_out)!r}).write_text(json.dumps({{k: os.environ.get(k) for k in "
                "['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'] }))"
            )
            job = bio_benchmark.Job(
                "classification", "dummy", 1, "0", [sys.executable, "-c", code], tmp_path / "job"
            )
            _job, returncode, _log = bio_benchmark._run_job(job, dry_run=False)
            self.assertEqual(returncode, 0)
            self.assertEqual(set(json.loads(env_out.read_text()).values()), {"1"})

            (job.output_dir / "last_result.json").write_text('{"accuracy": 1.0}')
            cached = bio_benchmark.Job(
                "classification", "dummy", 1, "0", [sys.executable, "-c", "raise SystemExit(9)"], job.output_dir
            )
            _job, returncode, status = bio_benchmark._run_job(cached, dry_run=False)
            self.assertEqual(returncode, 0)
            self.assertEqual(status, "cached-success")

    def test_channelvit_detection_collapses_channel_tokens_to_spatial_grid(self):
        class FakeBackbone:
            def get_intermediate_layers(self, images, **kwargs):
                self.assertTrue(kwargs["reshape"])
                return (torch.ones(images.shape[0], 5, 14, 14),)

        # Bind the assertion method into the small fake without making it an nn.Module.
        fake = FakeBackbone()
        fake.assertTrue = self.assertTrue
        model = PatchFeatureModel(fake, torch.bfloat16)
        with mock.patch("torch.autocast", return_value=nullcontext()):
            features = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(features.shape), (2, 196, 5))

    def test_frozen_datasets_can_be_sharded_per_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = make_args(
                tmp_path,
                tasks=["classification"],
                classification_datasets=["bloodmnist", "pathmnist", "tissuemnist"],
                segmentation_datasets=[],
                segmentation_multichannel=False,
                frozen_datasets_per_job=2,
            )
            jobs = bio_benchmark.build_jobs(
                args,
                {1: tmp_path / "ckpt" / "1" / "checkpoint.pth"},
                [1],
                ["0", "1"],
            )
        self.assertEqual(len(jobs), 2)
        self.assertIn("bloodmnist+pathmnist", jobs[0].dataset)
        datasets_at = jobs[0].cmd.index("--datasets")
        output_at = jobs[0].cmd.index("--output-dir")
        self.assertEqual(jobs[0].cmd[datasets_at + 1 : output_at], ["bloodmnist", "pathmnist"])

    def test_failed_frozen_dataset_returns_nonzero_without_success_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code = run_classification.main(
                [
                    "--checkpoint", str(Path(tmp) / "missing.pth"),
                    "--train-config", str(Path(tmp) / "missing.yaml"),
                    "--datasets", "chammi-cp-task4",
                    "--output-dir", str(out),
                    "--model-name", "test-model",
                ]
            )
            self.assertEqual(code, 1)
            self.assertTrue((out / "failed_result.json").is_file())
            self.assertFalse((out / "last_result.json").exists())


if __name__ == "__main__":
    unittest.main()
