from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from dinov3.eval import bio_benchmark
from dinov3.eval.bio_detection.center_probe import PatchFeatureModel
from dinov3.eval.bio_frozen_eval import run_classification


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
        detection_datasets=[],
        frozen_n_last_blocks=1,
        autocast_dtype="bf16",
        frozen_batch_size=32,
        frozen_datasets_per_job=1,
        num_workers=1,
        train_fraction=0.8,
        seed=0,
        frozen_channel_policy="auto",
        frozen_channel_tta_samples=8,
        frozen_channel_policy_seed=0,
        classification_resolution_protocol="best",
        classification_image_size=224,
        classification_resize_size=0,
        det_epochs=1,
        det_batch_size=2,
        detection_channel_policy="auto",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class BioEvalOrchestrationTests(unittest.TestCase):
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
