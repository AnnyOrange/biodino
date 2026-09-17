"""Resource policy checks; never starts GPU jobs."""
import unittest
import tempfile
import json
from pathlib import Path

from run_resident_seg_campaign_20260917 import admission, extracting, verify_launch_gates, REQUIRED_GATES


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.config = {"max_host_jobs": 24, "minimum_ram_gib": 96,
                       "minimum_disk_gib": 128, "minimum_gpu_free_mib": 16384}
        self.resources = {0: {"used": 2000, "total": 32000, "free": 30000}}
        self.ram = 200 * 1024**3
        self.disk = 500 * 1024**3

    def allowed(self, tasks, **overrides):
        return admission(self.config, 0, tasks, overrides.get("resources", self.resources),
                         overrides.get("ram", self.ram), self.disk, overrides.get("commands", []))

    def task(self, **changes):
        return dict(gpu=0, dataset="conic", probe_seen=True, cache_run_name="model", **changes)

    def test_idle_card_and_three_cached_probes(self):
        self.assertTrue(self.allowed([]))
        self.assertTrue(self.allowed([self.task(), self.task()]))
        self.assertFalse(self.allowed([self.task(), self.task(), self.task()]))

    def test_encoder_and_monuseg_peak_gate(self):
        task = self.task()
        task["probe_seen"] = False
        self.assertFalse(self.allowed([task]))
        task = self.task()
        task["dataset"] = "monuseg"
        self.assertFalse(self.allowed([task]))

    def test_ram_and_busy_card_limits(self):
        self.assertFalse(self.allowed([], ram=50 * 1024**3))
        busy = {0: {"used": 21000, "total": 32000, "free": 11000}}
        cfg = dict(self.config, minimum_gpu_free_mib=8000)
        self.assertTrue(admission(cfg, 0, [], busy, self.ram, self.disk, []))
        self.assertFalse(admission(cfg, 0, [self.task()], busy, self.ram, self.disk, []))

    def test_extraction_checks_model_and_dataset(self):
        command = "python -m dinov3.eval.bio_segmentation.feature_extractor --dataset conic --output-dir model/cache "
        self.assertTrue(extracting(self.task(), [command]))
        self.assertFalse(extracting(self.task(), [command.replace("conic", "livecell")]))

    def test_missing_preflight_cannot_launch(self):
        with self.assertRaises(RuntimeError):
            verify_launch_gates({}, "commit")

    def test_every_gate_and_commit_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preflight.json"
            report = {"status": "PASS", "git_commit": "commit",
                      "checks": {key: True for key in REQUIRED_GATES}}
            path.write_text(json.dumps(report))
            self.assertEqual(verify_launch_gates({"preflight_report": str(path)}, "commit"), report)
            with self.assertRaises(RuntimeError):
                verify_launch_gates({"preflight_report": str(path)}, "different")
            report["checks"]["extraction_batch_invariance"] = False
            path.write_text(json.dumps(report))
            with self.assertRaises(RuntimeError):
                verify_launch_gates({"preflight_report": str(path)}, "commit")


if __name__ == "__main__":
    unittest.main()
