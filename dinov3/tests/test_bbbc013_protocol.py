from __future__ import annotations

import unittest

import numpy as np

from dinov3.eval.bio_frozen_eval.probes import run_bbbc013_compound_oof_probe


class BBBC013ProtocolTests(unittest.TestCase):
    def test_compound_oof_uses_all_replicate_rows(self):
        wortmannin = np.array([0, 0, 0.98, 1.95, 3.91, 7.81, 15.63, 31.25, 62.5, 125, 250, 150])
        ly294002 = np.array([80, 0, 0.31, 0.63, 1.25, 2.5, 5, 10, 20, 40, 80, 0])
        targets = np.concatenate([np.tile(wortmannin, 4), np.tile(ly294002, 4)])
        compound = np.concatenate([np.zeros(48), np.ones(48)])
        log_dose = np.log1p(targets)
        features = np.stack([log_dose, compound, log_dose * compound, log_dose**2], axis=1)
        paths = [
            f"Channel1-{index + 1:02d}-{chr(ord('A') + index // 12)}-{index % 12 + 1:02d}.BMP"
            for index in range(96)
        ]

        result = run_bbbc013_compound_oof_probe(features, targets, paths)

        self.assertEqual(result.n_train, 36)
        self.assertEqual(result.n_test, 96)
        self.assertGreater(result.metrics["wortmannin_r2"], 0.95)
        self.assertGreater(result.metrics["ly294002_r2"], 0.95)
        self.assertGreater(result.metrics["spearman"], 0.95)

    def test_compound_oof_rejects_incomplete_plate(self):
        with self.assertRaisesRegex(ValueError, "must contain 48 wells"):
            run_bbbc013_compound_oof_probe(
                np.ones((12, 2)),
                np.arange(12),
                [f"Channel1-{i + 1:02d}-A-{i + 1:02d}.BMP" for i in range(12)],
            )


if __name__ == "__main__":
    unittest.main()
