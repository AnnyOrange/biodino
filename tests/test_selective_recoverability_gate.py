import unittest

import torch

from dinov3.train.selective_recoverability import RecoveryStream


class RecoveryGateTest(unittest.TestCase):
    def test_adaptive_floor_spans_fixed_one_to_three_weight(self):
        for dual, expected in ((0.0, 1.0 / 3.0), (0.5, 2.0 / 3.0), (1.0, 1.0)):
            stream = RecoveryStream(dim=4, mode="adaptive_floor")
            stream.steps.fill_(32)
            stream.second.fill_(1.0)
            stream.dual.fill_(dual)
            student = torch.zeros(4, 1, 4, requires_grad=True)
            anchor = torch.ones_like(student)
            loss, stats = stream(student, anchor)

            self.assertAlmostEqual(float(stats["gate"]), expected, places=6)
            self.assertAlmostEqual(float(loss.detach()), expected, places=6)
            (3.0 * loss).backward()
            self.assertIsNotNone(student.grad)


if __name__ == "__main__":
    unittest.main()
