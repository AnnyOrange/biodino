"""Verify complementary knowledge targets without teacher or student leakage."""
import copy
import unittest

import torch

from dinov3.train.external_distill import ExternalRecoveryStream, TeacherResidualizer


class TeacherResidualTests(unittest.TestCase):
    def test_removes_predictable_component_preserves_complement(self):
        torch.manual_seed(18)
        reference = torch.randn(512, 2, 3)
        novel = torch.randn(512, 2, 2)
        mapping = torch.randn(3, 2)
        bias = torch.randn(2)
        target = reference @ mapping + bias + novel
        residualizer = TeacherResidualizer(3, 2, ridge=1e-5, warmup=2, solve_every=1)
        for _ in range(3):
            residual, _ = residualizer(reference, target)
        self.assertLess(((residual - novel) ** 2).mean().item(), .03)
        self.assertLess(residual.square().mean().item(), target.square().mean().item() * .5)

    def test_projection_uses_previous_batch_and_is_detached(self):
        reference = torch.randn(8, 2, 3, requires_grad=True)
        target = torch.randn(8, 2, 4, requires_grad=True)
        residualizer = TeacherResidualizer(3, 4, warmup=0, solve_every=1)
        residual, _ = residualizer(reference, target)
        torch.testing.assert_close(residual, target.detach())
        self.assertFalse(residual.requires_grad)
        student = torch.randn(8, 2, 3, requires_grad=True)
        stream = ExternalRecoveryStream(3, 4, warmup=0, solve_every=1)
        stream(student, residual)
        loss, _ = stream(student, residual)
        loss.backward()
        self.assertGreater(student.grad.norm().item(), 0)
        self.assertIsNone(reference.grad)
        self.assertIsNone(target.grad)

    def test_ineligible_domain_cannot_change_projection(self):
        torch.manual_seed(19)
        a = TeacherResidualizer(3, 4, warmup=0, solve_every=1)
        b = copy.deepcopy(a)
        weights = torch.tensor([1., 1., 0., 0.])
        for _ in range(3):
            x, y = torch.randn(4, 2, 3), torch.randn(4, 2, 4)
            altered_x, altered_y = x.clone(), y.clone()
            altered_x[2:] += 1000
            altered_y[2:] -= 2000
            ra, _ = a(x, y, weights)
            rb, _ = b(altered_x, altered_y, weights)
            torch.testing.assert_close(ra[:2], rb[:2])
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key])


if __name__ == '__main__':
    unittest.main()
