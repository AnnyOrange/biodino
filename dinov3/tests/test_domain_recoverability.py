"""Check domain isolation and unequal-rank weighted statistics."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dinov3.data.teacher_domains import teacher_domain
from dinov3.train.external_distill import ExternalRecoveryStream


def distributed_worker(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    torch.manual_seed(18)
    x, y = torch.randn(8, 2, 3), torch.randn(8, 2, 4)
    weights = torch.tensor([0., 0., 0., 0., 1., 1., 1., 0.])
    model = ExternalRecoveryStream(3, 4, warmup=0, solve_every=1)
    for _ in range(3):
        local_x = x[rank*4:(rank+1)*4].clone().requires_grad_()
        loss, _ = model(local_x, y[rank*4:(rank+1)*4], weights[rank*4:(rank+1)*4])
        loss.backward()
    torch.save(dict(state=model.state_dict(), loss=loss.detach(), grad=local_x.grad), Path(output)/f'{rank}.pth')
    dist.destroy_process_group()


class DomainRecoveryTests(unittest.TestCase):
    def test_metadata_does_not_treat_all_r9_as_pathology(self):
        self.assertEqual(teacher_domain({'meta.json': b'{"imaging_modality":"fluorescence"}', '__url__':'r9.tar'}), 'other')
        self.assertEqual(teacher_domain({'meta.json': {'imaging_modality':'brightfield_pathology',
                                                     'tar_member_type':'pathology_rgb_patch'}}), 'pathology')
        with self.assertRaises(ValueError):
            teacher_domain({})

    def test_all_eligible_matches_original(self):
        torch.manual_seed(19)
        original = ExternalRecoveryStream(3, 4, warmup=2, solve_every=1)
        routed = copy.deepcopy(original)
        for _ in range(5):
            x, y = torch.randn(6, 2, 3), torch.randn(6, 2, 4)
            a, _ = original(x, y)
            b, _ = routed(x, y, torch.ones(6))
            torch.testing.assert_close(a, b)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, routed.state_dict()[key])

    def test_excluded_images_cannot_change_decoder_or_receive_gradients(self):
        torch.manual_seed(20)
        a = ExternalRecoveryStream(3, 4, warmup=0, solve_every=1)
        b = copy.deepcopy(a)
        weights = torch.tensor([1., 1., 0., 0.])
        for _ in range(3):
            x, y = torch.randn(4, 2, 3, requires_grad=True), torch.randn(4, 2, 4)
            altered_x, altered_y = x.detach().clone(), y.clone()
            altered_x[2:] += 1000
            altered_y[2:] -= 2000
            loss_a, _ = a(x, y, weights)
            loss_b, _ = b(altered_x, altered_y, weights)
            torch.testing.assert_close(loss_a, loss_b)
            loss_a.backward()
            self.assertEqual(x.grad[2:].count_nonzero().item(), 0)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key])
        before = copy.deepcopy(a.state_dict())
        empty, _ = a(x, y, torch.zeros(4))
        self.assertEqual(empty.item(), 0)
        for key in before:
            torch.testing.assert_close(before[key], a.state_dict()[key])

    def test_rank_with_no_eligible_images_matches_global_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(distributed_worker, args=('file://' + folder + '/rendezvous', folder), nprocs=2)
            ranks = [torch.load(Path(folder)/f'{i}.pth', weights_only=True) for i in range(2)]
        torch.manual_seed(18)
        x, y = torch.randn(8, 2, 3, requires_grad=True), torch.randn(8, 2, 4)
        weights = torch.tensor([0., 0., 0., 0., 1., 1., 1., 0.])
        model = ExternalRecoveryStream(3, 4, warmup=0, solve_every=1)
        for _ in range(3):
            x.grad = None
            loss, _ = model(x, y, weights)
            loss.backward()
        torch.testing.assert_close(loss, (ranks[0]['loss'] + ranks[1]['loss']) / 2)
        for i, record in enumerate(ranks):
            torch.testing.assert_close(record['grad'] / 2, x.grad[i*4:(i+1)*4])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, record['state'][key])


if __name__ == '__main__':
    unittest.main()
