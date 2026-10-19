"""A resumed offloaded optimizer must preserve moments and the next update."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcpsd
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

from dinov3.checkpointer.checkpointer import load_checkpoint, save_checkpoint


def sharded_worker(rank, world_size, rendezvous, folder, saving):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=world_size)
    mesh = init_device_mesh('cpu', (world_size,))
    model = torch.nn.Module()
    full = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    model.register_parameter('weight', torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)])))
    optim = torch.optim.AdamW(model.parameters(), lr=.003, foreach=False)
    if saving:
        optim.state[model.weight] = dict(step=torch.tensor(3.),
            exp_avg=distribute_tensor(full / 100, mesh, [Shard(0)]),
            exp_avg_sq=distribute_tensor(full / 50, mesh, [Shard(0)]))
        with patch.dict('os.environ', DINOV3_DCP_SINGLE_FILE_PER_RANK='0'):
            save_checkpoint(folder, model=model, optimizer=optim, iteration=2, sharded=True)
        # Each rank has multiple tensors; ensure the bounded-memory layout was
        # actually exercised before restoring into a different world size.
        assert len(list(Path(folder).glob('*.distcp'))) > world_size
    else:
        assert load_checkpoint(folder, model=model, optimizer=optim, optimizer_cpu_offload=True) == 2
        torch.testing.assert_close(optim.state[model.weight]['exp_avg'].full_tensor(), full / 100)
        torch.testing.assert_close(optim.state[model.weight]['exp_avg_sq'].full_tensor(), full / 50)
        torch.testing.assert_close(model.weight.full_tensor(), full)
        assert optim.state[model.weight]['step'].item() == 3
    dist.destroy_process_group()


class CPUOptimizerRestoreTests(unittest.TestCase):
    def test_optimizer_moments_reshard_from_two_ranks_to_one(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = folder + '/checkpoint'
            mp.spawn(sharded_worker, args=(2, 'file://' + folder + '/save', checkpoint, True), nprocs=2)
            mp.spawn(sharded_worker, args=(1, 'file://' + folder + '/load', checkpoint, False), nprocs=1)

    def test_preserves_state_groups_and_next_update_without_dummy_step(self):
        torch.manual_seed(18)
        original = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        def optimizer(model):
            return torch.optim.AdamW([
                dict(params=list(model[0].parameters()), lr=.003),
                dict(params=list(model[1].parameters()), lr=.001, weight_decay=.02),
            ], foreach=False)
        optim = optimizer(original)
        x, y = torch.randn(6, 3), torch.randn(6, 2)
        def update(model, opt):
            opt.zero_grad(set_to_none=True)
            (model(x) - y).square().mean().backward()
            opt.step()
        for _ in range(3):
            update(original, optim)
        restored = copy.deepcopy(original)
        new_optim = optimizer(restored)
        with tempfile.TemporaryDirectory() as folder:
            dcp.save(dict(model=dcpsd.get_model_state_dict(original),
                          optimizer=dcpsd.get_optimizer_state_dict(original, optim), iteration=2),
                     checkpoint_id=folder)
            with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('dummy optimizer step')):
                self.assertEqual(load_checkpoint(Path(folder), model=restored, optimizer=new_optim,
                                                 optimizer_cpu_offload=True), 2)
        for a, b in zip(optim.param_groups, new_optim.param_groups):
            self.assertEqual(a['lr'], b['lr'])
            self.assertEqual(a['weight_decay'], b['weight_decay'])
        for a, b in zip(original.parameters(), restored.parameters()):
            for key in optim.state[a]:
                torch.testing.assert_close(optim.state[a][key], new_optim.state[b][key])
        update(original, optim)
        update(restored, new_optim)
        for a, b in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
