"""Run the unchanged v4 detection evaluator with a CPU startup barrier.

The pinned environment segfaults in its NCCL barrier on deepcad, before loading
the model. Detection has one rank and no tensor collectives; changing only the
startup process-group backend leaves CUDA inference and probe training intact.
"""
import os
import torch.distributed as dist
from dinov3.eval.bio_detection import center_probe


def main():
    assert int(os.environ.get('WORLD_SIZE', '1')) == 1
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
    original = dist.init_process_group

    def single_rank_init(*args, **kwargs):
        assert not args and kwargs.get('backend') == 'nccl'
        kwargs['backend'] = 'gloo'
        print('[runtime] Single-rank startup barrier uses Gloo; CUDA evaluation unchanged.', flush=True)
        return original(**kwargs)

    dist.init_process_group = single_rank_init
    try:
        center_probe.main()
    finally:
        dist.init_process_group = original
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
