"""Apply a CUDA allocator limit to the evaluation process and its children only."""
import atexit
import json
import os
from pathlib import Path

_limit = os.environ.get('HS6_EVAL_ALLOCATOR_MIB')
if _limit:
    import torch

    _original_load = torch.load

    def _mapped_load(path, *args, **kwargs):
        # Consolidated EMA files use torch's zip format. Mapping preserves all
        # tensor values and avoids three simultaneous full teacher copies on CPU.
        if isinstance(path, (str, os.PathLike)) and Path(path).name in ('teacher_checkpoint.pth', 'checkpoint.pth'):
            kwargs.setdefault('mmap', True)
            try:
                physical = Path(os.getxattr(path, 'user.mergerfs.fullpath').decode())
                logical_stat, physical_stat = os.stat(path), physical.stat()
                if ((logical_stat.st_size, logical_stat.st_mtime_ns) ==
                        (physical_stat.st_size, physical_stat.st_mtime_ns)):
                    path = physical
            except OSError:
                pass
        return _original_load(path, *args, **kwargs)

    torch.load = _mapped_load
    _original_init = torch.cuda._lazy_init
    _configured = False
    _configuring = False

    def _bounded_init():
        global _configured, _configuring
        _original_init()
        if _configured or _configuring or torch.cuda._is_in_bad_fork():
            return
        _configuring = True
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(int(_limit) * 2**20 / total, 0)
            _configured = True
        finally:
            _configuring = False

    torch.cuda._lazy_init = _bounded_init

    def _report():
        if not _configured or not torch.cuda.is_initialized():
            return
        directory = os.environ.get('HS6_EVAL_MEMORY_REPORTS')
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / f'{os.getpid()}.json').write_text(json.dumps(dict(
                pid=os.getpid(), allocator_limit_mib=int(_limit),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)))

    atexit.register(_report)
