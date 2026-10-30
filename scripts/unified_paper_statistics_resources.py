"""Bound aggregate statistics work without changing seeds or resampling budgets."""
import contextlib
import fcntl
import multiprocessing
import os
from pathlib import Path
import resource
import time

from unified_paper_host_admission import DIRECTORY, GIB, memory

_slot = None


def initialize_worker():
    global _slot
    DIRECTORY.mkdir(mode=0o700, exist_ok=True)
    # Eight active workers across all participating statistics services, not
    # eight independent pools of 32 workers each. Kernel releases locks on exit.
    while _slot is None:
        total, available = memory()
        if available >= max(8 * GIB, total * .20) + 8 * GIB:
            for index in range(8):
                candidate = (DIRECTORY / f'statistics-{index}.lock').open('a')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                    continue
                _slot = candidate
                break
        if _slot is None:
            time.sleep(5)
    # Bound additional virtual allocations per worker. A bad allocation raises
    # MemoryError in the job instead of provoking the machine-wide OOM killer.
    vm = int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = vm + 8 * GIB
    if soft != resource.RLIM_INFINITY:
        limit = min(limit, soft)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


@contextlib.contextmanager
def pool(workers):
    if workers < 1:
        raise ValueError('workers must be positive')
    # Recycle workers to release allocator high-water marks between jobs.
    with multiprocessing.get_context('fork').Pool(min(workers, 4),
            initializer=initialize_worker, maxtasksperchild=16) as workers_pool:
        yield workers_pool
