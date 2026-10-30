"""Two host-local CTC scoring slots and a temporary allocation ceiling."""
import contextlib
import fcntl
import os
from pathlib import Path
import resource
import time
from scripts.unified_paper_memory import memory


@contextlib.contextmanager
def scoring_budget():
    directory = Path('/tmp') / f'dinov3-ctc-scoring-{os.getuid()}'
    directory.mkdir(mode=0o700, exist_ok=True)
    slot = None
    notice = 0
    while slot is None:
        total, available = memory()
        if available >= max(8 * 1024**3, .20 * total) + 16 * 1024**3:
            for index in range(2):
                candidate = (directory / f'{index}.lock').open('a')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                    continue
                slot = candidate
                break
        if slot is None:
            if time.monotonic() - notice > 60:
                print('CTC SCORING WAIT: two shared slots and host RAM headroom', flush=True)
                notice = time.monotonic()
            time.sleep(5)
    previous = resource.getrlimit(resource.RLIMIT_AS)
    try:
        vm = int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')
        limit = min([vm + 16 * 1024**3] + [x for x in previous if x != resource.RLIM_INFINITY])
        resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)
        slot.close()
