"""Admission must see account quotas even when the host has free RAM."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from unified_paper_memory import memory

GIB = 1024**3


class CgroupMemoryTests(unittest.TestCase):
    def fixture(self, root, kind):
        proc, mount = root / 'proc', root / 'cg'
        (proc / 'self').mkdir(parents=True)
        (mount / 'user/session').mkdir(parents=True)
        (proc / 'meminfo').write_text('MemTotal: 528482304 kB\nMemAvailable: 278921216 kB\n')
        membership = '0::/user/session' if kind == 'cgroup2' else '12:memory:/user/session'
        (proc / 'self/cgroup').write_text(membership + '\n')
        (proc / 'self/mountinfo').write_text(f'1 0 0:1 / {mount} rw - {kind} none rw,memory\n')
        return proc, mount

    def limit(self, directory, kind, limit, used, inactive=0):
        names = ('memory.max', 'memory.current', 'inactive_file') if kind == 'cgroup2' else ('memory.limit_in_bytes', 'memory.usage_in_bytes', 'total_inactive_file')
        (directory / names[0]).write_text(str(limit))
        (directory / names[1]).write_text(str(used))
        (directory / 'memory.stat').write_text(f'{names[2]} {inactive}\n')

    def test_v1_parent_account_limit_and_reclaimable_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, mount = self.fixture(Path(tmp), 'cgroup')
            self.limit(mount / 'user/session', 'cgroup', 2**63 - 4096, 20*GIB)
            self.limit(mount / 'user', 'cgroup', 400*GIB, 390*GIB, 50*GIB)
            self.assertEqual(memory(proc), (400*GIB, 60*GIB))
            # No cache escape hatch when account use is anonymous memory.
            self.limit(mount / 'user', 'cgroup', 400*GIB, 401*GIB)
            self.assertEqual(memory(proc), (400*GIB, 0))

    def test_v2_tighter_session_and_parent_both_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, mount = self.fixture(Path(tmp), 'cgroup2')
            self.limit(mount / 'user/session', 'cgroup2', 100*GIB, 60*GIB)
            self.limit(mount / 'user', 'cgroup2', 400*GIB, 380*GIB)
            self.assertEqual(memory(proc), (100*GIB, 20*GIB))
            self.limit(mount / 'user/session', 'cgroup2', 'max', 60*GIB)
            self.assertEqual(memory(proc), (400*GIB, 20*GIB))

    def test_host_available_remains_upper_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, mount = self.fixture(Path(tmp), 'cgroup')
            self.limit(mount / 'user', 'cgroup', 400*GIB, 20*GIB)
            self.assertEqual(memory(proc), (400*GIB, 266*GIB))


if __name__ == '__main__':
    unittest.main()
