"""Small CPU-only checks for admission, crash recovery and child reaping."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import unified_paper_host_admission as host


def wait_for(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError('Timed out')


class HostResourcesTests(unittest.TestCase):
    def test_queue_policy_reads_are_cached_and_refresh(self):
        from unittest.mock import patch
        import unified_paper_gpu_policy as policy
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'gpu_exclusions.json').write_text(json.dumps({'hosts': {'deepcad': {'gpus': [0]}}}))
            original_read = Path.read_text
            with patch.object(policy, 'OUT', root), patch.object(policy.time, 'monotonic', return_value=10):
                with patch.object(Path, 'read_text', autospec=True, side_effect=original_read) as read:
                    for _ in range(1000):
                        self.assertTrue(policy.excluded('deepcad', 0))
                    self.assertEqual(read.call_count, 1)
            (root / 'gpu_exclusions.json').write_text(json.dumps({'hosts': {'deepcad': {'gpus': [1]}}}))
            with patch.object(policy, 'OUT', root), patch.object(policy.time, 'monotonic', return_value=16):
                self.assertFalse(policy.excluded('deepcad', 0))
                self.assertTrue(policy.excluded('deepcad', 1))

    def test_future_growth_and_waiters(self):
        rows = {1: dict(entry='run_unified_paper_ctc.py', rss=16 * host.GIB),
                2: dict(entry='run_unified_paper_global.py', rss=host.GIB)}
        fits, info = host.budget(256 * host.GIB, 150 * host.GIB, rows, {}, 3, 32 * host.GIB)
        self.assertFalse(fits)  # 80 + 31 GiB unallocated peaks, plus new task.
        self.assertEqual(info['pending_peak_gib'], 111)
        fits, info = host.budget(256 * host.GIB, 150 * host.GIB, rows,
                                {'2': {'status': 'waiting'}}, 3, 16 * host.GIB)
        self.assertTrue(fits)
        self.assertEqual(info['active'], 1)

    def test_concurrent_admission_and_crashed_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'run_unified_paper_fixture.py'
            script.write_text(
                f'import sys,time,os\nfrom pathlib import Path\nsys.path.insert(0,{str(ROOT / "scripts")!r})\n'
                'import unified_paper_host_admission as h\n'
                f'h.DIRECTORY=Path({str(root / "locks")!r})\n'
                'h.memory=lambda:(64*h.GIB,64*h.GIB)\n'
                'scan=h.evaluators\nh.evaluators=lambda:{p:r for p,r in scan().items() if r["entry"]==Path(__file__).name}\n'
                'real_sleep=time.sleep\nh.time.sleep=lambda n:real_sleep(.05)\n'
                "h.reserve_for_current_command({'key':str(os.getpid()),'entrypoint':Path(__file__).name})\n"
                f'Path({str(root)!r},str(os.getpid())+".admitted").touch()\n'
                'real_sleep(30)\n')
            processes = [subprocess.Popen([sys.executable, str(script)], stdout=subprocess.DEVNULL)
                         for _ in range(3)]
            try:
                wait_for(lambda: len(list(root.glob('*.admitted'))) == 1)
                time.sleep(.3)
                self.assertEqual(len(list(root.glob('*.admitted'))), 1)
                pid = int(next(root.glob('*.admitted')).stem)
                first = next(p for p in processes if p.pid == pid)
                first.kill()
                first.wait()
                wait_for(lambda: len(list(root.glob('*.admitted'))) == 2)
                time.sleep(.3)
                self.assertEqual(len(list(root.glob('*.admitted'))), 2)
            finally:
                for proc in processes:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()

    def test_worker_allocation_fails_without_oom(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = (f'import sys\nfrom pathlib import Path\nsys.path.insert(0,{str(ROOT / "scripts")!r})\n'
                    'import unified_paper_statistics_resources as r\n'
                    f'r.DIRECTORY=Path({tmp!r})\nr.GIB=1024**2\n'
                    'r.initialize_worker()\n'
                    'try:\n x=bytearray(32*1024**2)\nexcept MemoryError:\n print("bounded")\n')
            proc = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                  text=True, timeout=10)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), 'bounded')

    def test_statistics_slots_are_shared_and_released_on_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = (f'import sys,os,time\nfrom pathlib import Path\nsys.path.insert(0,{str(ROOT / "scripts")!r})\n'
                    'import unified_paper_statistics_resources as r\n'
                    f'r.DIRECTORY=Path({tmp!r})\n'
                    'r.memory=lambda:(512*r.GIB,512*r.GIB)\n'
                    'sleep=time.sleep\nr.time.sleep=lambda n:sleep(.05)\n'
                    'r.initialize_worker()\n'
                    'r.DIRECTORY.joinpath(str(os.getpid())+".ready").touch()\n'
                    'sleep(30)\n')
            processes = [subprocess.Popen([sys.executable, '-c', code]) for _ in range(9)]
            try:
                wait_for(lambda: len(list(root.glob('*.ready'))) == 8)
                time.sleep(.2)
                self.assertEqual(len(list(root.glob('*.ready'))), 8)
                pid = int(next(root.glob('*.ready')).stem)
                first = next(p for p in processes if p.pid == pid)
                first.kill()
                first.wait()
                wait_for(lambda: len(list(root.glob('*.ready'))) == 9)
            finally:
                for proc in processes:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()

    def test_fleet_reaps_completed_service(self):
        import watch_unified_native_fleet as fleet
        proc = subprocess.Popen([sys.executable, '-c', 'pass'])
        fleet.SERVICE_CHILDREN.append(proc)
        wait_for(lambda: Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z')
        fleet.reap_services()
        self.assertNotIn(proc, fleet.SERVICE_CHILDREN)
        self.assertFalse(Path(f'/proc/{proc.pid}').exists())

    def test_sync_manager_reaps_completed_transfer(self):
        import manage_unified_ai_branches as manager
        proc = subprocess.Popen([sys.executable, '-c', 'pass'])
        manager.SYNC_CHILDREN['fixture'] = proc
        wait_for(lambda: Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z')
        manager.reap_sync_children()
        self.assertEqual(proc.returncode, 0)
        self.assertFalse(Path(f'/proc/{proc.pid}').exists())
        del manager.SYNC_CHILDREN['fixture']


if __name__ == '__main__':
    unittest.main()
