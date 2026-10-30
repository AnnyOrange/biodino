"""CPU-only process tests; all queue files and results live in temporary dirs."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate, timeout=35):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError('Timed out waiting for process state')


class QueueLifecycleTests(unittest.TestCase):
    def test_fleet_finds_stopped_worker_without_heartbeat(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        import watch_unified_native_fleet as fleet
        import signal
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'unified_paper_queue.py'
            script.write_text('import time;time.sleep(60)\n')
            worker = subprocess.Popen([sys.executable, str(script), '--host-label',
                'lifecycle-test', '--gpu', '1', '--slot', 'no-heartbeat'])
            try:
                os.kill(worker.pid, signal.SIGSTOP)
                wait_for(lambda: Path('/proc', str(worker.pid), 'stat').read_text()
                         .rsplit(')', 1)[1].split()[0] == 'T')
                _, data = fleet.check('server', set())
                rows = data['slots']['lifecycle-test_gpu1_slotno-heartbeat']
                self.assertIn(dict(pid=worker.pid, state='T'), rows)
            finally:
                worker.kill()
                worker.wait()

    def test_update_waits_for_child_and_preserves_pid_and_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / 'scripts'
            scripts.mkdir()
            out = root / 'outputs/02_eval_runs/unified_paper'
            for name in ('locks', 'states', 'logs'):
                (out / name).mkdir(parents=True)
            script = scripts / 'unified_paper_queue.py'
            shutil.copy2(ROOT / 'scripts/unified_paper_queue.py', script)
            shutil.copy2(ROOT / 'scripts/unified_paper_gpu_policy.py', scripts)
            # Isolate this lifecycle fixture from production host occupancy.
            (scripts / 'unified_paper_host_admission.py').write_text('def launch_ready(job):return True, {}\n')
            child_script = scripts / 'run_unified_paper_dummy.py'
            child_script.write_text(
                'import pathlib,sys,time\n'
                'p=pathlib.Path(sys.argv[1]);p.with_suffix(".started").write_text("yes")\n'
                'while not p.with_suffix(".release").exists():time.sleep(.02)\n'
                'p.write_text("complete")\n')
            result = out / 'result.json'
            (out / 'config.json').write_text(json.dumps({'models': [{'model': 'dummy'}]}))
            queue = out / 'queue.json'
            queue.write_text(json.dumps({'tasks': [dict(
                key='lifecycle_test_' + root.name, model='dummy', result=str(result),
                log=str(out / 'logs/child.log'), entrypoint=child_script.name,
                args=[str(result)])]}))
            args = [sys.executable, '-u', str(script), '--cpu', '--host-label',
                    'lifecycle-test', '--slot', 'one', '--queue-file', str(queue)]
            log = (out / 'logs/worker.log').open('w')
            worker = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            state_path = out / 'states/worker_lifecycle-test_gpu-1_slotone.json'
            marker = out / 'reloaded.json'
            child_pid = None
            try:
                wait_for(lambda: state_path.exists() and result.with_suffix('.started').exists())
                state = json.loads(state_path.read_text())
                child_pid = state['child_pid']
                self.assertEqual(state['status'], 'running')
                source = script.read_text()
                source = source.replace('ROOT=Path(__file__)',
                    f'Path({str(marker)!r}).write_text(str(os.getpid()))\nROOT=Path(__file__)')
                temporary = script.with_suffix('.next')
                temporary.write_text(source)
                temporary.replace(script)
                time.sleep(6)  # Beyond reload polling interval, with child still running.
                self.assertFalse(marker.exists())
                self.assertTrue(Path('/proc', str(child_pid)).exists())
                duplicate = subprocess.run(args, capture_output=True, timeout=5)
                self.assertEqual(duplicate.returncode, 0)
                # The duplicate imports the new code; remove its marker before checking exec.
                marker.unlink(missing_ok=True)
                result.with_suffix('.release').touch()
                wait_for(marker.exists)
                self.assertEqual(int(marker.read_text()), worker.pid)
                self.assertEqual(result.read_text(), 'complete')
                self.assertIsNone(worker.poll())
                lock_path = out / 'locks/worker_lifecycle-test_gpu-1_slotone.lock'
                with lock_path.open('a') as lock:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual((out / 'logs/child.log').read_text().count('START '), 1)
                self.assertFalse(any((out / 'locks').glob('*.claimdir')))
            finally:
                result.with_suffix('.release').touch()
                worker.terminate()
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
                log.close()

    def test_retired_refresh_cannot_freeze_or_spawn_workers(self):
        process = subprocess.run([sys.executable,
            str(ROOT / 'scripts/refresh_unified_idle_workers.py'),
            '--resident', 'test', '--watch'], capture_output=True, text=True, timeout=5)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(json.loads(process.stdout)['status'], 'disabled_external_rollout')

    def test_cleanup_guards_and_pidfd_signal(self):
        spec = importlib.util.spec_from_file_location('cleanup',
            ROOT / 'scripts/cleanup_stopped_unified_queues.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record = dict(pid=123, state='T', ppid=1, children=[])
        self.assertTrue(module.eligible(record, set()))
        self.assertFalse(module.eligible(record, {123}))
        for changes in (dict(state='S'), dict(state='R'), dict(state='Z'),
                        dict(state='t'), dict(ppid=42), dict(children=['456'])):
            self.assertFalse(module.eligible(dict(record, **changes), set()))
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        fd = None
        try:
            fd = module.pidfd_open(child.pid)
            module.pidfd_kill(fd)
            self.assertEqual(child.wait(timeout=5), -9)
        finally:
            if fd is not None:
                os.close(fd)
            if child.poll() is None:
                child.kill()
                child.wait()


if __name__ == '__main__':
    unittest.main()
