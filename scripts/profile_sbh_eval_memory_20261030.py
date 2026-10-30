#!/usr/bin/env python3
"""Measure frozen evaluator forwards within an atomically reserved GPU budget."""
import argparse
import fcntl
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--model', choices=['splus', 'b', 'hplus'], required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--reserve-mib', type=int, default=4000)
    parser.add_argument('--allocator-mib', type=int, default=3300)
    parser.add_argument('--ram-mib', type=int, default=32768)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.path.insert(0, str(args.repo))
    spec = importlib.util.spec_from_file_location('screening', args.repo / 'scripts/run_screening45_8k_20261024.py')
    screening = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screening)
    manifest = screening.read(args.campaign / 'evaluation/campaign_manifest.json')
    screening.fingerprint_ok(args.repo, manifest)
    if screening.available_ram_mib() < 65536 + args.ram_mib:
        raise RuntimeError('Insufficient host RAM for a bounded profiling job')
    task = {'key': 'memory_profile_' + args.model, 'reserve_mib': args.reserve_mib}
    reservation = screening.reserve_gpu(SimpleNamespace(repo=args.repo, gpu=args.gpu), task)
    directory, path, record = reservation
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        peers = screening.live_reservations(path)
        record['ram_mib'] = args.ram_mib
        peers[str(os.getpid())] = record
        screening.save(path, dict(reservations=peers, updated=time.time()))
    report = dict(model=args.model, gpu=args.gpu, pid=os.getpid(), started=time.time(),
                  reserve_mib=args.reserve_mib, allocator_mib=args.allocator_mib, cases=[])
    destination = args.campaign / 'evaluation_capacity_20261030' / ('profile_' + args.model + '.json')
    try:
        import torch
        from dinov3.eval.bio_frozen_eval.encoder import Dinov3CkptEncoder
        original_load = torch.load

        def mapped_load(path, *values, **kwargs):
            if isinstance(path, (str, os.PathLike)) and Path(path).suffix == '.pth':
                kwargs.setdefault('mmap', True)
                try:
                    physical = Path(os.getxattr(path, 'user.mergerfs.fullpath').decode())
                    logical_stat, physical_stat = os.stat(path), physical.stat()
                    if ((logical_stat.st_size, logical_stat.st_mtime_ns) ==
                            (physical_stat.st_size, physical_stat.st_mtime_ns)):
                        path = physical
                except OSError:
                    pass
            return original_load(path, *values, **kwargs)

        torch.load = mapped_load
        def check_host():
            while True:
                if screening.available_ram_mib() < 61440:
                    report.update(status='INTERRUPTED_HOST_RAM_FLOOR', finished=time.time())
                    screening.save(destination, report)
                    os._exit(75)
                time.sleep(3)
        threading.Thread(target=check_host, daemon=True).start()
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(args.allocator_mib * 2**20 / torch.cuda.get_device_properties(0).total_memory)
        encoder = Dinov3CkptEncoder(
            args.campaign / 'runs' / args.model / 'eval/training_487/teacher_checkpoint.pth',
            args.campaign / 'configs' / (args.model + '.yaml'), 'cuda', 1, True, torch.bfloat16)
        report['parameters_mib'] = sum(p.numel() * p.element_size() for p in encoder.model.parameters()) / 2**20
        for batch, size in [(4, 256), (4, 512), (4, 768), (64, 224), (64, 384), (64, 512)]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            item = dict(batch=batch, size=size)
            samples = features = None
            try:
                samples = torch.zeros(batch, 3, size, size, device='cuda')
                for _ in range(2):
                    features = encoder.model(samples)
                torch.cuda.synchronize()
                item.update(status='PASS', shape=list(features.shape),
                            finite=bool(torch.isfinite(features).all()))
            except torch.cuda.OutOfMemoryError as error:
                item.update(status='ALLOCATOR_BUDGET_EXCEEDED', error=str(error)[:500])
            item.update(allocated_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                        reserved_peak_mib=torch.cuda.max_memory_reserved() / 2**20,
                        nvml_mib=screening.gpu_state()[args.gpu]['apps'].get(os.getpid(), 0))
            report['cases'].append(item)
            screening.save(destination, report)
            print(json.dumps(item), flush=True)
            del samples, features
            gc.collect()
        report['status'] = 'PROFILE_COMPLETE'
    except Exception as error:
        report.update(status='PROFILE_FAILED', error=repr(error))
        print(json.dumps(report), flush=True)
    finally:
        report['finished'] = time.time()
        screening.save(destination, report)
        screening.update_gpu_reservation(reservation, os.getpid(), args.gpu, release=True)


if __name__ == '__main__':
    main()
