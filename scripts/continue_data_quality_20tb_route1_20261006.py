#!/usr/bin/env python3
"""Finish route-1 materialization, then train and evaluate its matched 1M arm."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path


REPO = Path("/mnt/huawei_deepcad/dinov3")
OUT = REPO / "plot/fig2/data_quality"
SAMPLE = Path("/mnt/huawei_blm/deepcad_20tb_route1_quality_1m_20261006")
PYTHON = "/home/bbnc/anaconda3/envs/dinov3/bin/python"
PACK_PYTHON = "/home/inspur/anaconda3/envs/pipeline_env/bin/python"
PACKER = "/mnt/deepcad_nfs/deepcad_100t/final-data/projection_tools/shuffle_selected_patches.py"
ARM = "20tb_route1_ddp"
GPU_GROUP = "0,1,2,3"


def save(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def status(state: str, **fields: object) -> None:
    save(OUT / "20tb_route1_controller_status.json",
         dict(state=state, time_unix=time.time(), **fields))


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_workers(directory: Path, session_prefix: str, count: int, label: str) -> list[dict]:
    retries = [0] * count
    while True:
        states = []
        for rank in range(count):
            path = directory / f"shuffle_state_rank{rank:03d}.json"
            states.append(read(path) if path.exists() else {})
        complete = sum(item.get("completed") is True for item in states)
        packed = sum(item.get("patch_count", 0) for item in states)
        skipped = sum(item.get("skipped_sources", 0) for item in states)
        status(label, completed_ranks=complete, expected_ranks=count,
               packed_images=packed, skipped_sources=skipped)
        if complete == count:
            return states
        dead = [rank for rank in range(count) if not states[rank].get("completed") and
                subprocess.run(["tmux", "has-session", "-t", f"{session_prefix}_{rank:02d}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0]
        for rank in dead:
            if retries[rank] >= 3:
                status("PACK_FAILED", phase=label, dead_ranks=dead, packed_images=packed,
                       retries=retries)
                raise RuntimeError(f"Materialization worker failed after three retries: {rank}")
            retries[rank] += 1
            backup = directory.name == "backfill_packed"
            partitions = SAMPLE / ("backfill_partitions" if backup else "partitions")
            log = SAMPLE / (f"backfill_worker_{rank:02d}.log" if backup else f"worker_{rank:02d}.log")
            command = (f"{PACK_PYTHON} -u {PACKER} worker --partitions {partitions} "
                       f"--output {directory} --world-size {count} --rank {rank} --resume "
                       f"--source-timeout 600 --tar-bytes 2000000000 >> {log} 2>&1")
            subprocess.run(["tmux", "new-session", "-d", "-s", f"{session_prefix}_{rank:02d}",
                            "--", "bash", "-lc", command], check=True)
            status("WORKER_RESTARTED", phase=label, rank=rank, attempt=retries[rank],
                   packed_images=packed)
        time.sleep(180)


def member_key(name: str) -> str:
    if name.endswith(".meta.json"):
        return name[:-10]
    if ".ch" in name and name.endswith(".tif"):
        return name.rsplit(".ch", 1)[0]
    raise ValueError(f"Unexpected backup tar member: {name}")


def finalize_backfill(missing: int) -> tuple[Path, int]:
    states = wait_for_workers(SAMPLE / "backfill_packed", "dq20r1_backfill", 4, "BACKFILL_PACKING")
    backfill = read(SAMPLE / "backfill_selection.json")
    initial = read(SAMPLE / "selection.json")
    if backfill["initial_manifest_sha256"] != initial["selected_manifest_sha256"] or \
            backfill["source_manifest_sha256"] != initial["source_manifest_sha256"]:
        raise ValueError("Backfill does not come from the same source pool")
    if sha256(SAMPLE / "backfill_selected.parquet") != backfill["backfill_manifest_sha256"]:
        raise ValueError("Backfill manifest changed after sampling")
    tars = sorted((SAMPLE / "backfill_packed").glob("filtered_projection_20TB_nested-r*.tar"))
    if len(tars) != sum(item["tar_index"] for item in states):
        raise ValueError("Backfill tar count differs from completed worker states")
    successful = []
    for path in tars:
        with tarfile.open(path, "r") as archive:
            for member in archive:
                if member.name.endswith(".meta.json"):
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Unreadable backup metadata: {path}:{member.name}")
                    meta = json.load(stream)
                    successful.append((int(meta["backfill_priority"]), str(meta["sample_id"])))
    if len(successful) != sum(item["patch_count"] for item in states) or \
            len({priority for priority, _ in successful}) != len(successful):
        raise ValueError("Backup success inventory is inconsistent")
    if len(successful) < missing:
        raise RuntimeError(f"Only {len(successful)} readable reserve samples for {missing} missing")
    chosen = {sample_id for _, sample_id in sorted(successful)[:missing]}
    final_dir = SAMPLE / "backfill_final"
    final_dir.mkdir(exist_ok=True)
    final = final_dir / "filtered_projection_20TB_nested-r900-000000.tar"
    temporary = final.with_suffix(".tar.part")
    if final.exists() or temporary.exists():
        raise FileExistsError(final)
    written = set()
    with tarfile.open(temporary, "w") as destination:
        for path in tars:
            with tarfile.open(path, "r") as archive:
                for member in archive:
                    if member_key(member.name) not in chosen:
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Unreadable backup member: {path}:{member.name}")
                    renamed = copy.copy(member)
                    renamed.name = f"backfill__{member.name}"
                    if member.name.endswith(".meta.json"):
                        metadata = json.load(stream)
                        metadata["source_sample_id"] = metadata["sample_id"]
                        metadata["sample_id"] = f"backfill__{metadata['sample_id']}"
                        payload = json.dumps(metadata, separators=(",", ":")).encode()
                        renamed.size = len(payload)
                        destination.addfile(renamed, io.BytesIO(payload))
                        written.add(member_key(member.name))
                    else:
                        destination.addfile(renamed, stream)
    if written != chosen:
        raise ValueError(f"Backup tar contains {len(written)}/{missing} chosen samples")
    os.replace(temporary, final)
    return final, backfill["reserve_size"] - len(successful)


def wait_for_sample() -> None:
    selected = read(SAMPLE / "selection.json")
    if selected["status"] != "PASS" or selected["sample_size"] != 1_000_000:
        raise ValueError("Route-1 selection is not one million records")
    if sha256(Path(selected["source_manifest"])) != selected["source_manifest_sha256"] or \
            sha256(Path(selected["selected_manifest"])) != selected["selected_manifest_sha256"]:
        raise ValueError("Route-1 source or selected manifest changed after sampling")
    states = wait_for_workers(SAMPLE / "packed", "dq20r1_pack", 8, "PACKING")
    packed = sum(item["patch_count"] for item in states)
    missing = 1_000_000 - packed
    if missing < 0:
        raise ValueError("More than one million primary samples were packed")
    backfill_tar = None
    backfill_skips = 0
    if missing:
        status("READ_REPAIR", primary_packed=packed, missing=missing)
        backfill_tar, backfill_skips = finalize_backfill(missing)
    shards = sorted((SAMPLE / "packed").glob("filtered_projection_20TB_nested-r*.tar"))
    if len(shards) != sum(item["tar_index"] for item in states):
        raise ValueError("Primary tar count differs from completed worker states")
    if backfill_tar is not None:
        shards.append(backfill_tar)
    if len(shards) < 4 or any(path.stat().st_size == 0 for path in shards):
        raise RuntimeError("Missing or empty materialized tar files")
    save(SAMPLE / "extraction_complete.json", dict(
        status="PASS", samples=1_000_000, shards=len(shards),
        bytes=sum(path.stat().st_size for path in shards),
        source_manifest_sha256=selected["source_manifest_sha256"],
        selected_manifest_sha256=selected["selected_manifest_sha256"],
        primary_read_failures=missing, backfill_read_failures=backfill_skips,
        sampling_claim="uniform among readable candidates under fixed read outcomes",
        completed_unix=time.time()))
    status("EXTRACTED", samples=1_000_000, shards=len(shards), primary_read_failures=missing)


def wait_for_gpus() -> None:
    while True:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True)
        rows = [[int(field.strip().split()[0]) for field in line.split(",")]
                for line in output.splitlines()]
        ready = all(any(index == gpu and free >= 20000 and util <= 20
                        for index, free, util in rows) for gpu in range(4))
        if ready:
            return
        status("WAIT_GPU", gpu_status=rows)
        time.sleep(180)


def run_logged(command: list[str], path: Path) -> None:
    with path.open("w") as stream:
        result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        status("ERROR", command=command, returncode=result.returncode, log=str(path))
        raise RuntimeError(f"Command failed rc={result.returncode}; inspect {path}")


def train_eval() -> None:
    if read(SAMPLE / "extraction_complete.json")["status"] != "PASS":
        raise ValueError("Materialization was not audited")
    run = OUT / "training" / ARM
    if not run.exists():
        assignment = read(SAMPLE / "rank_shard_assignment.json")
        if assignment["status"] != "PASS" or min(assignment["rank_sample_counts"]) < 249_856:
            raise ValueError("Route-1 DDP shard assignment is not ready")
        wait_for_gpus()
        smoke = OUT / "smoke" / ARM
        if not smoke.exists():
            status("SMOKE")
            run_logged([PYTHON, "-u", str(REPO / "scripts/launch_data_quality_1m_20261002.py"),
                        "--arm", ARM, "--gpu-group", GPU_GROUP, "--master-port", "29641",
                        "--smoke"], OUT / "20tb_route1_smoke_console.log")
        if read(smoke / "exit.json").get("returncode") != 0:
            raise RuntimeError("Route-1 smoke training did not complete successfully")
        status("TRAINING")
        run_logged([PYTHON, "-u", str(REPO / "scripts/launch_data_quality_1m_20261002.py"),
                    "--arm", ARM, "--gpu-group", GPU_GROUP, "--master-port", "29642"],
                   OUT / "20tb_route1_train_console.log")
    if read(run / "exit.json").get("returncode") != 0:
        raise RuntimeError("Route-1 training did not complete successfully")
    status("AUDITING")
    run_logged([PYTHON, "-u", str(REPO / "scripts/audit_data_quality_1m_20261002.py"),
                "--arm", ARM], OUT / "20tb_route1_audit_console.log")
    status("PREPARING_EVAL")
    if not (OUT / "eval" / ARM / "prepared.json").exists():
        run_logged([PYTHON, "-u", str(REPO / "scripts/prepare_data_quality_v4_id_20261002.py"),
                    "--arm", ARM], OUT / "20tb_route1_prepare_console.log")
    status("EVALUATING")
    run_logged([PYTHON, "-u", str(REPO / "scripts/run_data_quality_v4_id_20261002.py"),
                "--arm", ARM, "--gpus", "0", "1", "2", "3"],
               OUT / "20tb_route1_eval_console.log")
    route2_status = OUT / "eval/20tb_route2_ddp/driver_status.json"
    while not route2_status.is_file() or read(route2_status).get("state") != "COMPLETE":
        status("WAIT_ROUTE2_EVAL", route2_status=str(route2_status))
        time.sleep(180)
    status("COLLECTING")
    run_logged([PYTHON, "-u", str(REPO / "scripts/collect_data_quality_v4_id_20261002.py"),
                "--arms", "1tb", "5tb", "20tb_route2_ddp", ARM, "100tb", "1pb",
                "--stem", "v4_id_two20tb"], OUT / "20tb_route1_collect_console.log")
    status("PLOTTING")
    run_logged([PYTHON, "-u", str(OUT / "plot_data_quality_v4_id_two20tb.py")],
               OUT / "20tb_route1_plot_console.log")
    status("COMPLETE", training_audit=str(run / "audit.json"),
           evaluation_status=str(OUT / "eval" / ARM / "driver_status.json"),
           figure=str(OUT / "data_quality_v4_id_matched_1m_two20tb.svg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("orchestrate", "train-eval"), required=True)
    args = parser.parse_args()
    if args.mode == "train-eval":
        train_eval()
        return
    wait_for_sample()
    command = (f"tmux new-session -d -s dq20r1_train_eval_20261006 -- "
               f"{PYTHON} -u {REPO}/scripts/continue_data_quality_20tb_route1_20261006.py "
               "--mode train-eval")
    subprocess.run(["ssh", "-o", "BatchMode=yes", "3090-qi", command], check=True)
    status("REMOTE_CONTROLLER_STARTED", host="3090-qi", session="dq20r1_train_eval_20261006")


if __name__ == "__main__":
    main()
