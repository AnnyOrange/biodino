#!/usr/bin/env python
"""Build compressed copies of the 1TB WebDataset, preserving sample identity.

Reads every source shard once and writes two output shards in lockstep:

  1. deflate_lossless/ : each .tif re-encoded as the SAME dtype (uint16) with
     TIFF-internal DEFLATE. Pixels are bit-identical to the source -> no
     training needed, this version only changes on-disk size.

  2. uint8_deflate/    : each .tif cast uint16 -> uint8 with a single GLOBAL
     full-range map  v8 = round(v16 * 255 / 65535)  then DEFLATE. This is
     LOSSY (drops the low 8 bits of precision). The global map keeps the
     training-time normalisation consistent (loader divides uint8 by 255,
     which matches uint16/65535), so rgb_mean/std need no change.

Non-.tif members (.meta.json, etc.) are copied verbatim. Member names and
archive order are preserved exactly, so each output shard contains the same
samples in the same order as its source -> Raw vs compressed differ ONLY in
TIFF encoding, which is the precondition for an attributable benchmark delta.

Run with the dinov3 env (needs tifffile + imagecodecs):

    /home/lxy/miniconda3/envs/dinov3/bin/python \
        docs/data_compression/build_compressed_datasets.py \
        --src-dir /mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle \
        --out-root /mnt/huawei_deepcad/compression \
        --workers 32 --verify

Resumable: a shard whose final output exists in BOTH versions is skipped.
Output is written to a .tmp file and renamed on completion, so an interrupted
run never leaves a half-written shard that looks complete.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile

TIF_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")
UINT16_MAX = 65535.0


def encode_tiff(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    tifffile.imwrite(buf, array, compression="deflate")
    return buf.getvalue()


def quantize_uint8_global(array: np.ndarray) -> np.ndarray:
    """uint16 -> uint8 via fixed global full-range map (same map for every image)."""
    if array.dtype == np.uint8:
        return array
    a = array.astype(np.float32) * (255.0 / UINT16_MAX)
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def _add_bytes(tar: tarfile.TarFile, member: tarfile.TarInfo, data: bytes) -> None:
    ti = tarfile.TarInfo(name=member.name)
    ti.size = len(data)
    ti.mode = member.mode
    ti.mtime = member.mtime
    ti.uid = member.uid
    ti.gid = member.gid
    ti.uname = member.uname
    ti.gname = member.gname
    ti.type = member.type
    tar.addfile(ti, io.BytesIO(data))


def process_shard(src_path_s: str, out_deflate_s: str | None, out_uint8_s: str | None,
                  verify: bool) -> dict:
    src_path = Path(src_path_s)
    out_deflate = Path(out_deflate_s) if out_deflate_s else None
    out_uint8 = Path(out_uint8_s) if out_uint8_s else None

    if (out_deflate is None or out_deflate.exists()) and (out_uint8 is None or out_uint8.exists()):
        return {"shard": src_path.name, "status": "skip"}

    tmp_d = out_deflate.with_suffix(".tar.tmp") if out_deflate is not None else None
    tmp_u = out_uint8.with_suffix(".tar.tmp") if out_uint8 is not None else None

    stats = {
        "shard": src_path.name, "status": "ok",
        "n_tif": 0, "n_other": 0, "n_decode_fail": 0, "n_verify_fail": 0,
        "src_bytes": 0, "deflate_bytes": 0, "uint8_bytes": 0,
    }
    verified = 0

    try:
        with tarfile.open(src_path, "r") as tin, contextlib.ExitStack() as stack:
            td = stack.enter_context(tarfile.open(tmp_d, "w")) if tmp_d is not None else None
            tu = stack.enter_context(tarfile.open(tmp_u, "w")) if tmp_u is not None else None
            for member in tin:
                if not member.isfile():
                    continue
                data = tin.extractfile(member).read()
                stats["src_bytes"] += len(data)

                if member.name.endswith(TIF_SUFFIXES):
                    stats["n_tif"] += 1
                    try:
                        arr = tifffile.imread(io.BytesIO(data))
                    except Exception:
                        # keep the sample intact: copy original bytes verbatim
                        stats["n_decode_fail"] += 1
                        if td is not None:
                            _add_bytes(td, member, data)
                        if tu is not None:
                            _add_bytes(tu, member, data)
                        continue

                    db = encode_tiff(arr) if td is not None else None  # lossless, same dtype
                    ub = encode_tiff(quantize_uint8_global(arr)) if tu is not None else None  # lossy uint8

                    if verify and db is not None and verified < 50:
                        back = tifffile.imread(io.BytesIO(db))
                        if not np.array_equal(back, arr):
                            stats["n_verify_fail"] += 1
                        verified += 1

                    if db is not None and td is not None:
                        stats["deflate_bytes"] += len(db)
                        _add_bytes(td, member, db)
                    if ub is not None and tu is not None:
                        stats["uint8_bytes"] += len(ub)
                        _add_bytes(tu, member, ub)
                else:
                    stats["n_other"] += 1
                    if td is not None:
                        _add_bytes(td, member, data)
                    if tu is not None:
                        _add_bytes(tu, member, data)

        if tmp_d is not None and out_deflate is not None:
            tmp_d.replace(out_deflate)
        if tmp_u is not None and out_uint8 is not None:
            tmp_u.replace(out_uint8)
    except Exception as exc:  # noqa: BLE001
        for tmp in (tmp_d, tmp_u):
            if tmp is not None and tmp.exists():
                tmp.unlink()
        stats["status"] = f"ERROR: {type(exc).__name__}: {exc}"

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--glob", default="filtered_mixed_train_w*-*.tar")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument(
        "--mode",
        choices=("both", "deflate", "uint8"),
        default="both",
        help="which compressed copy to build; default preserves old behavior",
    )
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="process only the first N shards (testing)")
    ap.add_argument("--verify", action="store_true",
                    help="check deflate lossless on up to 50 tifs/shard")
    args = ap.parse_args()

    shards = sorted(p for p in args.src_dir.glob(args.glob) if p.is_file())
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if not shards:
        print(f"no shards matching {args.glob} in {args.src_dir}", file=sys.stderr)
        return 1

    out_deflate_dir = args.out_root / "deflate_lossless" if args.mode in ("both", "deflate") else None
    out_uint8_dir = args.out_root / "uint8_deflate" if args.mode in ("both", "uint8") else None
    if out_deflate_dir is not None:
        out_deflate_dir.mkdir(parents=True, exist_ok=True)
    if out_uint8_dir is not None:
        out_uint8_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (str(sp),
         str(out_deflate_dir / sp.name) if out_deflate_dir is not None else None,
         str(out_uint8_dir / sp.name) if out_uint8_dir is not None else None,
         args.verify)
        for sp in shards
    ]
    print(f"{len(jobs)} shards, {args.workers} workers, mode={args.mode}", file=sys.stderr)

    t0 = time.time()
    tot = {"n_tif": 0, "n_other": 0, "n_decode_fail": 0, "n_verify_fail": 0,
           "src_bytes": 0, "deflate_bytes": 0, "uint8_bytes": 0,
           "ok": 0, "skip": 0, "err": 0}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_shard, *j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            st = r["status"]
            if st == "ok":
                tot["ok"] += 1
                for k in ("n_tif", "n_other", "n_decode_fail", "n_verify_fail",
                          "src_bytes", "deflate_bytes", "uint8_bytes"):
                    tot[k] += r[k]
            elif st == "skip":
                tot["skip"] += 1
            else:
                tot["err"] += 1
                print(f"  !! {r['shard']}: {st}", file=sys.stderr)
            if done % 10 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"[{done}/{len(jobs)}] ok={tot['ok']} skip={tot['skip']} "
                      f"err={tot['err']} elapsed={el:.0f}s", file=sys.stderr)

    el = time.time() - t0
    sb, dbb, ub = tot["src_bytes"], tot["deflate_bytes"], tot["uint8_bytes"]
    print("\n===== SUMMARY =====")
    print(f"shards: ok={tot['ok']} skip={tot['skip']} err={tot['err']}")
    print(f"tif={tot['n_tif']} other={tot['n_other']} "
          f"decode_fail={tot['n_decode_fail']} verify_fail={tot['n_verify_fail']}")
    if sb:
        print(f"source tif payload : {sb/1e9:.1f} GB")
        if args.mode in ("both", "deflate"):
            print(f"deflate (lossless) : {dbb/1e9:.1f} GB ({100*dbb/sb:.1f}%)")
        if args.mode in ("both", "uint8"):
            print(f"uint8 + deflate    : {ub/1e9:.1f} GB ({100*ub/sb:.1f}%)")
    print(f"elapsed: {el:.0f}s")
    if tot["n_verify_fail"]:
        print("WARNING: lossless verify FAILED on some tifs", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
