#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试索引生成器 — 从 dedupIndex_100t 中采样少量条目。

遍历 dedupIndex_100t 下所有子文件夹（ori、slfmandhighna），
从每个 .txt 文件中仅取前 N 行，保持完全相同的目录结构保存到
dedupIndex_100t_test 目录，供后续 test_extraction.py 使用。

用法:
    cd /mnt/huawei_deepcad/dinov3
    python dinov3/dataset_webdataset/test/test_subset_generator.py
    python dinov3/dataset_webdataset/test/test_subset_generator.py --lines 10
"""

import argparse
import random
import shutil
from pathlib import Path

ORIGINAL_ROOT = "/mnt/huawei_deepcad/ssl-data-curation/dedupIndex_100t"
TEST_ROOT = "/mnt/deepcad_nfs/ssl-data-curation/dedupIndex_100t_test"
DEFAULT_LINES_PER_FILE = 5


def generate_test_index(
    src_root: Path,
    dst_root: Path,
    lines_per_file: int,
    mode: str,
    seed: int,
) -> None:
    """
    遍历源目录中的所有 .txt 文件，取前 N 行写入目标目录。

    Args:
        src_root: 原始索引根目录。
        dst_root: 输出的测试索引根目录。
        lines_per_file: 每个 .txt 保留的行数。
    """
    if dst_root.exists():
        print(f"⚠️  目标目录已存在，将清空: {dst_root}")
        shutil.rmtree(dst_root)

    total_files = 0
    total_lines = 0

    for txt_path in sorted(src_root.rglob("*.txt")):
        rel = txt_path.relative_to(src_root)
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with open(txt_path, "r", encoding="utf-8") as fin:
            all_lines = fin.readlines()

        if mode == "head":
            sampled_lines = all_lines[:lines_per_file]
        else:
            # 使用“路径 + 全局种子”保证单文件随机可复现
            # 这样每次运行（同 seed）都会得到相同子集，便于 debug 对比
            local_rng = random.Random(f"{seed}:{rel.as_posix()}")
            if len(all_lines) <= lines_per_file:
                sampled_lines = all_lines
            else:
                idxs = sorted(local_rng.sample(range(len(all_lines)), lines_per_file))
                sampled_lines = [all_lines[i] for i in idxs]

        # 写入
        with open(dst_path, "w", encoding="utf-8") as fout:
            fout.writelines(sampled_lines)

        total_files += 1
        total_lines += len(sampled_lines)
        print(f"  {rel}: {len(sampled_lines)} 行")

    print(f"\n✅ 完成: {total_files} 个文件, 共 {total_lines} 行")
    print(f"   输出到: {dst_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 dedupIndex_100t 采样生成测试索引 dedupIndex_100t_test"
    )
    parser.add_argument(
        "--lines", type=int, default=DEFAULT_LINES_PER_FILE,
        help=f"每个 .txt 保留的行数 (默认 {DEFAULT_LINES_PER_FILE})",
    )
    parser.add_argument(
        "--mode", type=str, default="random", choices=["head", "random"],
        help="采样方式: head=按顺序取前N行, random=随机抽N行 (默认 head)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260306,
        help="随机种子，仅在 --mode random 下生效 (默认 20260306)",
    )
    parser.add_argument(
        "--src", type=str, default=ORIGINAL_ROOT,
        help=f"原始索引根目录 (默认 {ORIGINAL_ROOT})",
    )
    parser.add_argument(
        "--dst", type=str, default=TEST_ROOT,
        help=f"输出目录 (默认 {TEST_ROOT})",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"❌ 源目录不存在: {src_root}")
        return

    print(f"源目录: {src_root}")
    print(f"目标目录: {dst_root}")
    print(f"每文件保留: {args.lines} 行")
    print(f"采样模式: {args.mode}")
    if args.mode == "random":
        print(f"随机种子: {args.seed}")
    print()

    generate_test_index(src_root, dst_root, args.lines, args.mode, args.seed)


if __name__ == "__main__":
    main()

