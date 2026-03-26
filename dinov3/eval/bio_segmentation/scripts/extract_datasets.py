"""
Dataset extraction script for bio-segmentation benchmarks.

Extracts the following datasets from zip/tar archives:
  BBBC038, CoNIC, LIVECell, MoNuSeg, PanNuke, TissueNet

Two typical usage patterns are supported:

  Pattern A – point --src-dir to the dataset-specific folder directly:
    python -m dinov3.eval.bio_segmentation.scripts.extract_datasets \\
        --dataset bbbc038 \\
        --src-dir /data1/xuzijing/dataset/BBBC038 \\
        --dst-dir /data1/xuzijing/dataset

  Pattern B – point --src-dir to a common pool of archives and let the
  keyword filter find the right files:
    python -m dinov3.eval.bio_segmentation.scripts.extract_datasets \\
        --src-dir /data1/xuzijing/dataset \\
        --datasets bbbc038 conic livecell monuseg pannuke tissuenet \\
        --dst-dir /data1/xuzijing/dataset

When no keyword-matching archive is found the script falls back to
extracting EVERY archive (.zip / .tar.*) inside --src-dir, so Pattern A
always works regardless of how the archive files are named.
"""

import argparse
import logging
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('extract_datasets')


# -----------------------------------------------------------------------
# Name-to-expected-archive-prefix mapping (case-insensitive substring match)
# These are best-effort hints; if none match the script falls back to
# extracting all archives in src_dir.
# -----------------------------------------------------------------------
DATASET_PATTERNS = {
    'bbbc038':   ['bbbc038', 'dsb2018', 'data-science-bowl-2018',
                  'stage1', 'stage2'],          # stage1_train.zip etc.
    'conic':     ['conic', 'colon_nuclei',
                  'patch', 'slide'],
    'livecell':  ['livecell'],
    'monuseg':   ['monuseg', 'monuseg',          # handles both cases
                  'training', 'testdata'],
    'pannuke':   ['pannuke', 'pan_nuke',
                  'fold', 'part'],               # fold1.zip / Part1.zip
    'tissuenet': ['tissuenet', 'tissue_net'],
}

# Archive suffixes recognised as extractable
_ARCHIVE_SUFFIXES = {'.zip', '.gz', '.bz2', '.xz', '.tgz'}


def _is_archive(path: Path) -> bool:
    """Return True if the file looks like a zip/tar archive."""
    if path.suffix.lower() in _ARCHIVE_SUFFIXES:
        return True
    if len(path.suffixes) >= 2 and path.suffixes[-2].lower() == '.tar':
        return True
    return False


def _find_archives(src_dir: Path, keywords: list) -> list:
    """Return archives whose filename contains at least one keyword (case-insensitive)."""
    archives = []
    for path in sorted(src_dir.iterdir()):
        if not _is_archive(path):
            continue
        low = path.name.lower()
        if any(kw.lower() in low for kw in keywords):
            archives.append(path)
    return archives


def _find_all_archives(src_dir: Path) -> list:
    """Return ALL archives in src_dir (fallback when no keyword matches)."""
    return [p for p in sorted(src_dir.iterdir()) if _is_archive(p)]


def _extract_zip(archive: Path, dst: Path):
    with zipfile.ZipFile(archive, 'r') as zf:
        zf.extractall(dst)
    logger.info(f"  Extracted ZIP → {dst}")


def _extract_tar(archive: Path, dst: Path):
    with tarfile.open(archive, 'r:*') as tf:
        tf.extractall(dst)
    logger.info(f"  Extracted TAR → {dst}")


# Data-file extensions that indicate an already-extracted dataset
_DATA_EXTENSIONS = {'.npy', '.npz', '.csv', '.json', '.xml', '.tif', '.tiff',
                    '.png', '.jpg', '.h5', '.hdf5', '.txt', '.md', '.yaml', '.yml'}

# Small-archive threshold: skip archives below this size (e.g. script-only zips)
_MIN_ARCHIVE_BYTES = 50_000   # 50 KB


def _is_data_file(path: Path) -> bool:
    """Return True if path is a data/annotation file (not an archive or hidden)."""
    return (path.is_file()
            and not path.name.startswith('.')
            and path.suffix.lower() in _DATA_EXTENSIONS)


def _copy_data_files(src_dir: Path, target: Path):
    """Copy all data files from *src_dir* (non-recursive) into *target*."""
    copied = 0
    for p in sorted(src_dir.iterdir()):
        if _is_data_file(p):
            dst = target / p.name
            if not dst.exists():
                shutil.copy2(str(p), str(dst))
            copied += 1
    return copied


def extract_dataset(name: str, src_dir: Path, dst_dir: Path, overwrite: bool = False):
    """
    Find and extract (or copy) dataset files for *name* from *src_dir* into
    *dst_dir/<name>/extracted/*.

    Special cases handled:
      - CoNIC / array datasets: data is already unpacked as .npy/.csv files
        → copy directly (no archive extraction).
      - LIVECell: images.zip is extracted IN-PLACE into src_dir so that
        annotations and images share the same directory tree.
      - TissueNet: the tiny 'tissuenet.zip' (scripts only) is skipped;
        only archives ≥ 50 KB are extracted.

    General search order:
      1. Keyword match against filenames.
      2. Fallback → all archives ≥ 50 KB in src_dir.
      3. Fallback → copy data files directly (already-extracted datasets).
    """
    target = dst_dir / name / 'extracted'

    # ------------------------------------------------------------------ #
    # LIVECell special case: extract images in-place so JSON + images     #
    # share the same directory tree for easy lookup.                       #
    # ------------------------------------------------------------------ #
    if name == 'livecell':
        # LIVECell stores everything under a nested subdirectory
        # (e.g. LIVECell_dataset_2021/).  Search recursively for images.zip
        # so the script works whether --src-dir is the outer folder or the
        # inner one.
        all_zips = sorted(src_dir.rglob('images.zip'))
        img_archives = [a for a in all_zips if a.stat().st_size >= _MIN_ARCHIVE_BYTES]

        if not img_archives:
            # Check if images/ is already extracted somewhere beneath src_dir
            already_imgs = list(src_dir.rglob('livecell_train_val_images'))
            if already_imgs:
                nested_root = already_imgs[0].parent.parent  # …/images/..  → go up twice
                # Actually we want the parent of images/
                nested_root = already_imgs[0].parent.parent
                logger.info(
                    "[livecell] images/ already extracted at %s — no action needed.",
                    already_imgs[0].parent,
                )
                logger.info(
                    "[livecell] Use --data-root %s when running experiments.",
                    nested_root.parent,
                )
            else:
                logger.warning(
                    "[livecell] Neither images.zip nor extracted images/ found under %s.\n"
                    "  Try pointing --src-dir to the LIVECell_dataset_2021 subdirectory.",
                    src_dir,
                )
        else:
            for arch in img_archives:
                extract_here = arch.parent   # extract in-place next to the zip
                already_done = extract_here / 'images'
                if already_done.exists() and not overwrite:
                    logger.info("[livecell] images/ already exists at %s — skipping.", already_done)
                    continue
                logger.info("[livecell] Extracting %s → %s", arch.name, extract_here)
                if arch.suffix.lower() == '.zip':
                    _extract_zip(arch, extract_here)
                else:
                    _extract_tar(arch, extract_here)
                logger.info("[livecell] Done.  Use --data-root %s when running experiments.",
                            extract_here.parent)
        return

    # ------------------------------------------------------------------ #
    # BBBC038 special case: extract each zip to its own named sub-dir.  #
    # stage1_train.zip → extracted/stage1_train/                        #
    # stage1_test.zip  → extracted/stage1_test/   (no GT, infer only)  #
    # stage2_test_final.zip → extracted/stage2_test_final/  (optional) #
    # This keeps train/test samples separate so get_bbbc038_paths can   #
    # find them reliably.                                                #
    # ------------------------------------------------------------------ #
    if name == 'bbbc038':
        keywords = DATASET_PATTERNS.get('bbbc038', ['stage1'])
        archives = [a for a in _find_archives(src_dir, keywords)
                    if a.stat().st_size >= _MIN_ARCHIVE_BYTES]
        if not archives:
            archives = [a for a in _find_all_archives(src_dir)
                        if a.stat().st_size >= _MIN_ARCHIVE_BYTES]
        if not archives:
            logger.error("[bbbc038] No archives found in %s", src_dir)
            return
        for arch in archives:
            # Derive subdirectory name from zip stem
            # e.g. stage1_train.zip → extracted/stage1_train/
            sub_name = arch.stem.replace('-', '_')   # normalise hyphens
            sub_dst  = target / sub_name
            if sub_dst.exists() and not overwrite:
                logger.info("[bbbc038] %s already extracted at %s", arch.name, sub_dst)
                continue
            sub_dst.mkdir(parents=True, exist_ok=True)
            logger.info("[bbbc038] Extracting %s → %s", arch.name, sub_dst)
            if arch.suffix.lower() == '.zip':
                _extract_zip(arch, sub_dst)
            else:
                _extract_tar(arch, sub_dst)
        logger.info("[bbbc038] Done.  Output: %s", target)
        return

    # ------------------------------------------------------------------ #
    # Standard path                                                        #
    # ------------------------------------------------------------------ #
    if target.exists() and not overwrite:
        logger.info(
            f"[{name}] Already extracted at {target}, skipping.  "
            f"Use --overwrite to re-extract."
        )
        return

    target.mkdir(parents=True, exist_ok=True)
    keywords = DATASET_PATTERNS.get(name, [name])

    # --- Step 1: keyword-based archive search ---
    archives = [a for a in _find_archives(src_dir, keywords)
                if a.stat().st_size >= _MIN_ARCHIVE_BYTES]

    # --- Step 2: fallback → all large-enough archives in src_dir ---
    if not archives:
        fallback = [a for a in _find_all_archives(src_dir)
                    if a.stat().st_size >= _MIN_ARCHIVE_BYTES]
        if fallback:
            logger.info(
                f"[{name}] No keyword-matched archives (keywords: {keywords}).  "
                f"Falling back to all {len(fallback)} archive(s) in {src_dir}."
            )
            archives = fallback

    # --- Step 3: fallback → copy already-unpacked data files ---
    if not archives:
        n_copied = _copy_data_files(src_dir, target)
        if n_copied > 0:
            logger.info(
                f"[{name}] No archives found but detected {n_copied} data file(s) in "
                f"{src_dir} — copied directly to {target}."
            )
            logger.info(f"[{name}] Done.  Output: {target}")
        else:
            # Try keyword-matched sub-directories
            for kw in keywords:
                for p in src_dir.iterdir():
                    if kw.lower() in p.name.lower() and p.is_dir():
                        shutil.copytree(str(p), str(target / p.name), dirs_exist_ok=True)
                        logger.info(f"[{name}] Copied directory {p} → {target}/{p.name}")
                        return
            logger.error(
                f"[{name}] Nothing found for '{name}' in {src_dir}.  "
                f"Please verify the dataset is present."
            )
        return

    for arch in archives:
        logger.info(f"[{name}] Extracting {arch.name} → {target}")
        if arch.suffix.lower() == '.zip':
            _extract_zip(arch, target)
        else:
            _extract_tar(arch, target)

    logger.info(f"[{name}] Done.  Output: {target}")


def main():
    parser = argparse.ArgumentParser(description='Extract bio-segmentation datasets')
    parser.add_argument('--src-dir', required=True,
                        help='Directory containing the downloaded zip/tar archives')
    parser.add_argument('--dst-dir', default='/data1/xuzijing/dataset',
                        help='Root destination directory (default: /data1/xuzijing/dataset)')
    parser.add_argument('--datasets', nargs='+',
                        default=list(DATASET_PATTERNS.keys()),
                        choices=list(DATASET_PATTERNS.keys()),
                        help='Datasets to extract (default: all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-extract even if destination already exists')
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    if not src_dir.exists():
        parser.error(f"--src-dir does not exist: {src_dir}")

    for name in args.datasets:
        extract_dataset(name, src_dir, dst_dir, overwrite=args.overwrite)

    logger.info("All done.")


if __name__ == '__main__':
    main()
