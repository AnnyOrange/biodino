"""CTC mask matching using observed overlaps instead of a dense label square."""
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile


def match_arrays(reference, computed, chunk_size=262144):
    """Match the pinned implementation's ordering, >50% rule and IoUs.

    Preserve its first-label/background convention, including masks without
    background. No matrix proportional to the square of the label count exists.
    """
    return _match_chunks([(reference, computed)], chunk_size)


def _match_chunks(chunks, chunk_size=262144):
    refs, comps, overlaps = Counter(), Counter(), Counter()
    for reference, computed in chunks:
        if reference.shape != computed.shape:
            raise ValueError(f'Mask shape mismatch: {reference.shape} != {computed.shape}')
        if reference.dtype.kind not in 'ui' or computed.dtype.kind not in 'ui':
            raise ValueError('CTC masks must have nonnegative integer labels')
        left, right = reference.reshape(-1), computed.reshape(-1)
        for start in range(0, left.size, chunk_size):
            a, b = left[start:start + chunk_size], right[start:start + chunk_size]
            if a.size and (a.min() < 0 or b.min() < 0 or a.max() > 0xffffffff or b.max() > 0xffffffff):
                raise ValueError('CTC mask label outside uint32 range')
            labels, counts = np.unique(a, return_counts=True)
            refs.update(dict(zip(map(int, labels), map(int, counts))))
            labels, counts = np.unique(b, return_counts=True)
            comps.update(dict(zip(map(int, labels), map(int, counts))))
            foreground = (a != 0) & (b != 0)
            packed = (a[foreground].astype(np.uint64) << 32) | b[foreground].astype(np.uint64)
            labels, counts = np.unique(packed, return_counts=True)
            overlaps.update(dict(zip(map(int, labels), map(int, counts))))
    labels_ref, labels_comp = sorted(refs)[1:], sorted(comps)[1:]
    allowed_ref, allowed_comp = set(labels_ref), set(labels_comp)
    mapped_ref, mapped_comp, ious = [], [], []
    for pair, intersection in sorted(overlaps.items()):
        r, c = pair >> 32, pair & 0xffffffff
        if r in allowed_ref and c in allowed_comp and intersection > .5 * refs[r]:
            mapped_ref.append(r)
            mapped_comp.append(c)
            ious.append(intersection / (refs[r] + comps[c] - intersection))
    if len(set(mapped_ref)) != len(mapped_ref):
        raise ValueError('Reference node assigned to multiple computed nodes')
    return labels_ref, labels_comp, mapped_ref, mapped_comp, ious


def match_paths(ref_path, comp_path):
    """Support the pinned evaluator's full masks and 3-D SEG slice naming."""
    def read(path):
        return tifffile.imread(path, maxworkers=1)

    if comp_path is None:
        labels = np.unique(read(ref_path))
        return labels[labels != 0].tolist(), [], [], [], []
    if ref_path is None:
        ref_path = comp_path
    if Path(ref_path).exists():
        if ref_path != comp_path:
            with tifffile.TiffFile(ref_path) as left, tifffile.TiffFile(comp_path) as right:
                if len(left.pages) == len(right.pages) and len(left.pages) > 1:
                    return _match_chunks((a.asarray(maxworkers=1), b.asarray(maxworkers=1))
                                         for a, b in zip(left.pages, right.pages))
        reference = read(ref_path)
        if ref_path == comp_path:
            labels = np.unique(reference)
            positive = labels[labels != 0].tolist()
            # Preserve the pinned trivial branch's IoU length as well.
            return positive, positive, positive, positive, np.ones(len(labels)).tolist()
        computed = read(comp_path)
    else:
        path = Path(ref_path)
        files = sorted(p for p in path.parent.iterdir()
                       if p.suffix == '.tif' and p.name.startswith(path.stem))
        if not files:
            raise FileNotFoundError(f'No annotated SEG slices for {ref_path}')
        indices = [int(p.stem.removeprefix(path.stem + '_')) for p in files]
        with tifffile.TiffFile(comp_path) as prediction:
            if len(prediction.pages) > 1 and len(prediction.series[0].shape) == 3:
                return _match_chunks((read(p), prediction.pages[index].asarray(maxworkers=1))
                                     for p, index in zip(files, indices))
        reference = np.stack([read(p) for p in files])
        volume = read(comp_path)
        computed = np.stack([volume[index] for index in indices])
        del volume
    return match_arrays(reference, computed)
