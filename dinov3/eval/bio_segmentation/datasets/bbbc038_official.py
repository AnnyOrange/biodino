"""Complete DSB2018 released scoring sets, preserving instance IDs."""
import json,os
from pathlib import Path
import numpy as np
from .bbbc038 import BBBC038Dataset

def resolve_path(path):
    value=str(path)
    for source,target in [('/mnt/huawei_deepcad/benchmark',os.environ.get('UNIFIED_PAPER_BENCHMARK_ROOT','/mnt/huawei_deepcad/benchmark')),('/mnt/huawei_deepcad/dinov3',str(Path(__file__).resolve().parents[4]))]:
        if value.startswith(source+'/'):return Path(target)/value[len(source)+1:]
    return Path(value)


def decode_rle(encoded, height, width):
    values = np.fromstring(encoded, dtype=np.int64, sep=' ')
    if len(values) % 2:
        raise ValueError('Odd RLE length')
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in values.reshape(-1, 2):
        if start < 1 or length < 1 or start - 1 + length > len(mask):
            raise ValueError('RLE outside image')
        mask[start - 1:start - 1 + length] = 1
    return mask.reshape((height, width), order='F')


class BBBC038OfficialDataset(BBBC038Dataset):
    def __init__(self, manifest, split, do_normalize=True):
        self.manifest = Path(manifest)
        self.split = split
        self.records = json.loads(self.manifest.read_text())['splits'][split]
        self._tensor_cache=None
        super().__init__([str(resolve_path(r['image'])) for r in self.records],
                         [str(resolve_path(r['instances'])) for r in self.records], size=None,
                         augment=False, do_normalize=do_normalize)

    def _load_instance_map(self, path):
        return np.load(resolve_path(path)).astype(np.int64)

    def preload(self):
        """Deterministic decode once; worker crops copy these immutable tensors."""
        self._tensor_cache=[super(BBBC038OfficialDataset,self).__getitem__(i)for i in range(len(self))]

    def __getitem__(self,index):
        return self._tensor_cache[index]if self._tensor_cache is not None else super().__getitem__(index)


def dsb_mean_precision(prediction, truth):
    """Image precision averaged over strict IoU > .50:.05:.95 matches.

    Disjoint instances and thresholds >= .5 make accepted pairs one-to-one.
    This is object-count precision, never a confidence-ranked COCO AP.
    """
    # Explicit background padding handles an image filled by one instance.
    gids = np.unique(truth[truth > 0]); pids = np.unique(prediction[prediction > 0])
    if not len(gids) and not len(pids):
        return 1.0
    if not len(gids) or not len(pids):
        return 0.0
    gl = np.zeros(int(truth.max()) + 1, dtype=np.int64)
    pl = np.zeros(int(prediction.max()) + 1, dtype=np.int64)
    gl[gids] = np.arange(1, len(gids) + 1); pl[pids] = np.arange(1, len(pids) + 1)
    gg = gl[truth].ravel(); pp = pl[prediction].ravel()
    table = np.bincount(gg * (len(pids) + 1) + pp,
                        minlength=(len(gids) + 1) * (len(pids) + 1)).reshape(len(gids) + 1, len(pids) + 1)
    intersection = table[1:, 1:]
    union = table[1:, :].sum(1)[:, None] + table[:, 1:].sum(0)[None, :] - intersection
    iou = intersection / union
    values = []
    for threshold in [round(.5 + .05 * i, 2) for i in range(10)]:
        matched = iou > threshold
        assert np.all(matched.sum(0) <= 1) and np.all(matched.sum(1) <= 1)
        tp = int(matched.sum())
        values.append(tp / (len(gids) + len(pids) - tp))
    return float(np.mean(values))
