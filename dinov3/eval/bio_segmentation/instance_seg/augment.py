"""
Training-side augmentation for the instance-seg track (backbone stays FROZEN —
these only touch the input image + targets, never the DINOv3 weights).

Mirrors the tricks the multimodality cell-seg challenge winners used (Ma et al.,
Nat. Methods 2024, Table 1): strong intensity perturbation, cell-aware per-cell
intensity jitter, Mosaic, and spatial (rotate/scale/flip) augmentation. All
operate on (img[0,1] HWC, sem, inst) before HoVerNet target generation.
"""

from __future__ import annotations

import cv2
import numpy as np


def _contig(a):
    return np.ascontiguousarray(a)


def random_flip_rot(img, sem, inst, rng):
    if rng.random() < 0.5:
        img, sem, inst = img[:, ::-1], sem[:, ::-1], inst[:, ::-1]
    if rng.random() < 0.5:
        img, sem, inst = img[::-1], sem[::-1], inst[::-1]
    k = int(rng.integers(0, 4))
    if k:
        img, sem, inst = np.rot90(img, k), np.rot90(sem, k), np.rot90(inst, k)
    return _contig(img), _contig(sem), _contig(inst)


def random_scale_rotate(img, sem, inst, rng, scale_range=(0.7, 1.4), max_angle=180.0, prob=0.7):
    """Arbitrary rotation + scale (image bilinear, masks nearest, reflect border)."""
    if rng.random() > prob:
        return img, sem, inst
    h, w = img.shape[:2]
    angle = float(rng.uniform(-max_angle, max_angle))
    scale = float(rng.uniform(*scale_range))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    img2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    sem2 = cv2.warpAffine(sem.astype(np.int32), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    inst2 = cv2.warpAffine(inst.astype(np.int32), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return img2.astype(np.float32), sem2.astype(np.int64), inst2.astype(np.int64)


def random_intensity(img, rng):
    """Brightness/contrast, gamma, gaussian noise, blur on a [0,1] HWC image."""
    if rng.random() < 0.6:
        a = float(rng.uniform(0.65, 1.35)); b = float(rng.uniform(-0.12, 0.12))
        img = np.clip(img * a + b, 0, 1)
    if rng.random() < 0.5:
        g = float(rng.uniform(0.6, 1.6)); img = np.clip(img ** g, 0, 1)
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, float(rng.uniform(0.01, 0.05)), img.shape), 0, 1)
    if rng.random() < 0.25:
        k = int(rng.choice([3, 5])); img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.15:  # channel shuffle / drop — robustness to stain/channel order
        if rng.random() < 0.5:
            img = img[:, :, rng.permutation(3)]
    return img.astype(np.float32)


def cell_aware_intensity(img, inst, rng, prob=0.5, frac=0.5):
    """Per-instance intensity gain/bias — robustness to per-cell intensity variation."""
    if rng.random() > prob:
        return img
    out = img.copy()
    ids = np.unique(inst); ids = ids[ids > 0]
    for i in ids:
        if rng.random() < frac:
            m = inst == i
            g = float(rng.uniform(0.55, 1.45)); b = float(rng.uniform(-0.12, 0.12))
            out[m] = np.clip(out[m] * g + b, 0, 1)
    return out.astype(np.float32)


def _resize_to(img, sem, inst, S):
    if img.shape[0] != S or img.shape[1] != S:
        img = cv2.resize(img, (S, S), interpolation=cv2.INTER_LINEAR)
        sem = cv2.resize(sem.astype(np.int32), (S, S), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        inst = cv2.resize(inst.astype(np.int32), (S, S), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    return img, sem, inst


def mosaic(samples, crop, rng):
    """4 samples → 2x2 grid (each SxS) → random SxS crop. Instance ids offset per tile."""
    S = crop
    big_img = np.zeros((2 * S, 2 * S, 3), np.float32)
    big_sem = np.zeros((2 * S, 2 * S), np.int64)
    big_inst = np.zeros((2 * S, 2 * S), np.int64)
    offset = 0
    for q, (img, sem, inst) in enumerate(samples):
        img, sem, inst = _resize_to(img, sem, inst, S)
        yy, xx = (q // 2) * S, (q % 2) * S
        big_img[yy:yy + S, xx:xx + S] = img
        big_sem[yy:yy + S, xx:xx + S] = sem
        inst2 = inst.copy()
        inst2[inst2 > 0] += offset
        offset = int(inst2.max())
        big_inst[yy:yy + S, xx:xx + S] = inst2
    y, x = int(rng.integers(0, S + 1)), int(rng.integers(0, S + 1))
    return (_contig(big_img[y:y + S, x:x + S]),
            _contig(big_sem[y:y + S, x:x + S]),
            _contig(big_inst[y:y + S, x:x + S]))
