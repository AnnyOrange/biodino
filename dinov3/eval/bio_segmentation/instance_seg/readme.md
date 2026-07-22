# Instance-Segmentation Track (Line 2) — DINOv3 vs cell specialists

This is the **second** bio-segmentation track. It exists to compare DINOv3 *fairly*
against **specialist** cell/nucleus models (Cellpose, Cellpose-SAM/cpsam), which the
linear-probe track (`../linear_probe.py`) cannot do.

## Why a separate track

The linear-probe track produces a *semantic* mask and fakes instances with connected
components — which **merges touching nuclei** and so understates AJI/PQ/AP. That is fine
for ranking **foundation models against each other** (every FM is handicapped equally),
but it is unfair against specialists that are built on flow/distance representations and
*do* split touching cells.

This track attaches a **CellViT/UNETR-style HoVerNet decoder** on the DINOv3 backbone and
produces **real** instances (touching nuclei separated via horizontal/vertical distance
maps + marker-controlled watershed), scored with the **same** instance metrics
(`../metrics`) used for the specialists.

## The comparison logic

Hold the **decoder + data + metrics fixed**, vary only the **backbone** — then any score
delta is attributable to the backbone. Run the pipeline once per backbone:

| row | backbone (`--checkpoints-dir` / `--train-config`) | role |
|---|---|---|
| bio-DINOv3 | our microscopy-continual ckpt | main |
| **generic DINOv3** | LVD ckpt | **zero point: bio − generic = value of our continual pretraining** |
| other bio FM | their ckpt | vs peers |
| Cellpose / cpsam | — (`run_specialist.py`) | external reference, **same metric code** |

Report the controlled adaptation ladder with the same decoder and splits:

- `frozen`: decoder-only (`--freeze-backbone`)
- `last2` / `last4`: decoder plus the last N transformer blocks and final norm
- `finetune`: decoder plus the complete backbone

The pipeline accepts all four names through `--modes frozen last2 last4 finetune`.

## Layers are configurable

The decoder upsamples 16× (4 stages, patch16). The number of ViT layers you *tap* is free:
`--layers 4 11 17 23` (even-4, default) or `--layers 1 3 5 ... 23` (more). A fusion
front-end maps any K taps onto the 4 skip slots (K=4 → 1/bucket, K=8 → 2/bucket).

## Quick start

```bash
conda activate dinov3

# CPU smoke test (no GPU/data) — validates the engine end-to-end
python -m dinov3.eval.bio_segmentation.instance_seg.smoke_test

# One backbone, frozen, two datasets
python -m dinov3.eval.bio_segmentation.instance_seg.scripts.run_cellvit_pipeline \
  --datasets pannuke monuseg \
  --checkpoints-dir /ckpt/bio_dinov3 --checkpoint-iters latest \
  --train-config dinov3/configs/train/microscopy_continual_vitl16.yaml \
  --data-root-base /data/segmentation_datasets \
  --output-root ./outputs/instance_seg/bio_dinov3 \
  --modes frozen --epochs 50 --gpu 0

# Specialist baseline, SAME metrics (run in a cellpose-enabled env)
python -m dinov3.eval.bio_segmentation.instance_seg.scripts.run_specialist \
  --dataset monuseg --data-root /data/segmentation_datasets/monuseg/extracted \
  --model nuclei --output-dir ./outputs/instance_seg/specialist/cellpose/monuseg
```

## Datasets

Uses the 6 datasets that ship true instance maps: `pannuke` (6-class), `conic` (7-class),
`monuseg`, `livecell`, `bbbc038`, `tissuenet` (binary). Multi-class datasets enable the TP
(type) branch and report `mPQ`; binary datasets report `bPQ` / `AJI` / `AP`. PanNuke/CoNIC
are 256² (single forward); MoNuSeg/LiveCell are tiled (`tiling.py`) then post-processed once.

## Files

| file | role |
|---|---|
| `targets.py`   | `(inst, sem)` → NP / HV / TP targets |
| `decoder.py`   | UNETR/CellViT decoder, flexible-tap fusion front-end |
| `model.py`     | `DINOHoVerNet` = backbone (frozen\|ft) + decoder |
| `losses.py`    | NP(CE+Dice) + HV(MSE+MSGE) + TP(CE+Dice) |
| `postproc.py`  | HoVerNet watershed → instance + per-instance class |
| `tiling.py`    | sliding-window inference for large images |
| `train.py`     | train + eval; writes `results.json` (same schema as linear probe) |
| `smoke_test.py`| CPU engine test |
| `scripts/run_cellvit_pipeline.py` | driver: datasets × ckpts × {frozen,ft} |
| `scripts/run_specialist.py`       | Cellpose/cpsam → **same** metrics |

## Outputs

`results.json` per run with `val` / `test` instance metrics (`AJI`, `AP`, `AP50`, `AP75`,
`bPQ`, and `mPQ` for multi-class) — directly comparable to the specialist JSONs.

## References

- Graham et al., *HoVer-Net*, Medical Image Analysis 2019.
- Hörst et al., *CellViT*, Medical Image Analysis 2024.
- Hatamizadeh et al., *UNETR*, WACV 2022.
