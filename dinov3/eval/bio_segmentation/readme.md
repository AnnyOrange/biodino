# Bio-Segmentation Evaluation

This folder provides a one-command pipeline for biological segmentation linear-probe evaluation:

1. build dataset objects;
2. extract frozen DINOv3 patch features;
3. train/evaluate a cached linear segmentation head.

## Quick Start

```bash
conda activate dinov3

python -m dinov3.eval.bio_segmentation.scripts.run_linear_probe_pipeline \
  --datasets conic pannuke tissuenet bbbc038 cellpose livecell monuseg \
  --checkpoints-dir /path/to/ckpt_root \
  --checkpoint-iters latest \
  --train-config /path/to/eval_config.yaml \
  --data-root-base /path/to/segmentation_datasets \
  --protocol best \
  --gpu 0
```

Use `--dry-run` first if you want to inspect the commands without executing them.

## Protocols

`--protocol best` applies the current dataset-specific evaluation settings:

| dataset | image/resize | layers | CE weighting |
|---|---|---|---|
| conic | `256 native` | `even4` | `sqrt_inverse` |
| pannuke | `256 native` | `even4` | none |
| tissuenet | `256 native` | `last1` | none |
| bbbc038 | `512 pad` | `even4` | none |
| cellpose | `512 pad` | `last1` | none |
| livecell | `512 pad` | `even4` | none |
| monuseg | `768 pad` | `last1` | none |

Notes:

- `native` means the original square patch size is preserved when possible.
- `pad` means long-side resize with aspect ratio preserved; padded labels use ignore index `255`.
- `even4` is resolved from backbone depth. For `vit_huge2`, it is `[7, 15, 23, 31]`.
- `--protocol manual` keeps the user-provided CLI settings (`--feature-img-size`, `--resize-mode`, `--layers`, `--layer-preset`, etc.).

## Important Arguments

- `--datasets`: one or more of `bbbc038 cellpose conic livecell monuseg pannuke tissuenet`.
- `--checkpoints-dir`: checkpoint root with `<iter>/checkpoint.pth` layout.
- `--checkpoint-iters`: `latest`, `all`, or explicit ids such as `14349` / `1000,2000`.
- `--train-config`: eval config matching the checkpoint architecture.
- `--data-root-base`: dataset root. `livecell` uses `<base>/LIVECell`; most others use `<base>/<dataset>/extracted`.
- `--cache-root` / `--output-root`: feature cache and result directories.
- `--skip-test-eval`: run train/val only.
- `--semantic-only`: skip instance metrics for faster screening.

## Outputs

```text
<cache-root>/<run-name>[__best__...]/<dataset>/<iter>/*.npz
<output-root>/<run-name>[__best__...]/<dataset>/<iter>/results.json
```

Each `results.json` contains validation metrics and, unless `--skip-test-eval` is used, test metrics.
