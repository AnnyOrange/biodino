# L e4 tuned evaluation on HXW (2026-09-10)

- Checkpoint: C-scale L e4 tuned-lr run, `ckpt/4099`.
- Datasets: all registered benchmark defaults: 24 classification, 4 regression,
  4 retrieval, 8 segmentation, and 3 detection datasets.
- Protocol: frozen batch 64, resolution `best`, image size 224, channel policy
  `auto`; the three C-scale classification datasets additionally run k=5/10,
  seeds 0/1/2.
- Resources: HXW GPUs 4-7. Core frozen jobs use five slots per GPU (20 total).
  Dense segmentation/detection jobs use one slot per GPU to avoid L-model OOM.
- Valid existing JSON/CSV results are skipped.
- The checkpoint remains on HXW; no model checkpoint is transferred.
