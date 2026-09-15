# S+ e8 evaluation then B e8 migration (2026-09-10)

- Source checkpoint: XR C-scale S+ e8 `ckpt/8199`.
- Evaluation datasets: `bloodmnist`, `tissuemnist`, `cyclops-protein-loc`.
- Protocol: frozen full linear plus k=5/10, seeds 0/1/2; batch 64;
  resolution `best`; image size 224; channel policy `auto`.
- Resources: XR GPUs 4-7, at most three concurrent jobs per GPU.
- Resume rule: valid existing JSON/CSV entries are skipped.
- After all evaluation jobs complete successfully, copy only the latest valid B
  e8 checkpoint and its config from HXW to XR, verify byte size, and resume B
  on XR GPUs 4-7 with `num_workers=0`.
- Do not copy H+ or L checkpoints.
