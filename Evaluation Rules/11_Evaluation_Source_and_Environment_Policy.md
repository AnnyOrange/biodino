# Evaluation Source and Environment Policy

Status: adopted for new evaluation campaigns; historical campaigns remain bound to their recorded source until explicitly migrated.

## Canonical source

New evaluation launchers must run from this repository's `scripts/` and load protocol and dataset rules from this repository's `Evaluation Rules/`. They must derive the repository root from the launcher location (or an explicit repository-root argument) and write campaign artifacts under this repository's `outputs/02_eval_runs/`.

New launchers must not point at sibling `dinov3_*snapshot*` trees or `/mnt/huawei_deepcad/eval_envs`. A campaign record must capture the Git commit, dirty-file list and hashes, evaluator/rules hashes, launcher command, and output directory before work starts. If the working tree is dirty, the exact changed files used by the campaign must be archived with its source record.

## Runtime environment

Use an existing host-local conda environment only after a read-only compatibility check against the campaign's recorded requirements. Record Python, Torch, torchvision, NumPy, SciPy, scikit-learn, Pillow, transformers, timm, huggingface-hub, tokenizers, safetensors, CUDA runtime, and driver versions where applicable. Do not install or upgrade packages as part of preflight. If the environment is incompatible, do not launch; resolve the environment source and record it in the project before resuming evaluation.

Environment paths are host-specific. A launcher must not assume that a conda prefix from one machine exists on another. The campaign manifest must identify the environment on each worker host.

## Historical campaigns and snapshot retirement

Existing campaign launchers that pin an archived snapshot or evaluator fingerprint are historical evidence. Do not rewrite their source path in place. To continue one, create a new campaign record bound to the canonical repository, compare source and protocol behavior, and record the migration before launching.

An external snapshot or evaluation environment may be removed only after its complete archive and checksum are verified, no running process or queued job references it on every worker host, and all source/configuration references have been migrated or explicitly retired. An inaccessible host means this condition is unverified and retirement must wait.
