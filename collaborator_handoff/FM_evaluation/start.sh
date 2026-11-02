#!/usr/bin/env bash
set -euo pipefail
# Usage: bash start.sh ARCHIVES WORKSPACE BENCHMARK WEIGHTS GPU_ID
if [ "$#" -ne 5 ]; then
  echo 'Usage: bash start.sh /data/uploads /data/new_eval_workspace /data/benchmark /data/weights 0' >&2
  exit 2
fi
bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
archives=$(realpath "$1")
workspace=$(realpath -m "$2")
benchmark=$(realpath -m "$3")
weights=$(realpath "$4")
gpu_id=$5
[[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo 'GPU_ID must be one allocated physical GPU index' >&2; exit 2; }
docker info >/dev/null
python3 "$bundle/tools/verify_bundle.py"
if [ ! -e "$workspace" ]; then
  python3 "$bundle/tools/materialize_workspace.py" "$workspace"
elif [ ! -f "$workspace/.acceptance_workspace" ]; then
  echo 'Workspace exists without this bundle marker; choose a fresh directory.' >&2
  exit 1
fi
printf '%s\n' 'FM_end_to_end_acceptance_20261031' > "$workspace/.acceptance_workspace"
mkdir -p "$benchmark"
docker build -t biodino-fm-acceptance:20261031 "$bundle"
docker run --rm -it --gpus "device=$gpu_id" --memory=192g --memory-swap=192g --cpus=8 --shm-size=8g \
  --mount "type=bind,src=$bundle,dst=/acceptance,readonly" \
  --mount "type=bind,src=$workspace,dst=/mnt/huawei_deepcad/dinov3" \
  --mount "type=bind,src=$benchmark,dst=/mnt/huawei_deepcad/benchmark" \
  --mount "type=bind,src=$benchmark,dst=/data/xuzijing/benchmark" \
  --mount "type=bind,src=$bundle/benchmark_model,dst=/mnt/huawei_deepcad/benchmark_model,readonly" \
  --mount "type=bind,src=$archives,dst=/uploads,readonly" \
  --mount "type=bind,src=$weights,dst=/weights,readonly" \
  biodino-fm-acceptance:20261031 bash -lc 'python /acceptance/tools/bootstrap_runtime.py --repo /mnt/huawei_deepcad/dinov3 && exec bash'
