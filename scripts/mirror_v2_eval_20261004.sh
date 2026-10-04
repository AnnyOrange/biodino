#!/usr/bin/env bash
# Mirror the Adaptive-v2 evaluation results (non-seg lanes, v4 detection_b8, v3 segmentation cells, MoNuSeg-30/7/14 site
# campaigns) from both eval hosts into the local mirror read by scripts/score_v2_arms_20261002.py.  One tar stream per
# host x arm (the link to the hosts is slow per file but fine per byte: ~55 MB / 3000 files per complete arm); feature caches,
# logs and adapters are excluded.      usage: bash scripts/mirror_v2_eval_20261004.sh [arm ...]
set -uo pipefail
M=/mnt/huawei_deepcad/dinov3/outputs/02_eval_runs/hs6_l5_v2_recovery_eval_20260930
ARMS=("$@"); [[ ${#ARMS[@]} -gt 0 ]] || ARMS=(global global_local global_cls global_slow global_w03 global_cls_slow global_cls_slow2 global_cls_early)
FIND='find results/point_*/*/bio_* results/point_*/*/command_manifest.json v4/detection_b8/point_*/*/*.json v3/cells/*/validation_report.json v3/cells/*/results adapters/checkpoint_curve.tsv -type f -not -path "*/features/*" -print0 2>/dev/null'
for host_root in "5090-hxw-xzj:/data/hs6_l5_v2_recovery_eval_20260930" "5090-lyx-xr:/data/xuzijing/hs6_l5_v2_recovery_eval_20260930"; do
  host=${host_root%%:*}; root=${host_root#*:}
  for arm in "${ARMS[@]}"; do
    ssh -o ConnectTimeout=20 "$host" "test -d $root/$arm" 2>/dev/null || continue
    mkdir -p "$M/$arm"
    ssh -o ConnectTimeout=20 "$host" "cd $root/$arm && $FIND | tar czf - --null -T -" | tar xzf - -C "$M/$arm" || echo "WARN tar stream failed: $host_root/$arm"
    echo "$(date -u +%T) $host $arm: $(find "$M/$arm" -type f | wc -l) files"
  done
done
# MoNuSeg 30/7/14 site campaigns (hxw campaign_v2/v3, lyx campaign_v3)
for src in "5090-hxw-xzj:/home/xzj/monuseg_t30v7_remote_20260930/campaign_v2" "5090-hxw-xzj:/home/xzj/monuseg_t30v7_remote_20260930/campaign_v3" "5090-lyx-xr:/data/xuzijing/monuseg_t30v7_remote_20260930/campaign_v3"; do
  host=${src%%:*}; dir=${src#*:}
  ssh -o ConnectTimeout=20 "$host" "test -d $dir" 2>/dev/null || continue
  dst="$M/monuseg30_$(basename "$dir")"; mkdir -p "$dst"
  ssh -o ConnectTimeout=20 "$host" "cd $dir && find cells/*/validation_report.json cells/*/results -type f -print0 2>/dev/null | tar czf - --null -T -" | tar xzf - -C "$dst" || echo "WARN tar stream failed: $src"
  echo "$(date -u +%T) $host $(basename "$dir"): $(find "$dst" -type f | wc -l) files"
done
echo "MIRROR DONE $(date -u +%FT%TZ)"
