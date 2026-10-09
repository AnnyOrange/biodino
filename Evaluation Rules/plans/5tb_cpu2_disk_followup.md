# 5TB GRAM two further cpu2 disk failures: identical-protocol retry

The ck13663 Multimodal and ck13175 LIVECell frozen-v3 segmentation fits
were interrupted when cpu2 had 0 bytes available on its root `/tmp`;
both original logs show `torch_shm_manager` unable to create a socket
directory. Retain all original failure evidence. Admission requires the
registered teacher checkpoint stat/config SHA, pinned source snapshot and
dataset split hashes to remain identical. Retry both in an independent
campaign with their original high-resolution feature batch32, fixed probe
batch32, E20/E50 × seeds0/1/2 and every-epoch validation. Reuse only the
retained checkpoint/split-named feature caches via symlink, not numeric
completion markers. Schedule the two fits sequentially on a free local
5090 GPU0 after the first two cpu2 disk-failure retries validate; no
two dense tests may share a GPU.
