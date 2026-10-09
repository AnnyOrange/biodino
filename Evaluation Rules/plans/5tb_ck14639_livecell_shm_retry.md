# 5TB GRAM ck14639 LIVECell frozen-v3 retry after cpu2 full disk

The registered ck14639 LIVECell segmentation component failed on cpu2
because its root filesystem was full and `torch_shm_manager` could not
create a socket directory. This is a host resource error; no metric from
this failure is accepted. Keep the failed cell and log intact. Retry in
an independent campaign using the identical registered evaluator snapshot,
checkpoint/config and frozen dataset/split receipts, batch32, six
E20/E50 x seed0/1/2 probes, and best validation then one test. An exact
checkpoint/split-encoded feature cache from the failed run may be read via
symlink; do not rewrite or treat it as a valid numeric result. Start only
after the ck15127 Multimodal host-disk retry is validated, on an available
local 5090 with at least 24 GiB free. Never overlap two high-resolution
segmentation tasks on one GPU.
