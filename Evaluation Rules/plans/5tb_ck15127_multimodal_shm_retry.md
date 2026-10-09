# 5TB GRAM ck15127 Multimodal frozen-v3 retry after host disk failure

The original frozen-v3 task
`5tb_gram12687_ck15127__segmentation__multimodal_cellseg__primary-last__formal-static-v1`
failed on cpu2 because the node's root filesystem (including `/tmp`) had
0 bytes free and PyTorch could not create a shared-memory manager socket.
This is a host resource failure, not an observed model result. Retain the
original failed cell and its log unmodified.

Retry only this single task on a local 5090 when one card has at least
24 GiB free. Use the same evaluator source snapshot, checkpoint/config
receipt, frozen data/split hashes and batch32, six E20/E50 x seed0/1/2
fits with best validation and one test. No changes to the numerical
environment, feature geometry or model. Put all new JSON and validation
reports in an independent `5tb_ck15127_multimodal_shm_retry` campaign.
The prior run extracted its exact checkpoint/split-encoded frozen features
and failed during linear probing. Its retained feature cache may be reused
by symlink without rewriting the prior cell; the retry must still produce
all six newly validated numeric results or remain incomplete.
