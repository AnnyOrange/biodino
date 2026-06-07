"""Frozen-feature benchmark (classification / regression / multilabel / retrieval / clustering).

In-repo, self-contained port of the external `benchmark_model/` harness that
produced the reported ``benchmark_results_*.md`` numbers. The classification,
regression and multilabel probes are sklearn (StandardScaler + LogisticRegression
/ Ridge / OneVsRest) on L2-normalised frozen DINOv3 features with a deterministic
80/20 split (seed 0) — replacing the older torch-SGD probes in
``dinov3/eval/bio_classification`` so others reproduce the published numbers.

Entry points:
- ``python -m dinov3.eval.bio_frozen_eval.run_classification`` — cls / reg / multilabel
- ``python -m dinov3.eval.bio_frozen_eval.run_retrieval_clustering`` — retrieval / clustering
"""
