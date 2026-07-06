"""Frozen-feature benchmark (classification / regression / multilabel / retrieval / clustering).

In-repo, self-contained port of the external `benchmark_model/` harness that
produced the reported ``benchmark_results_*.md`` numbers. The classification,
regression and multilabel probes are sklearn (StandardScaler + LogisticRegression
/ Ridge / OneVsRest) on L2-normalised frozen DINOv3 features. They use official
test splits when available, committed group-aware splits for source-linked
datasets, and a deterministic 80/20 split (seed 0) only as a fallback. The older
torch-SGD bio-classification linear probe has been removed so classification runs
produce only the published-number protocol. Classification defaults to the
2026-06-23 five-dataset resolution ablation table where available, with 224px as
the fallback/manual size.

Entry points:
- ``python -m dinov3.eval.bio_frozen_eval.run_classification`` — cls / reg / multilabel
- ``python -m dinov3.eval.bio_frozen_eval.run_retrieval_clustering`` — retrieval / clustering
"""
