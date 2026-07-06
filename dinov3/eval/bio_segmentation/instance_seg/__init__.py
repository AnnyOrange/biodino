"""
Instance-segmentation evaluation track (Line 2) for bio_segmentation.

Unlike the linear-probe track (`../linear_probe.py`), which produces a semantic
mask and fakes instances via connected components (cannot split touching nuclei),
this track attaches a CellViT/UNETR-style **HoVerNet** decoder on top of the
DINOv3 backbone and produces *real* instances (touching cells separated via
horizontal/vertical distance maps + marker-controlled watershed).

Design goal: hold the decoder + data + metrics fixed, vary only the **backbone**,
so any win is attributable to our model. The key baseline is bio-DINOv3 vs a
generic DINOv3 in the same harness. Specialist models (Cellpose / cpsam) are run
through `scripts/run_specialist.py` and scored with the SAME metric code
(`..metrics.accumulate_instance_metrics`), guaranteeing one ruler for everyone.

References:
    Graham et al., "HoVer-Net", Medical Image Analysis 2019.
    Hörst et al., "CellViT", Medical Image Analysis 2024.
"""
