"""Offline repackage pipeline: per-channel → packed multi-channel WebDataset."""

from .config import RepackConfig

__all__ = ["RepackConfig", "run_pipeline"]


def __getattr__(name: str):
    if name == "run_pipeline":
        from .pipeline import run_pipeline
        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
