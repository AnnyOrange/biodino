"""Compatibility facade for dedup whitelist APIs.

This module keeps external imports stable while delegating real work
to SRP-compliant submodules.
"""

from .dedup_filter import filter_and_shuffle
from .dedup_loader import build_whitelist
from .dedup_types import Whitelist

__all__ = ["Whitelist", "build_whitelist", "filter_and_shuffle"]


