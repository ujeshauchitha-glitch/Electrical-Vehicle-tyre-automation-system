"""Phase 2 — Observable / feature extraction.

Every extractor is a pure function with signature:

    def extract(frame, vehicle_config, tyre_config) -> tuple[Feature, ...]

The Feature dataclass and its supporting enums are defined in contract.py.
"""

from .contract import (
    Classification,
    Directionality,
    Feature,
    FeatureStatus,
)

__all__ = [
    "Classification",
    "Directionality",
    "Feature",
    "FeatureStatus",
]
