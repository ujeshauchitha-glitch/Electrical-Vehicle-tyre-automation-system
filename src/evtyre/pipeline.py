"""Pipeline: wire TelemetryFrame → extractors → Features → estimator.

Builds a registry that collects extractors and runs them over a frame,
tolerating any single extractor raising (Worker 5's resonance extractor
WILL raise NotImplementedError by design — catch it and record the feature
as UNAVAILABLE with the exception message, do not let it kill the run).
"""

from __future__ import annotations

from typing import Callable, Sequence

from .config.tyre import TyreConfig
from .config.vehicle import VehicleConfig
from .estimation.estimator import EstimatorResult, TyreEstimator
from .estimation.schema import TyreStateEstimate
from .features.contract import (
    Classification,
    Directionality,
    Feature,
    FeatureStatus,
)
from .schema.telemetry import TelemetryFrame


# Type alias for an extractor function
ExtractorFn = Callable[..., tuple[Feature, ...]]


class Pipeline:
    """Runs all registered extractors over a TelemetryFrame.

    Tolerates individual extractor failures — a raising extractor
    produces UNAVAILABLE features with the exception message, not a
    crash.
    """

    def __init__(
        self,
        vehicle_config: VehicleConfig,
        tyre_config: TyreConfig,
        road_load_params: object | None = None,
    ) -> None:
        self.vehicle_config = vehicle_config
        self.tyre_config = tyre_config
        self.road_load_params = road_load_params
        self._extractors: list[tuple[str, ExtractorFn, dict]] = []

    def register(
        self,
        name: str,
        extractor: ExtractorFn,
        extra_kwargs: dict | None = None,
    ) -> None:
        """Register a feature extractor.

        Parameters
        ----------
        name : str
            Human-readable name for logging.
        extractor : callable
            Must match the signature: extract(frame, vehicle_config, tyre_config) -> tuple[Feature, ...]
        extra_kwargs : dict, optional
            Extra keyword arguments to pass (e.g. road_load_params for
            the road load extractor).
        """
        self._extractors.append((name, extractor, extra_kwargs or {}))

    def extract_features(
        self,
        frame: TelemetryFrame,
    ) -> tuple[Feature, ...]:
        """Run all registered extractors over a frame.

        If an extractor raises, it produces UNAVAILABLE features with the
        exception message.  The pipeline continues with other extractors.
        """
        all_features: list[Feature] = []

        for name, extractor, extra_kwargs in self._extractors:
            try:
                features = extractor(
                    frame,
                    self.vehicle_config,
                    self.tyre_config,
                    **extra_kwargs,
                )
                all_features.extend(features)
            except NotImplementedError as e:
                # Worker 5's resonance extractor raises by design.
                # Record as UNAVAILABLE with the exception message.
                all_features.append(Feature(
                    name=f"{name}_error",
                    value=None,
                    unit="",
                    status=FeatureStatus.UNAVAILABLE,
                    unavailable_reason=str(e),
                    directionality=Directionality.NATURAL,
                    classification=Classification.D,
                    inputs=(),
                    corner=None,
                    timestamp_s=frame.timestamp_s,
                    provenance=frame.source,
                    extractor_version="pipeline",
                ))
            except Exception as e:
                # Catch-all for unexpected extractor failures.
                all_features.append(Feature(
                    name=f"{name}_error",
                    value=None,
                    unit="",
                    status=FeatureStatus.UNAVAILABLE,
                    unavailable_reason=f"Extractor {name} failed: {e}",
                    directionality=Directionality.NATURAL,
                    classification=Classification.D,
                    inputs=(),
                    corner=None,
                    timestamp_s=frame.timestamp_s,
                    provenance=frame.source,
                    extractor_version="pipeline",
                ))

        return tuple(all_features)

    def estimate(
        self,
        features: Sequence[Feature],
    ) -> TyreStateEstimate:
        """Run the estimator on extracted features.

        Returns a TyreStateEstimate (the Phase 3 -> Phase 4 contract),
        not a raw EstimatorResult.
        """
        estimator = TyreEstimator(self.vehicle_config, self.tyre_config)
        result = estimator.estimate(features)
        return estimator.to_schema(result, features)

    def run(
        self,
        frame: TelemetryFrame,
    ) -> tuple[tuple[Feature, ...], TyreStateEstimate]:
        """Full pipeline: extract features, then estimate.

        Returns (features, estimator_result).
        """
        features = self.extract_features(frame)
        result = self.estimate(features)
        return features, result
