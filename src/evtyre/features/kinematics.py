"""Wheel-speed-derived kinematic features.

Key project rules:
- Compare a wheel to its AXLE PARTNER, never to all four.  A rear-drive EV
  wears its drive axle ~1.7x faster; cross-axle comparison is a permanent
  false alarm.
- Slip ratio (R*ω - V) / max(V, ε).  V is estimated from vehicle_speed_ms.
  If V is unavailable or below a threshold, slip is UNAVAILABLE.
- Classification: A for direct ratios, B for anything inferring rolling-radius
  *change* from speed ratios (the assumption is stated in docstrings).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.tyre import TyreConfig
from ..config.vehicle import DriveLayout, VehicleConfig
from ..schema.common import CORNERS, SensorReading, SensorStatus
from ..schema.telemetry import TelemetryFrame
from .contract import Classification, Directionality, Feature, FeatureStatus

if TYPE_CHECKING:
    pass

EXTRACTOR_VERSION = "0.1.0"

# Minimum vehicle speed (m/s) for slip-ratio calculation.  Below this,
# the ratio is dominated by noise and is not physically meaningful.
_MIN_SPEED_FOR_SLIP_MS: float = 0.5

# Small constant to avoid division by zero in slip ratio denominator.
_EPSILON_SLIP: float = 1e-6

# Axle definitions: which corners share an axle.
_FRONT_AXLE: tuple[str, str] = ("FL", "FR")
_REAR_AXLE: tuple[str, str] = ("RL", "RR")


def _get_speed_ms(frame: TelemetryFrame) -> float | None:
    """Return vehicle speed in m/s, or None if unavailable."""
    vr = frame.vehicle_speed_ms
    if vr.is_usable and vr.value is not None and vr.value >= 0.0:
        return vr.value
    return None


def _wheel_omega(frame: TelemetryFrame, corner: str) -> float | None:
    """Return wheel angular velocity in rad/s, or None if unavailable."""
    wr = frame.wheel_speed_rad_s[corner]
    if wr.is_usable and wr.value is not None:
        return wr.value
    return None


def _axle_speed_ratio_features(
    frame: TelemetryFrame,
    left: str,
    right: str,
    axle_label: str,
    ts: float,
    prov: str,
) -> list[Feature]:
    """Speed ratio between the two wheels on one axle.

    omega_left / omega_right.  If either wheel is missing, both features
    for that axle are UNAVAILABLE.
    """
    omega_l = _wheel_omega(frame, left)
    omega_r = _wheel_omega(frame, right)

    features: list[Feature] = []

    # Speed ratio: omega_left / omega_right
    if omega_l is None or omega_r is None:
        reasons = []
        if omega_l is None:
            reasons.append(f"wheel speed missing for {left}")
        if omega_r is None:
            reasons.append(f"wheel speed missing for {right}")
        features.append(Feature(
            name=f"axle_speed_ratio_{axle_label}",
            value=None,
            unit="",
            status=FeatureStatus.UNAVAILABLE,
            unavailable_reason="; ".join(reasons),
            directionality=Directionality.NATURAL,
            classification=Classification.A,
            inputs=("wheel_speed_rad_s",),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))
    else:
        # Guard against degenerate case: if both are essentially zero,
        # the ratio is undefined.
        denom = max(abs(omega_r), _EPSILON_SLIP)
        if abs(omega_r) < _EPSILON_SLIP and abs(omega_l) < _EPSILON_SLIP:
            features.append(Feature(
                name=f"axle_speed_ratio_{axle_label}",
                value=None,
                unit="",
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason=(
                    f"Both wheels on {axle_label} axle have near-zero speed; "
                    f"ratio is undefined"
                ),
                directionality=Directionality.NATURAL,
                classification=Classification.A,
                inputs=("wheel_speed_rad_s",),
                corner=None,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))
        else:
            ratio = omega_l / omega_r
            features.append(Feature(
                name=f"axle_speed_ratio_{axle_label}",
                value=ratio,
                unit="",
                status=FeatureStatus.OK,
                unavailable_reason=None,
                directionality=Directionality.NATURAL,
                classification=Classification.A,
                inputs=("wheel_speed_rad_s",),
                corner=None,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))

    return features


def extract(
    frame: TelemetryFrame,
    vehicle_config: VehicleConfig,
    tyre_config: TyreConfig,
) -> tuple[Feature, ...]:
    """Extract wheel-speed-derived kinematic features.

    All comparison features are axle-scoped (front pair compared to front
    pair, rear pair to rear pair).  See module docstring for the rationale.
    """
    features: list[Feature] = []
    ts = frame.timestamp_s
    prov = frame.source
    v_ms = _get_speed_ms(frame)

    # --- Per-corner effective rolling radius ratio ---
    # R_eff / R_nominal = V / (omega * R_belt)
    # Classification B: assumes the effective rolling radius model
    # R_eff ≈ V / omega holds, which is an approximation (the 1/3
    # deflection factor from tyre mechanics).
    for corner in CORNERS:
        omega = _wheel_omega(frame, corner)
        inputs = ("wheel_speed_rad_s", "vehicle_speed_ms")
        if v_ms is None or omega is None:
            reasons = []
            if v_ms is None:
                reasons.append("vehicle speed unavailable")
            if omega is None:
                reasons.append(f"wheel speed missing for {corner}")
            features.append(Feature(
                name=f"effective_rolling_radius_ratio_{corner}",
                value=None,
                unit="",
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason="; ".join(reasons),
                directionality=Directionality.NATURAL,
                classification=Classification.B,
                inputs=inputs,
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))
        elif abs(omega) < _EPSILON_SLIP:
            features.append(Feature(
                name=f"effective_rolling_radius_ratio_{corner}",
                value=None,
                unit="",
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason=(
                    f"Wheel speed for {corner} is near zero; "
                    f"rolling radius ratio undefined"
                ),
                directionality=Directionality.NATURAL,
                classification=Classification.B,
                inputs=inputs,
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))
        else:
            r_ratio = v_ms / (omega * tyre_config.wheel_belt_radius_m)
            features.append(Feature(
                name=f"effective_rolling_radius_ratio_{corner}",
                value=r_ratio,
                unit="",
                status=FeatureStatus.OK,
                unavailable_reason=None,
                directionality=Directionality.NATURAL,
                classification=Classification.B,
                inputs=inputs,
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))

    # --- Per-corner longitudinal slip ratio ---
    # slip = (R*omega - V) / max(V, epsilon)
    # V is estimated from vehicle_speed_ms (CAN bus / odometry).
    # If V is unavailable, slip is UNAVAILABLE.
    # Classification A: direct telemetry ratio.
    if v_ms is not None and v_ms >= _MIN_SPEED_FOR_SLIP_MS:
        for corner in CORNERS:
            omega = _wheel_omega(frame, corner)
            inputs = ("wheel_speed_rad_s", "vehicle_speed_ms")
            if omega is None:
                features.append(Feature(
                    name=f"slip_ratio_{corner}",
                    value=None,
                    unit="",
                    status=FeatureStatus.UNAVAILABLE,
                    unavailable_reason=f"wheel speed missing for {corner}",
                    directionality=Directionality.NATURAL,
                    classification=Classification.A,
                    inputs=inputs,
                    corner=corner,
                    timestamp_s=ts,
                    provenance=prov,
                    extractor_version=EXTRACTOR_VERSION,
                ))
            else:
                r_eff = tyre_config.wheel_belt_radius_m
                slip = (r_eff * omega - v_ms) / max(v_ms, _EPSILON_SLIP)
                features.append(Feature(
                    name=f"slip_ratio_{corner}",
                    value=slip,
                    unit="",
                    status=FeatureStatus.OK,
                    unavailable_reason=None,
                    directionality=Directionality.NATURAL,
                    classification=Classification.A,
                    inputs=inputs,
                    corner=corner,
                    timestamp_s=ts,
                    provenance=prov,
                    extractor_version=EXTRACTOR_VERSION,
                ))
    else:
        reason = "vehicle speed unavailable" if v_ms is None else (
            f"vehicle speed {v_ms} m/s below threshold "
            f"{_MIN_SPEED_FOR_SLIP_MS} m/s"
        )
        for corner in CORNERS:
            features.append(Feature(
                name=f"slip_ratio_{corner}",
                value=None,
                unit="",
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason=reason,
                directionality=Directionality.NATURAL,
                classification=Classification.A,
                inputs=("wheel_speed_rad_s", "vehicle_speed_ms"),
                corner=corner,
                timestamp_s=ts,
                provenance=prov,
                extractor_version=EXTRACTOR_VERSION,
            ))

    # --- Axle-pair speed ratios ---
    # Compare a wheel to its axle partner only, never cross-axle.
    features.extend(_axle_speed_ratio_features(
        frame, "FL", "FR", "front", ts, prov,
    ))
    features.extend(_axle_speed_ratio_features(
        frame, "RL", "RR", "rear", ts, prov,
    ))

    # --- Front-to-rear axle speed ratio difference ---
    # Differences between the two axle ratios can indicate different
    # wear states, but are NOT used for cross-axle comparison of
    # individual wheels.
    front_ratio = _axle_speed_ratio_features(
        frame, "FL", "FR", "front", ts, prov,
    )
    rear_ratio = _axle_speed_ratio_features(
        frame, "RL", "RR", "rear", ts, prov,
    )
    # Both need to be OK to compute the difference
    fr_feat = [f for f in front_ratio if f.name == "axle_speed_ratio_front"]
    rr_feat = [f for f in rear_ratio if f.name == "axle_speed_ratio_rear"]
    if fr_feat and rr_feat and fr_feat[0].status == FeatureStatus.OK and rr_feat[0].status == FeatureStatus.OK:
        diff = fr_feat[0].value - rr_feat[0].value  # type: ignore[union-attr]
        features.append(Feature(
            name="axle_speed_ratio_diff_front_minus_rear",
            value=diff,
            unit="",
            status=FeatureStatus.OK,
            unavailable_reason=None,
            directionality=Directionality.NATURAL,
            classification=Classification.A,
            inputs=("wheel_speed_rad_s",),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))
    else:
        reasons = []
        if fr_feat and fr_feat[0].status != FeatureStatus.OK:
            reasons.append(fr_feat[0].unavailable_reason or "front axle unavailable")
        if rr_feat and rr_feat[0].status != FeatureStatus.OK:
            reasons.append(rr_feat[0].unavailable_reason or "rear axle unavailable")
        features.append(Feature(
            name="axle_speed_ratio_diff_front_minus_rear",
            value=None,
            unit="",
            status=FeatureStatus.UNAVAILABLE,
            unavailable_reason="; ".join(reasons) if reasons else "axle ratio unavailable",
            directionality=Directionality.NATURAL,
            classification=Classification.A,
            inputs=("wheel_speed_rad_s",),
            corner=None,
            timestamp_s=ts,
            provenance=prov,
            extractor_version=EXTRACTOR_VERSION,
        ))

    return tuple(features)
