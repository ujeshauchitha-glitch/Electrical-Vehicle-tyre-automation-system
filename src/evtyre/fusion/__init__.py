"""Phase 4 — multi-variable fusion and the decision layer.

Turns a Phase 3 TyreStateEstimate into decision-ready output. The governing
rule, enforced by contract.require_observed(), is that a decision may only be
OK if every state it consumes is OBSERVED.

What that means in practice today:

    maintenance view       AVAILABLE     pressure is OBSERVED
    torque ceiling         WITHHELD      tread is WEAK; the 2-sigma lower bound
                                         would be prior width, not measurement
    recoverable energy     PARTIAL       pressure component only; the alignment
                                         component needs toe, which is
                                         UNOBSERVABLE
"""

from .config import FrictionConfig, MaintenanceConfig, driven_corners
from .contract import Decision, DecisionStatus, require_observed
from .energy import recoverable_energy
from .maintenance import cold_equivalent_pressure_kpa, maintenance_view
from .multisnapshot import SnapshotFusionError, fuse_snapshots
from .report import REPORT_SCHEMA_VERSION, FusedTyreReport, build_report
from .traction import torque_ceiling, wet_friction
from .trend import TrendClassification, TrendReport, TrendResult, detect_trends, format_trend_report
from .forecast import (
    DegradationForecast,
    MaintenanceForecast,
    Urgency,
    compute_forecasts,
    format_forecast,
)
from .nonlinear_wear import (
    NonLinearFit,
    NonLinearWearReport,
    WearModel,
    fit_nonlinear_wear,
    format_nonlinear_report,
)
from .asymmetric import (
    AsymmetryReport,
    AsymmetryType,
    CornerAnalysis,
    Severity,
    detect_asymmetry,
    format_asymmetry_report,
)
from .anomaly import (
    AnomalyEvent,
    AnomalyReport,
    AnomalySeverity,
    AnomalyType,
    detect_anomalies,
    format_anomaly_report,
)
from .thermal_degradation import (
    ThermalDegradationReport,
    ThermalModelConfig,
    ThermalRegime,
    ThermalState,
    adjust_wear_rate_for_temperature,
    compute_thermal_degradation_report,
    compute_thermal_state,
    format_thermal_report,
    predict_pressure_from_temperature,
)

__all__ = [
    "Decision",
    "DecisionStatus",
    "require_observed",
    "FrictionConfig",
    "MaintenanceConfig",
    "driven_corners",
    "maintenance_view",
    "cold_equivalent_pressure_kpa",
    "torque_ceiling",
    "wet_friction",
    "recoverable_energy",
    "fuse_snapshots",
    "SnapshotFusionError",
    "FusedTyreReport",
    "build_report",
    "REPORT_SCHEMA_VERSION",
    # Phase 5: temporal degradation
    "TrendClassification",
    "TrendReport",
    "TrendResult",
    "detect_trends",
    "format_trend_report",
    "DegradationForecast",
    "MaintenanceForecast",
    "Urgency",
    "compute_forecasts",
    "format_forecast",
    # Phase 5 enhancements: non-linear wear
    "NonLinearFit",
    "NonLinearWearReport",
    "WearModel",
    "fit_nonlinear_wear",
    "format_nonlinear_report",
    # Phase 5 enhancements: asymmetric wear
    "AsymmetryReport",
    "AsymmetryType",
    "CornerAnalysis",
    "Severity",
    "detect_asymmetry",
    "format_asymmetry_report",
    # Phase 5 enhancements: anomaly detection
    "AnomalyEvent",
    "AnomalyReport",
    "AnomalySeverity",
    "AnomalyType",
    "detect_anomalies",
    "format_anomaly_report",
    # Phase 5 enhancements: thermal degradation
    "ThermalDegradationReport",
    "ThermalModelConfig",
    "ThermalRegime",
    "ThermalState",
    "adjust_wear_rate_for_temperature",
    "compute_thermal_degradation_report",
    "compute_thermal_state",
    "format_thermal_report",
    "predict_pressure_from_temperature",
]
