"""Phase 6 tests — digital twin simulation.

Tests cover:
1. Mock simulator produces valid telemetry
2. Ground-truth scenario changes affect simulated telemetry
3. Missing channels are represented as missing, not zero
4. Estimator receives simulator telemetry correctly
5. Replay is deterministic
6. Ground-truth comparison works
7. All scenario types produce valid results
8. Existing Phase 1-5 tests continue to pass (run separately)
"""

import math
import pytest

from src.evtyre.config.tyre import TyreConfig
from src.evtyre.config.vehicle import DriveLayout, VehicleConfig
from src.evtyre.estimation.estimator import DEFAULT_PHYSICS, TyreEstimator
from src.evtyre.features.contract import FeatureStatus
from src.evtyre.schema.common import CORNERS, SensorStatus
from src.evtyre.schema.telemetry import TelemetryFrame

from src.evtyre.simulation.ground_truth import GroundTruthTyre, GroundTruthVehicle
from src.evtyre.simulation.interface import GroundTruthSnapshot, SimulationState, VehicleSimulator
from src.evtyre.simulation.mock_adapter import MockVehicleAdapter, MockVehicleConfig
from src.evtyre.simulation.replay import ReplayEngine
from src.evtyre.simulation.scenarios import (
    Scenario,
    ScenarioType,
    load_scenario,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def mock_adapter():
    adapter = MockVehicleAdapter()
    adapter.reset(load_scenario(ScenarioType.NORMAL))
    return adapter


@pytest.fixture
def tyre_config():
    return TyreConfig(
        tyre_model_id="test_tyre",
        wheel_belt_radius_m=0.322,
        tread_new_mm=8.0,
        tread_legal_mm=1.6,
        placard_pressure_kpa=240.0,
        cold_reference_temperature_c=25.0,
    )


@pytest.fixture
def vehicle_config():
    return VehicleConfig(
        vehicle_id="test_ev",
        mass_kg=1800.0,
        front_weight_fraction=0.48,
        drive_layout=DriveLayout.RWD,
    )


# ===========================================================================
# Test: Mock simulator produces valid telemetry
# ===========================================================================

class TestMockSimulator:
    def test_produces_valid_telemetry_frame(self, mock_adapter):
        state = mock_adapter.step(dt_s=1.0)
        assert isinstance(state, SimulationState)
        assert isinstance(state.telemetry, TelemetryFrame)
        assert state.telemetry.source == "simulated"
        assert state.timestamp_s > 0

    def test_all_corners_have_wheel_speeds(self, mock_adapter):
        state = mock_adapter.step(dt_s=1.0)
        for corner in CORNERS:
            ws = state.telemetry.wheel_speed_rad_s[corner]
            assert ws.status == SensorStatus.OK
            assert ws.value is not None
            assert ws.value > 0

    def test_all_corners_have_pressure(self, mock_adapter):
        state = mock_adapter.step(dt_s=1.0)
        for corner in CORNERS:
            pr = state.telemetry.tpms_pressure_kpa[corner]
            assert pr.status == SensorStatus.OK
            assert pr.value is not None
            assert pr.value > 0

    def test_motor_torque_is_available(self, mock_adapter):
        state = mock_adapter.step(dt_s=1.0)
        mt = state.telemetry.motor_torque_nm
        assert mt.status == SensorStatus.OK
        assert mt.value is not None

    def test_vehicle_speed_is_positive(self, mock_adapter):
        state = mock_adapter.step(dt_s=1.0)
        vs = state.telemetry.vehicle_speed_ms
        assert vs.status == SensorStatus.OK
        assert vs.value is not None
        assert vs.value > 0

    def test_ground_truth_matches_scenario(self, mock_adapter):
        state = mock_adapter.step(dt_s=0.1)
        gt = state.ground_truth
        # NORMAL scenario starts with 5.0 mm tread
        for corner in CORNERS:
            assert abs(gt.tread_mm[corner] - 5.0) < 0.01
            assert abs(gt.pressure_kpa[corner] - 240.0) < 5.0  # noise


# ===========================================================================
# Test: Scenarios
# ===========================================================================

class TestScenarios:
    def test_all_scenario_types_are_loadable(self):
        for st in ScenarioType:
            scenario = load_scenario(st)
            assert isinstance(scenario, Scenario)
            assert scenario.name
            assert scenario.description

    def test_normal_scenario_is_symmetric(self):
        scenario = load_scenario(ScenarioType.NORMAL)
        gt = scenario.tyre_state
        for corner in CORNERS:
            assert gt.tread_mm[corner] == 5.0

    def test_asymmetric_wear_scenario(self):
        scenario = load_scenario(ScenarioType.ASYMMETRIC_WEAR)
        gt = scenario.tyre_state
        assert gt.tread_mm["RR"] < gt.tread_mm["FL"]
        snapshot = gt.as_ground_truth_snapshot()
        assert snapshot.is_asymmetric

    def test_low_pressure_scenario(self):
        scenario = load_scenario(ScenarioType.LOW_PRESSURE)
        gt = scenario.tyre_state
        assert gt.pressure_kpa["RR"] < gt.pressure_kpa["FL"]

    def test_toe_misalignment_scenario(self):
        scenario = load_scenario(ScenarioType.TOE_MISALIGNMENT)
        gt = scenario.tyre_state
        assert gt.toe_sq_deg2 > 0

    def test_degradation_evolution(self):
        scenario = load_scenario(ScenarioType.ASYMMETRIC_WEAR)
        initial_rr = scenario.tyre_state.tread_mm["RR"]
        scenario.apply_degradation(1000.0)
        final_rr = scenario.tyre_state.tread_mm["RR"]
        assert final_rr < initial_rr


# ===========================================================================
# Test: Missing channels
# ===========================================================================

class TestMissingChannels:
    def test_missing_motor_torque(self):
        scenario = load_scenario(ScenarioType.SENSOR_MISSINGNESS)
        adapter = MockVehicleAdapter()
        adapter.reset(scenario)
        state = adapter.step(dt_s=1.0)
        mt = state.telemetry.motor_torque_nm
        assert mt.status == SensorStatus.MISSING
        assert mt.value is None

    def test_missing_temperature(self):
        scenario = load_scenario(ScenarioType.SENSOR_MISSINGNESS)
        adapter = MockVehicleAdapter()
        adapter.reset(scenario)
        state = adapter.step(dt_s=1.0)
        for corner in CORNERS:
            tc = state.telemetry.tpms_temperature_c[corner]
            assert tc.status == SensorStatus.MISSING
            assert tc.value is None

    def test_nonzero_not_substituted(self):
        """Missing channels must not be silently replaced with zero."""
        scenario = load_scenario(ScenarioType.SENSOR_MISSINGNESS)
        adapter = MockVehicleAdapter()
        adapter.reset(scenario)
        state = adapter.step(dt_s=1.0)
        # Verify that a MISSING reading's value is None, not 0.0
        mt = state.telemetry.motor_torque_nm
        assert mt.value is None  # NOT 0.0


# ===========================================================================
# Test: Estimator receives simulator telemetry correctly
# ===========================================================================

class TestEstimatorIntegration:
    def test_pipeline_runs_on_simulated_telemetry(self, mock_adapter, vehicle_config, tyre_config):
        """Full pipeline: simulator → features → estimator."""
        from src.evtyre.features.kinematics import extract as kin_extract
        from src.evtyre.features.pressure_thermal import extract as pt_extract
        from src.evtyre.features.road_load import extract as rl_extract, RoadLoadParams
        from src.evtyre.pipeline import Pipeline

        pipeline = Pipeline(vehicle_config, tyre_config)
        pipeline.register("kinematics", kin_extract)
        pipeline.register("pressure_thermal", pt_extract)
        road_load_params = RoadLoadParams(
            drag_coefficient=0.25,
            frontal_area_m2=2.3,
            driveline_efficiency=0.95,
        )
        pipeline.register("road_load", rl_extract, {"road_load_params": road_load_params})

        state = mock_adapter.step(dt_s=1.0)
        features, estimate = pipeline.run(state.telemetry)

        assert estimate is not None
        assert len(estimate.states) == 10  # Phase 3 state dimension
        assert estimate.n_measurements_available > 0
        assert not estimate.all_unobservable

    def test_estimates_are_physically_plausible(self, mock_adapter, vehicle_config, tyre_config):
        """Estimates should be within physical bounds."""
        from src.evtyre.features.kinematics import extract as kin_extract
        from src.evtyre.features.pressure_thermal import extract as pt_extract
        from src.evtyre.features.road_load import extract as rl_extract, RoadLoadParams
        from src.evtyre.pipeline import Pipeline

        pipeline = Pipeline(vehicle_config, tyre_config)
        pipeline.register("kinematics", kin_extract)
        pipeline.register("pressure_thermal", pt_extract)
        road_load_params = RoadLoadParams(
            drag_coefficient=0.25,
            frontal_area_m2=2.3,
            driveline_efficiency=0.95,
        )
        pipeline.register("road_load", rl_extract, {"road_load_params": road_load_params})

        state = mock_adapter.step(dt_s=1.0)
        features, estimate = pipeline.run(state.telemetry)

        for s in estimate.states:
            if "tread" in s.name:
                assert 0.5 <= s.value <= 9.0, f"{s.name} = {s.value} out of bounds"
            elif "press" in s.name:
                assert 50.0 <= s.value <= 500.0, f"{s.name} = {s.value} out of bounds"


# ===========================================================================
# Test: Replay
# ===========================================================================

class TestReplay:
    def test_replay_records_and_plays_back(self, mock_adapter):
        engine = ReplayEngine()
        for i in range(10):
            state = mock_adapter.step(dt_s=1.0)
            engine.record(state)
        engine.finalize()

        assert engine.frame_count == 10
        assert engine.total_duration_s > 0
        assert engine.total_distance_km > 0

    def test_replay_is_deterministic(self):
        """Two identical runs produce identical results."""
        def run_sim():
            adapter = MockVehicleAdapter()
            adapter.reset(load_scenario(ScenarioType.NORMAL))
            frames = []
            for _ in range(5):
                state = adapter.step(dt_s=1.0)
                frames.append(state.telemetry.motor_torque_nm.value)
            return frames

        run1 = run_sim()
        run2 = run_sim()
        assert run1 == run2

    def test_replay_frame_lookup(self, mock_adapter):
        engine = ReplayEngine()
        for _ in range(10):
            state = mock_adapter.step(dt_s=1.0)
            engine.record(state)
        engine.finalize()

        frame = engine.get_frame_at_distance(0.0)
        assert frame is not None

    def test_ground_truth_history(self, mock_adapter):
        engine = ReplayEngine()
        for _ in range(5):
            state = mock_adapter.step(dt_s=1.0)
            engine.record(state)
        engine.finalize()

        history = engine.get_ground_truth_history()
        assert len(history) == 5
        odom, treads = history[-1]
        assert odom > 0
        for corner in CORNERS:
            assert corner in treads
            assert treads[corner] > 0


# ===========================================================================
# Test: Ground truth vs estimate comparison
# ===========================================================================

class TestGroundTruthComparison:
    def test_ground_truth_snapshot(self, mock_adapter):
        gt = mock_adapter.get_ground_truth()
        assert isinstance(gt, GroundTruthSnapshot)
        for corner in CORNERS:
            assert corner in gt.tread_mm
            assert corner in gt.pressure_kpa

    def test_asymmetry_detection(self):
        gt = GroundTruthSnapshot(
            tread_mm={"FL": 5.0, "FR": 5.0, "RL": 4.5, "RR": 3.7},
            pressure_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        assert gt.is_asymmetric

        gt_sym = GroundTruthSnapshot(
            tread_mm={"FL": 5.0, "FR": 5.0, "RL": 5.0, "RR": 5.0},
            pressure_kpa={"FL": 240.0, "FR": 240.0, "RL": 240.0, "RR": 240.0},
        )
        assert not gt_sym.is_asymmetric


# ===========================================================================
# Test: Ground truth tyre model
# ===========================================================================

class TestGroundTruthTyre:
    def test_apply_wear(self):
        tyres = GroundTruthTyre()
        tyres.apply_wear("FL", -0.5)
        assert tyres.tread_mm["FL"] == 4.5

    def test_apply_wear_minimum(self):
        tyres = GroundTruthTyre()
        tyres.tread_mm["FL"] = 0.6
        tyres.apply_wear("FL", -1.0)
        assert tyres.tread_mm["FL"] == 0.5  # Clamped at minimum

    def test_apply_pressure_change(self):
        tyres = GroundTruthTyre()
        tyres.apply_pressure_change("RR", -20.0)
        assert tyres.pressure_kpa["RR"] == 220.0

    def test_apply_pressure_minimum(self):
        tyres = GroundTruthTyre()
        tyres.pressure_kpa["RR"] = 55.0
        tyres.apply_pressure_change("RR", -100.0)
        assert tyres.pressure_kpa["RR"] == 50.0  # Clamped at minimum


# ===========================================================================
# Test: CARLA adapter (import only, no connection)
# ===========================================================================

class TestCarlaAdapter:
    def test_carla_import_check(self):
        from src.evtyre.simulation.carla_adapter import CARLA_AVAILABLE
        # Just verify the flag exists; CARLA likely not installed
        assert isinstance(CARLA_AVAILABLE, bool)
