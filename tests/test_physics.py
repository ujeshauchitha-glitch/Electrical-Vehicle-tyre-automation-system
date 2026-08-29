"""Phase 3.2 physics model tests.

Covers the effective-rolling-radius chain that makes the within-axle tread
difference observable:

    r_free = r_belt + tread/1000
    k_z    = k_z0 * (P / P_placard)
    delta  = Fz / k_z
    r_eff  = r_free - D * delta

and the axle-ratio direction, which no other test would catch if inverted.
"""

import unittest

from evtyre.config.tyre import TyreConfig
from evtyre.config.vehicle import DriveLayout, VehicleConfig
from evtyre.estimation.estimator import PhysicsConfig, _predict, prior
from evtyre.estimation.physics import (
    corner_weight,
    effective_rolling_radius,
    rolling_resistance_coeff,
    toe_drag_from_sq,
)
from evtyre.estimation.state import MEAS, STATE

G = 9.80665


def _vehicle() -> VehicleConfig:
    return VehicleConfig(
        vehicle_id="test",
        mass_kg=1800.0,
        front_weight_fraction=0.48,
        drive_layout=DriveLayout.RWD,
    )


def _tyre() -> TyreConfig:
    return TyreConfig(
        tyre_model_id="test-tyre",
        wheel_belt_radius_m=0.322,
        tread_new_mm=8.0,
        tread_legal_mm=1.6,
        placard_pressure_kpa=240.0,
        cold_reference_temperature_c=25.0,
    )


def _physics() -> PhysicsConfig:
    return PhysicsConfig(
        k_z0=210_000.0,
        cornering_stiffness=55_000.0,
        tread_rr_span=0.20,
        c_rr0=0.0090,
        p_exponent=0.45,
        t_coeff=0.0015,
        deflection_factor=1.0 / 3.0,
    )


def _r_eff(tread_mm, p_kpa=240.0, fz_n=4000.0):
    ph, tc = _physics(), _tyre()
    return effective_rolling_radius(
        tread_mm, p_kpa, fz_n,
        tc.wheel_belt_radius_m, ph.k_z0, tc.placard_pressure_kpa,
        ph.deflection_factor,
    )


class CornerWeightTests(unittest.TestCase):
    def test_front_corner_load(self):
        vc = _vehicle()
        expected = vc.mass_kg * G * vc.front_weight_fraction / 2.0
        self.assertAlmostEqual(corner_weight(vc, "FL"), expected, places=6)
        self.assertAlmostEqual(corner_weight(vc, "FR"), expected, places=6)

    def test_rear_corner_load(self):
        vc = _vehicle()
        expected = vc.mass_kg * G * (1.0 - vc.front_weight_fraction) / 2.0
        self.assertAlmostEqual(corner_weight(vc, "RL"), expected, places=6)
        self.assertAlmostEqual(corner_weight(vc, "RR"), expected, places=6)

    def test_all_four_corners_sum_to_vehicle_weight(self):
        vc = _vehicle()
        total = sum(corner_weight(vc, c) for c in ("FL", "FR", "RL", "RR"))
        self.assertAlmostEqual(total, vc.mass_kg * G, places=6)


class EffectiveRollingRadiusTests(unittest.TestCase):
    def test_more_tread_gives_larger_radius(self):
        self.assertGreater(_r_eff(6.0), _r_eff(4.0))

    def test_tread_enters_as_millimetres(self):
        """1 mm more tread must raise the free radius by exactly 1 mm."""
        self.assertAlmostEqual(_r_eff(5.0) - _r_eff(4.0), 0.001, places=9)

    def test_higher_pressure_gives_larger_radius(self):
        """Higher pressure -> stiffer -> less deflection -> larger r_eff."""
        self.assertGreater(_r_eff(5.0, p_kpa=280.0), _r_eff(5.0, p_kpa=200.0))

    def test_heavier_load_gives_smaller_radius(self):
        self.assertLess(_r_eff(5.0, fz_n=6000.0), _r_eff(5.0, fz_n=3000.0))

    def test_matches_the_closed_form(self):
        """r_eff = r_b + d/1000 - D * Fz*P0/(k_z0*P)."""
        ph, tc = _physics(), _tyre()
        d, p, fz = 5.0, 220.0, 4200.0
        expected = (
            tc.wheel_belt_radius_m
            + d / 1000.0
            - ph.deflection_factor * (fz * tc.placard_pressure_kpa) / (ph.k_z0 * p)
        )
        self.assertAlmostEqual(_r_eff(d, p, fz), expected, places=12)

    def test_deflection_factor_is_required(self):
        """No default: the constant must have exactly one home, PhysicsConfig."""
        tc, ph = _tyre(), _physics()
        with self.assertRaises(TypeError):
            effective_rolling_radius(
                5.0, 240.0, 4000.0,
                tc.wheel_belt_radius_m, ph.k_z0, tc.placard_pressure_kpa,
            )


class RollingResistanceTests(unittest.TestCase):
    def test_worn_tread_lowers_rolling_resistance(self):
        """Project rule: worn tyres roll MORE freely, not less."""
        tc, ph = _tyre(), _physics()
        worn = rolling_resistance_coeff(2.0, 240.0, 25.0, tc, ph)
        new = rolling_resistance_coeff(8.0, 240.0, 25.0, tc, ph)
        self.assertLess(worn, new)

    def test_lower_pressure_raises_rolling_resistance(self):
        tc, ph = _tyre(), _physics()
        soft = rolling_resistance_coeff(5.0, 180.0, 25.0, tc, ph)
        hard = rolling_resistance_coeff(5.0, 280.0, 25.0, tc, ph)
        self.assertGreater(soft, hard)

    def test_temperature_term_is_applied(self):
        tc, ph = _tyre(), _physics()
        cold = rolling_resistance_coeff(5.0, 240.0, 10.0, tc, ph)
        hot = rolling_resistance_coeff(5.0, 240.0, 40.0, tc, ph)
        self.assertNotAlmostEqual(cold, hot, places=9)


class ToeDragTests(unittest.TestCase):
    def test_zero_toe_gives_zero_drag(self):
        self.assertAlmostEqual(toe_drag_from_sq(0.0, _physics()), 0.0, places=12)

    def test_drag_is_linear_in_toe_squared(self):
        ph = _physics()
        self.assertAlmostEqual(
            toe_drag_from_sq(2.0, ph), 2.0 * toe_drag_from_sq(1.0, ph), places=9
        )

    def test_drag_is_non_negative(self):
        for toe_sq in (0.0, 0.1, 1.0, 4.0):
            self.assertGreaterEqual(toe_drag_from_sq(toe_sq, _physics()), 0.0)


class PhysicsConfigTests(unittest.TestCase):
    def test_every_field_is_required(self):
        with self.assertRaises(TypeError):
            PhysicsConfig()  # type: ignore[call-arg]

    def test_partial_construction_is_rejected(self):
        with self.assertRaises(TypeError):
            PhysicsConfig(k_z0=210_000.0)  # type: ignore[call-arg]


class AxleRatioDirectionTests(unittest.TestCase):
    """The direction check that no other test would catch.

    The measurement is omega_left/omega_right, and omega is inversely
    proportional to rolling radius, so the prediction must be r_right/r_left.
    If this is inverted, every tread difference silently carries the wrong sign.
    """

    def _predict_with(self, tread_fl, tread_fr):
        tc, ph, vc = _tyre(), _physics(), _vehicle()
        x, _ = prior(tc)
        x = x.copy()
        x[STATE.tread_fl] = tread_fl
        x[STATE.tread_fr] = tread_fr
        return _predict(x, tc, ph, 30.0, vc)

    def test_more_tread_on_left_predicts_ratio_below_one(self):
        """Bigger left radius -> left wheel turns SLOWER -> omega_L/omega_R < 1."""
        z = self._predict_with(tread_fl=6.0, tread_fr=4.0)
        self.assertLess(z[MEAS.ratio_front], 1.0)

    def test_more_tread_on_right_predicts_ratio_above_one(self):
        z = self._predict_with(tread_fl=4.0, tread_fr=6.0)
        self.assertGreater(z[MEAS.ratio_front], 1.0)

    def test_equal_tread_predicts_unity(self):
        z = self._predict_with(tread_fl=5.0, tread_fr=5.0)
        self.assertAlmostEqual(z[MEAS.ratio_front], 1.0, places=9)

    def test_prediction_equals_r_right_over_r_left(self):
        tc, ph, vc = _tyre(), _physics(), _vehicle()
        x, _ = prior(tc)
        x = x.copy()
        x[STATE.tread_fl], x[STATE.tread_fr] = 6.0, 4.0
        r_l = effective_rolling_radius(
            6.0, x[STATE.press_fl], corner_weight(vc, "FL"),
            tc.wheel_belt_radius_m, ph.k_z0, tc.placard_pressure_kpa,
            ph.deflection_factor,
        )
        r_r = effective_rolling_radius(
            4.0, x[STATE.press_fr], corner_weight(vc, "FR"),
            tc.wheel_belt_radius_m, ph.k_z0, tc.placard_pressure_kpa,
            ph.deflection_factor,
        )
        z = _predict(x, tc, ph, 30.0, vc)
        self.assertAlmostEqual(z[MEAS.ratio_front], r_r / r_l, places=12)


class NoToeInRoadLoadPredictionTests(unittest.TestCase):
    """The predicted road load must NOT contain a toe term.

    The Phase 2 road_load_coefficient is mean(C_rr) recomputed from pressure and
    carries no toe. Predicting a toe term against it drives toe^2 -> 0 and
    labels toe OBSERVED on zero toe information.
    """

    def test_toe_does_not_change_the_road_load_prediction(self):
        tc, ph, vc = _tyre(), _physics(), _vehicle()
        x, _ = prior(tc)
        a, b = x.copy(), x.copy()
        a[STATE.toe_sq] = 0.0
        b[STATE.toe_sq] = 4.0
        self.assertAlmostEqual(
            _predict(a, tc, ph, 30.0, vc)[MEAS.roadload],
            _predict(b, tc, ph, 30.0, vc)[MEAS.roadload],
            places=12,
        )

    def test_toe_has_no_effect_on_any_predicted_channel(self):
        tc, ph, vc = _tyre(), _physics(), _vehicle()
        x, _ = prior(tc)
        a, b = x.copy(), x.copy()
        a[STATE.toe_sq] = 0.0
        b[STATE.toe_sq] = 4.0
        za, zb = _predict(a, tc, ph, 30.0, vc), _predict(b, tc, ph, 30.0, vc)
        for i in range(MEAS.N):
            with self.subTest(channel=i):
                self.assertAlmostEqual(za[i], zb[i], places=12)


if __name__ == "__main__":
    unittest.main()
