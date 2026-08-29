import unittest

from evtyre.config import DriveLayout, TyreConfig, VehicleConfig


class VehicleConfigTests(unittest.TestCase):
    def test_valid_config_constructs(self):
        cfg = VehicleConfig(
            vehicle_id="example-1",
            mass_kg=1800.0,
            front_weight_fraction=0.48,
            drive_layout=DriveLayout.AWD,
        )
        self.assertEqual(cfg.drive_layout, DriveLayout.AWD)

    def test_rejects_non_positive_mass(self):
        with self.assertRaises(ValueError):
            VehicleConfig(
                vehicle_id="bad",
                mass_kg=0.0,
                front_weight_fraction=0.5,
                drive_layout=DriveLayout.FWD,
            )

    def test_rejects_out_of_bounds_weight_fraction(self):
        with self.assertRaises(ValueError):
            VehicleConfig(
                vehicle_id="bad",
                mass_kg=1800.0,
                front_weight_fraction=1.2,
                drive_layout=DriveLayout.FWD,
            )

    def test_rejects_empty_vehicle_id(self):
        with self.assertRaises(ValueError):
            VehicleConfig(
                vehicle_id="",
                mass_kg=1800.0,
                front_weight_fraction=0.5,
                drive_layout=DriveLayout.FWD,
            )


class TyreConfigTests(unittest.TestCase):
    def test_valid_config_constructs(self):
        cfg = TyreConfig(
            tyre_model_id="example-tyre",
            wheel_belt_radius_m=0.322,
            tread_new_mm=8.0,
            tread_legal_mm=1.6,
            placard_pressure_kpa=240.0,
            cold_reference_temperature_c=25.0,
        )
        self.assertEqual(cfg.tread_new_mm, 8.0)

    def test_rejects_tread_new_not_greater_than_legal(self):
        with self.assertRaises(ValueError):
            TyreConfig(
                tyre_model_id="bad",
                wheel_belt_radius_m=0.322,
                tread_new_mm=1.6,
                tread_legal_mm=1.6,
                placard_pressure_kpa=240.0,
            cold_reference_temperature_c=25.0,
            )

    def test_rejects_non_positive_placard_pressure(self):
        with self.assertRaises(ValueError):
            TyreConfig(
                tyre_model_id="bad",
                wheel_belt_radius_m=0.322,
                tread_new_mm=8.0,
                tread_legal_mm=1.6,
                placard_pressure_kpa=0.0,
            cold_reference_temperature_c=25.0,
            )


if __name__ == "__main__":
    unittest.main()
