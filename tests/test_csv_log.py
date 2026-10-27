import tempfile
import unittest
from pathlib import Path

from evtyre.logio.csv_log import CsvLogSource
from evtyre.schema.common import SensorStatus

HEADER = ("timestamp_s,wheel_speed_rpm_FL,wheel_speed_rpm_FR,wheel_speed_rpm_RL,"
          "wheel_speed_rpm_RR,tpms_pressure_psi_FL,motor_torque_nm,vehicle_speed_kmh\n")


class CsvLogTests(unittest.TestCase):
    def _log(self, body, header=HEADER):
        d = tempfile.mkdtemp()
        p = Path(d) / "log.csv"
        p.write_text(header + body, encoding="utf-8")
        return CsvLogSource(p)

    def test_reads_frames_labelled_replay(self):
        src = self._log("0.0,600,601,599,600,35,120,80\n0.1,600,,599,600,35,nan,80\n")
        f0, f1 = src.read_frame(), src.read_frame()
        self.assertEqual(f0.source, "replay")
        self.assertTrue(f0.wheel_speed_rad_s["FL"].is_usable)
        self.assertIs(f1.wheel_speed_rad_s["FR"].status, SensorStatus.MISSING)
        self.assertIs(f1.motor_torque_nm.status, SensorStatus.MISSING)
        self.assertIsNone(src.read_frame())

    def test_absent_columns_are_missing_not_defaulted(self):
        f = self._log("0.0,600,600,600,600,35,120,80\n").read_frame()
        self.assertIs(f.tpms_pressure_kpa["RR"].status, SensorStatus.MISSING)
        self.assertIs(f.odometer_km.status, SensorStatus.MISSING)

    def test_requires_timestamp(self):
        with self.assertRaises(ValueError):
            self._log("1\n", header="a\n").read_raw()

    def test_rows_without_timestamp_skipped(self):
        src = self._log(",1,2,3,4,5,6,7\n1.0,600,600,600,600,35,120,80\n")
        self.assertEqual(src.read_raw().timestamp_s, 1.0)


if __name__ == "__main__":
    unittest.main()
