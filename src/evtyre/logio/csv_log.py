"""CSV log replay: decoded vehicle or bench logs -> RawTelemetrySample.

Expected columns (native units, empty cell = missing): ``timestamp_s``,
``wheel_speed_rpm_{FL,FR,RL,RR}``, ``tpms_pressure_psi_{..}``,
``tpms_temperature_f_{..}``, ``motor_torque_nm``, ``motor_speed_rpm``,
``accel_long_g``, ``ambient_temp_c``, ``vehicle_speed_kmh``, ``odometer_km``.
Absent columns are treated as missing channels. This reads an already-decoded
log; decoding raw CAN frames with a DBC is a separate step (see
docs/HARDWARE-TEST-PLAN.md). Frames from this source are labelled "replay".
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Optional

from ..schema.common import CORNERS
from ..ingest.interface import TelemetrySource
from ..ingest.raw import RawTelemetrySample

_SCALARS = ("motor_torque_nm", "motor_speed_rpm", "accel_long_g",
            "ambient_temp_c", "vehicle_speed_kmh", "odometer_km")
_PER_CORNER = ("wheel_speed_rpm", "tpms_pressure_psi", "tpms_temperature_f")


def _number(cell: Optional[str]) -> Optional[float]:
    if cell is None or cell.strip() == "":
        return None
    try:
        value = float(cell)
    except ValueError:
        return None
    return value if value == value and abs(value) != float("inf") else None


class CsvLogSource(TelemetrySource):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._rows: Iterator[dict[str, str]] | None = None
        self._file = None

    def source_label(self) -> str:
        return "replay"

    def _open(self) -> Iterator[dict[str, str]]:
        self._file = self._path.open(newline="", encoding="utf-8")
        reader = csv.DictReader(self._file)
        if reader.fieldnames is None or "timestamp_s" not in reader.fieldnames:
            self._file.close()
            raise ValueError("log must have a timestamp_s column")
        return reader

    def read_raw(self) -> Optional[RawTelemetrySample]:
        if self._rows is None:
            self._rows = self._open()
        for row in self._rows:
            t = _number(row.get("timestamp_s"))
            if t is None:
                continue
            return RawTelemetrySample(
                timestamp_s=t,
                **{
                    channel: {c: _number(row.get(f"{channel}_{c}")) for c in CORNERS}
                    for channel in _PER_CORNER
                },
                **{name: _number(row.get(name)) for name in _SCALARS},
            )
        if self._file is not None:
            self._file.close()
        return None
