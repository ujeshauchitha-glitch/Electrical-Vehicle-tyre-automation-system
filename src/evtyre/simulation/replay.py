"""Replay engine — deterministic replay of simulation runs.

Records SimulationState objects during a simulation run and allows
playback with consistent results. Same scenario + same adapter
= same replay, every time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .interface import SimulationState


@dataclass
class ReplayFrame:
    """A single recorded frame for replay."""
    state: SimulationState
    step_index: int


@dataclass
class ReplayEngine:
    """Records and replays simulation runs.

    Usage:
        engine = ReplayEngine()
        engine.record(state)  # during simulation
        engine.finalize()

        # Later:
        for frame in engine.frames:
            process(frame.state)
    """

    _frames: list[ReplayFrame] = field(default_factory=list)
    _finalized: bool = False

    def record(self, state: SimulationState) -> None:
        """Record a simulation state."""
        if self._finalized:
            raise RuntimeError("Cannot record to a finalized replay")
        self._frames.append(ReplayFrame(
            state=state,
            step_index=state.step_index,
        ))

    def finalize(self) -> None:
        """Mark replay as complete (no more recording)."""
        self._finalized = True

    @property
    def frames(self) -> list[ReplayFrame]:
        """Return all recorded frames in order."""
        return list(self._frames)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def total_duration_s(self) -> float:
        if not self._frames:
            return 0.0
        return self._frames[-1].state.timestamp_s

    @property
    def total_distance_km(self) -> float:
        if not self._frames:
            return 0.0
        return self._frames[-1].state.odometer_km

    def get_frame_at_time(self, timestamp_s: float) -> ReplayFrame | None:
        """Find the frame closest to a given timestamp."""
        if not self._frames:
            return None
        best = min(self._frames, key=lambda f: abs(f.state.timestamp_s - timestamp_s))
        return best

    def get_frame_at_distance(self, odometer_km: float) -> ReplayFrame | None:
        """Find the frame closest to a given odometer reading."""
        if not self._frames:
            return None
        best = min(self._frames, key=lambda f: abs(f.state.odometer_km - odometer_km))
        return best

    def get_ground_truth_history(self) -> list[tuple[float, dict[str, float]]]:
        """Return (odometer_km, tread_mm_dict) for each frame."""
        return [
            (f.state.odometer_km, dict(f.state.ground_truth.tread_mm))
            for f in self._frames
        ]

    def get_estimates_history(self) -> list[tuple[float, dict[str, float]]]:
        """Placeholder — actual estimates are stored externally."""
        return []
