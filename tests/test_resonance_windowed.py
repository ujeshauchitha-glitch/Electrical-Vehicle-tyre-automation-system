import unittest

import numpy as np

from evtyre.features.contract import FeatureStatus
from evtyre.features.resonance_windowed import estimate_peak, extract
from evtyre.schema.common import CORNERS
from evtyre.signal.windowed import WindowedSample
from evtyre.simulation.wheel_speed_synth import synth_window


def _freqs(base):
    return {c: base + i * 1.7 for i, c in enumerate(CORNERS)}


class RecoveryTests(unittest.TestCase):
    def test_first_mode_hypothesis_recovered(self):
        truth = _freqs(48.0)
        feats = extract(synth_window(truth, 1000.0, duration_s=8.0, seed=1), band_hz=(30.0, 80.0))
        for f in feats:
            self.assertIs(f.status, FeatureStatus.OK)
            self.assertAlmostEqual(f.value, truth[f.corner], delta=1.0)

    def test_second_mode_hypothesis_recovered(self):
        truth = _freqs(180.0)
        feats = extract(synth_window(truth, 1000.0, duration_s=8.0, seed=2), band_hz=(100.0, 300.0))
        for f in feats:
            self.assertAlmostEqual(f.value, truth[f.corner], delta=3.0)

    def test_seeded_runs_are_reproducible(self):
        a = synth_window(_freqs(50.0), 1000.0, seed=3)
        b = synth_window(_freqs(50.0), 1000.0, seed=3)
        np.testing.assert_array_equal(a.data, b.data)

    def test_accuracy_over_seeds(self):
        errs = []
        for seed in range(20):
            truth = _freqs(55.0)
            for f in extract(synth_window(truth, 1000.0, duration_s=8.0, seed=seed), (30.0, 80.0)):
                errs.append(abs(f.value - truth[f.corner]))
        self.assertLess(np.sqrt(np.mean(np.square(errs))), 0.5)


class HonestyTests(unittest.TestCase):
    def test_pure_noise_is_unavailable_not_a_number(self):
        rng = np.random.default_rng(0)
        n = 2000
        names = tuple(f"wheel_speed_{c}" for c in CORNERS)
        sample = WindowedSample(rng.normal(70, 0.03, (n, 4)), names, 1000.0, 0.0,
                                "simulated", "noise")
        for f in extract(sample, (30.0, 80.0), min_snr_db=15.0):
            self.assertIs(f.status, FeatureStatus.UNAVAILABLE)
            self.assertIsNone(f.value)

    def test_too_many_missing_samples_unavailable(self):
        s = synth_window(_freqs(50.0), 1000.0, missing_fraction=0.35, seed=4)
        for f in extract(s, (30.0, 80.0)):
            self.assertIs(f.status, FeatureStatus.UNAVAILABLE)

    def test_mid_window_gap_not_interpolated(self):
        s = synth_window(_freqs(50.0), 1000.0, gap=(0.9, 1.1), seed=5)
        for f in extract(s, (30.0, 80.0)):
            self.assertIs(f.status, FeatureStatus.UNAVAILABLE)
            self.assertIn("gap", f.unavailable_reason)

    def test_leading_gap_shortens_window(self):
        truth = _freqs(50.0)
        s = synth_window(truth, 1000.0, gap=(0.0, 0.2), seed=6)
        for f in extract(s, (30.0, 80.0)):
            self.assertIs(f.status, FeatureStatus.OK)

    def test_band_above_nyquist_rejected(self):
        with self.assertRaises(ValueError):
            estimate_peak(np.zeros(500), 500.0, (100.0, 300.0))

    def test_second_mode_band_at_500hz_rate_is_unavailable(self):
        s = synth_window(_freqs(180.0), 500.0, seed=7)
        for f in extract(s, (100.0, 300.0)):
            self.assertIs(f.status, FeatureStatus.UNAVAILABLE)

    def test_aliased_tone_creates_false_peak_when_unfiltered(self):
        """An 880 Hz tone folds to 120 Hz at 1 kHz: why anti-alias metadata matters."""
        s = synth_window(_freqs(48.0), 1000.0, alias_hz=880.0, amplitude_rad_s=0.02,
                         seed=8)
        peak = estimate_peak(s.channel("wheel_speed_FL")[0], 1000.0, (100.0, 150.0))
        self.assertAlmostEqual(peak.frequency_hz, 120.0, delta=1.5)

    def test_window_validation(self):
        with self.assertRaises(ValueError):
            WindowedSample(np.zeros((10, 2)), ("a",), 100.0, 0.0, "simulated", "x")
        with self.assertRaises(ValueError):
            WindowedSample(np.zeros((10, 1)), ("a",), 100.0, 0.0, "simulated", "x",
                           anti_alias_cutoff_hz=80.0)


if __name__ == "__main__":
    unittest.main()
