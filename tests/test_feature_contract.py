"""Tests for the Phase 2 Feature contract.

These verify the双向 invariant:
- OK ⇒ value is not None
- UNAVAILABLE ⇒ value is None AND unavailable_reason is non-empty
- OUT_OF_RANGE ⇒ value is not None
"""

import unittest

from evtyre.features.contract import (
    Classification,
    Directionality,
    Feature,
    FeatureStatus,
)


def _valid_feature(**overrides) -> dict:
    """Return a dict that satisfies the Feature contract invariants."""
    defaults = dict(
        name="test_feature",
        value=1.0,
        unit="Pa",
        status=FeatureStatus.OK,
        unavailable_reason=None,
        directionality=Directionality.NATURAL,
        classification=Classification.A,
        inputs=("tpms_pressure_kpa",),
        corner="FL",
        timestamp_s=0.0,
        provenance="simulated",
        extractor_version="test",
    )
    defaults.update(overrides)
    return defaults


class FeatureStatusInvariantTests(unittest.TestCase):
    def test_ok_with_value_succeeds(self):
        f = Feature(**_valid_feature())
        self.assertEqual(f.status, FeatureStatus.OK)
        self.assertIsNotNone(f.value)

    def test_ok_with_none_value_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Feature(**_valid_feature(value=None))
        self.assertIn("OK but value is None", str(ctx.exception))

    def test_unavailable_with_none_and_reason_succeeds(self):
        f = Feature(**_valid_feature(
            value=None,
            status=FeatureStatus.UNAVAILABLE,
            unavailable_reason="sensor missing",
        ))
        self.assertEqual(f.status, FeatureStatus.UNAVAILABLE)
        self.assertIsNone(f.value)

    def test_unavailable_with_value_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Feature(**_valid_feature(
                value=42.0,
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason="sensor missing",
            ))
        self.assertIn("UNAVAILABLE but value is", str(ctx.exception))

    def test_unavailable_without_reason_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Feature(**_valid_feature(
                value=None,
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason=None,
            ))
        self.assertIn("unavailable_reason is missing or empty", str(ctx.exception))

    def test_unavailable_with_empty_reason_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Feature(**_valid_feature(
                value=None,
                status=FeatureStatus.UNAVAILABLE,
                unavailable_reason="",
            ))
        self.assertIn("unavailable_reason is missing or empty", str(ctx.exception))

    def test_out_of_range_with_value_succeeds(self):
        f = Feature(**_valid_feature(
            value=999.0,
            status=FeatureStatus.OUT_OF_RANGE,
        ))
        self.assertEqual(f.status, FeatureStatus.OUT_OF_RANGE)
        self.assertEqual(f.value, 999.0)

    def test_out_of_range_with_none_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Feature(**_valid_feature(
                value=None,
                status=FeatureStatus.OUT_OF_RANGE,
            ))
        self.assertIn("OUT_OF_RANGE but value is None", str(ctx.exception))


class FeatureCornerTests(unittest.TestCase):
    def test_valid_corner_succeeds(self):
        f = Feature(**_valid_feature(corner="FL"))
        self.assertEqual(f.corner, "FL")

    def test_none_corner_succeeds(self):
        f = Feature(**_valid_feature(corner=None))
        self.assertIsNone(f.corner)

    def test_invalid_corner_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Feature(**_valid_feature(corner="CENTER"))
        self.assertIn("not a member of CORNERS", str(ctx.exception))


class FeatureInputsTests(unittest.TestCase):
    def test_empty_inputs_succeeds(self):
        # Features derived purely from config (no telemetry channels) may
        # legitimately have an empty inputs tuple.
        f = Feature(**_valid_feature(inputs=()))
        self.assertEqual(f.inputs, ())

    def test_non_empty_inputs_succeeds(self):
        f = Feature(**_valid_feature(inputs=("tpms_pressure_kpa",)))
        self.assertEqual(f.inputs, ("tpms_pressure_kpa",))


class FeatureFrozenTests(unittest.TestCase):
    def test_feature_is_frozen(self):
        f = Feature(**_valid_feature())
        with self.assertRaises(AttributeError):
            f.name = "changed"  # type: ignore[misc]


class FeatureTupleOutputTests(unittest.TestCase):
    """Verify extractors can return tuples of Features as specified."""

    def test_multiple_features_in_tuple(self):
        features = (
            Feature(**_valid_feature(name="f1")),
            Feature(**_valid_feature(name="f2", corner="FR")),
            Feature(**_valid_feature(name="f3", corner=None)),
        )
        self.assertEqual(len(features), 3)
        self.assertEqual(features[0].corner, "FL")
        self.assertEqual(features[1].corner, "FR")
        self.assertIsNone(features[2].corner)


if __name__ == "__main__":
    unittest.main()
