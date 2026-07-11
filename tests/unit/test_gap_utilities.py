import unittest
from datetime import datetime

import igraph as ig

from temporal_networks._gap_utilities import (
    parse_flexible_datetime,
    format_large_numbers,
    detect_temporal_gaps,
    _infer_unit_and_threshold,
    _vertex_keys,
)

class TestFormatLargeNumbers(unittest.TestCase):
    def test_format_small_numbers(self):
        """Test formatting of numbers less than 1,000."""
        self.assertEqual(format_large_numbers(0, 0), "0")
        self.assertEqual(format_large_numbers(500, 0), "500")
        self.assertEqual(format_large_numbers(999, 0), "999")

    def test_format_thousands(self):
        """Test formatting of numbers in the thousands (k)."""
        self.assertEqual(format_large_numbers(1_000, 0), "1.0k")
        self.assertEqual(format_large_numbers(1_500, 0), "1.5k")
        self.assertEqual(format_large_numbers(999_999, 0), "1000.0k") # Based on logic, < 1_000_000 is k

    def test_format_millions(self):
        """Test formatting of numbers in the millions (M)."""
        self.assertEqual(format_large_numbers(1_000_000, 0), "1.0M")
        self.assertEqual(format_large_numbers(2_500_000, 0), "2.5M")
        self.assertEqual(format_large_numbers(999_999_999, 0), "1000.0M") # Based on logic

    def test_format_billions(self):
        """Test formatting of numbers in the billions (B)."""
        self.assertEqual(format_large_numbers(1_000_000_000, 0), "1.0B")
        self.assertEqual(format_large_numbers(3_500_000_000, 0), "3.5B")
        self.assertEqual(format_large_numbers(10_000_000_000, 0), "10.0B")

    def test_format_negative_numbers(self):
        """Test formatting of negative numbers (current logic treats as < 1000)."""
        self.assertEqual(format_large_numbers(-500, 0), "-500")
        self.assertEqual(format_large_numbers(-1_000_000, 0), "-1000000")

    def test_format_floats(self):
        """Test formatting of floating point inputs."""
        self.assertEqual(format_large_numbers(1500.5, 0), "1.5k")
        self.assertEqual(format_large_numbers(999.9, 0), "1000") # 999.9 formatted with .0f is 1000


class TestParseFlexibleDatetime(unittest.TestCase):
    def test_parse_flexible_datetime_valid_formats(self):
        # YYYY-MM
        self.assertEqual(parse_flexible_datetime("2024-03"), datetime(2024, 3, 1))

        # YYYY-MM-DD
        self.assertEqual(parse_flexible_datetime("2024-03-15"), datetime(2024, 3, 15))

        # YYYY-W## (First day of ISO week 12 in 2024 is Monday, March 18)
        self.assertEqual(parse_flexible_datetime("2024-W12"), datetime(2024, 3, 18))

    def test_parse_iso_weeks_at_year_boundaries(self):
        """ISO week years differ from calendar years around January.

        Regression test: parsing with the non-ISO %Y/%W directives placed
        2025-W01 at Jan 6 instead of Dec 30, so continuous weekly data
        reported a false gap at every year boundary.
        """
        # ISO 2025-W01 starts Monday 2024-12-30 (calendar year mismatch)
        self.assertEqual(parse_flexible_datetime("2025-W01"),
                         datetime(2024, 12, 30))
        self.assertEqual(parse_flexible_datetime("2026-W01"),
                         datetime(2025, 12, 29))
        # 2020 is a 53-week ISO year
        self.assertEqual(parse_flexible_datetime("2020-W53"),
                         datetime(2020, 12, 28))
        # 2024-W01 starts exactly on Jan 1 (a Monday)
        self.assertEqual(parse_flexible_datetime("2024-W01"),
                         datetime(2024, 1, 1))

        # YYYY-Q# (First day of quarter 2 is April 1)
        self.assertEqual(parse_flexible_datetime("2024-Q2"), datetime(2024, 4, 1))

        # YYYY
        self.assertEqual(parse_flexible_datetime("2024"), datetime(2024, 1, 1))

        # Leading and trailing spaces
        self.assertEqual(parse_flexible_datetime("  2024-03  "), datetime(2024, 3, 1))
        self.assertEqual(parse_flexible_datetime("2024-03-15 "), datetime(2024, 3, 15))
        self.assertEqual(parse_flexible_datetime(" 2024-Q2"), datetime(2024, 4, 1))
        self.assertEqual(parse_flexible_datetime("2024 "), datetime(2024, 1, 1))

    def test_parse_flexible_datetime_invalid_fallback(self):
        # Invalid format
        self.assertIsNone(parse_flexible_datetime("invalid"))

        # Valid format but invalid date values
        self.assertIsNone(parse_flexible_datetime("2024-13-45"))

        # Another invalid string
        self.assertIsNone(parse_flexible_datetime("not-a-date"))

        # Empty string
        self.assertIsNone(parse_flexible_datetime(""))

class TestGapDetectionLabelFormats(unittest.TestCase):
    """Gap detection must adapt to the label format, not assume monthly data."""

    def test_infer_unit_and_threshold(self):
        self.assertEqual(_infer_unit_and_threshold(["2024-01", "2024-02"]),
                         ("months", 1))
        self.assertEqual(_infer_unit_and_threshold(["2024-01-01", "2024-01-02"]),
                         ("days", 1))
        self.assertEqual(_infer_unit_and_threshold(["2024-W01", "2024-W02"]),
                         ("weeks", 1))
        self.assertEqual(_infer_unit_and_threshold(["2024-Q1", "2024-Q2"]),
                         ("months", 3))
        self.assertEqual(_infer_unit_and_threshold(["2020", "2021"]),
                         ("years", 1))
        # Unparseable labels fall back to the monthly default
        self.assertEqual(_infer_unit_and_threshold(["phase1", "phase2"]),
                         ("months", 1))

    def test_regularly_spaced_series_report_no_gaps(self):
        """A consecutive series of any supported format must report 0 gaps."""
        for labels in (
            ["2024-01", "2024-02", "2024-03", "2024-04"],          # monthly
            ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"],          # quarterly
            ["2020", "2021", "2022", "2023"],                      # yearly
            ["2024-03-01", "2024-03-02", "2024-03-03"],            # daily
            ["2024-W01", "2024-W02", "2024-W03"],                  # weekly
            # weekly across a year boundary (ISO year != calendar year)
            ["2024-W51", "2024-W52", "2025-W01", "2025-W02"],
            # weekly across a 53-week ISO year
            ["2020-W52", "2020-W53", "2021-W01", "2021-W02"],
        ):
            with self.subTest(labels=labels):
                self.assertFalse(detect_temporal_gaps(labels)["has_gaps"])

    def test_skipped_week_at_year_boundary_is_detected(self):
        """A genuinely missing week near New Year must still be flagged."""
        result = detect_temporal_gaps(["2024-W51", "2024-W52",
                                       "2025-W02", "2025-W03"])
        self.assertEqual(result["num_gaps"], 1)
        self.assertEqual(result["gaps"][0]["gap_size"], 2.0)

    def test_skipped_period_is_detected(self):
        # Skipped quarter (Q2 missing)
        self.assertEqual(
            detect_temporal_gaps(["2024-Q1", "2024-Q3", "2024-Q4"])["num_gaps"], 1)
        # Skipped year
        self.assertEqual(
            detect_temporal_gaps(["2020", "2022", "2023"])["num_gaps"], 1)
        # Sub-monthly gap that the old months-only logic missed
        self.assertEqual(
            detect_temporal_gaps(
                ["2024-03-01", "2024-03-02", "2024-03-20"])["num_gaps"], 1)

    def test_explicit_unit_overrides_inference(self):
        # Forcing months on yearly labels reproduces the old (coarser) behavior
        info = detect_temporal_gaps(["2020", "2021"], unit="months",
                                    gap_threshold=1)
        self.assertTrue(info["has_gaps"])  # 12 months apart > threshold 1


class TestUnparseableLabelWarning(unittest.TestCase):
    """Disabled gap detection must be loud, not silent (regression)."""

    def test_unparseable_labels_warn(self):
        """User-supplied labels that fail to parse emit a UserWarning."""
        with self.assertWarns(UserWarning) as cm:
            info = detect_temporal_gaps(["Jan 2024", "Feb 2024"])
        self.assertFalse(info["has_gaps"])
        self.assertIn("gap detection is DISABLED", str(cm.warning))
        self.assertIn("Jan 2024", str(cm.warning))

    def test_mixed_labels_warn(self):
        """One bad label among good ones still disables detection loudly."""
        with self.assertWarns(UserWarning):
            info = detect_temporal_gaps(["2024-01", "2024-XX", "2024-03"])
        self.assertFalse(info["has_gaps"])

    def test_default_placeholder_labels_stay_silent(self):
        """Auto-generated 'Graph N' labels mean no dates were supplied —
        disabling gap detection then is expected, so no warning."""
        import warnings as w
        with w.catch_warnings():
            w.simplefilter("error")  # any warning would fail the test
            info = detect_temporal_gaps(["Graph 1", "Graph 2", "Graph 3"])
        self.assertFalse(info["has_gaps"])

    def test_parseable_labels_stay_silent(self):
        import warnings as w
        with w.catch_warnings():
            w.simplefilter("error")
            info = detect_temporal_gaps(["2024-01", "2024-02"])
        self.assertFalse(info["has_gaps"])


class TestVertexKeys(unittest.TestCase):
    """Vertex identity keys: name > label > integer index."""

    def test_name_attribute_preferred(self):
        g = ig.Graph(n=3)
        g.vs["name"] = ["a", "b", "c"]
        self.assertEqual(_vertex_keys(g), ["a", "b", "c"])

    def test_label_used_when_no_name(self):
        g = ig.Graph(n=3)
        g.vs["label"] = ["x", "y", "z"]
        self.assertEqual(_vertex_keys(g), ["x", "y", "z"])

    def test_name_wins_over_label(self):
        g = ig.Graph(n=2)
        g.vs["name"] = ["n0", "n1"]
        g.vs["label"] = ["l0", "l1"]
        self.assertEqual(_vertex_keys(g), ["n0", "n1"])

    def test_index_fallback(self):
        g = ig.Graph(n=4)
        self.assertEqual(_vertex_keys(g), [0, 1, 2, 3])

    def test_empty_graph(self):
        self.assertEqual(_vertex_keys(ig.Graph(n=0)), [])

    def test_duplicate_names_raise(self):
        """Duplicate identity keys must raise, not silently merge nodes."""
        g = ig.Graph(n=3)
        g.vs["name"] = ["a", "b", "a"]
        with self.assertRaises(ValueError) as ctx:
            _vertex_keys(g)
        self.assertIn("'a'", str(ctx.exception))

    def test_duplicate_labels_raise(self):
        g = ig.Graph(n=2)
        g.vs["label"] = ["x", "x"]
        with self.assertRaises(ValueError):
            _vertex_keys(g)

    def test_unique_names_still_fine(self):
        g = ig.Graph(n=3)
        g.vs["name"] = ["a", "b", "c"]
        self.assertEqual(_vertex_keys(g), ["a", "b", "c"])


if __name__ == '__main__':
    unittest.main()
