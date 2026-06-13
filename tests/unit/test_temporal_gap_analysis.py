import unittest

import temporal_networks.temporal_gap_analysis as temporal_gap_analysis
import temporal_networks._gap_utilities as _gap_utilities
from temporal_networks.temporal_gap_analysis import (
    detect_temporal_gaps,
    create_gap_dataframe,
)


class TestTemporalGapAnalysis(unittest.TestCase):
    def test_all_exports(self):
        """Test that __all__ contains exactly the expected functions."""
        expected_exports = [
            "parse_flexible_datetime",
            "calculate_time_difference",
            "detect_temporal_gaps",
            "print_gap_report",
            "create_gap_dataframe",
        ]
        self.assertCountEqual(temporal_gap_analysis.__all__, expected_exports)
        self.assertEqual(temporal_gap_analysis.__all__, expected_exports)

    def test_exported_functions(self):
        """Test that the functions exported match the original implementations in _gap_utilities."""
        self.assertIs(
            temporal_gap_analysis.parse_flexible_datetime,
            _gap_utilities.parse_flexible_datetime
        )
        self.assertIs(
            temporal_gap_analysis.calculate_time_difference,
            _gap_utilities.calculate_time_difference
        )
        self.assertIs(
            temporal_gap_analysis.detect_temporal_gaps,
            _gap_utilities.detect_temporal_gaps
        )
        self.assertIs(
            temporal_gap_analysis.print_gap_report,
            _gap_utilities.print_gap_report
        )
        self.assertIs(
            temporal_gap_analysis.create_gap_dataframe,
            _gap_utilities.create_gap_dataframe
        )


class TestCreateGapDataframe(unittest.TestCase):
    """Behavioural coverage for create_gap_dataframe (previously only its
    re-export identity was tested)."""

    def test_gapped_labels_produce_one_row_per_gap(self):
        labels = ["2024-01", "2024-02", "2024-09", "2024-10"]
        gap_info = detect_temporal_gaps(labels)
        gdf = create_gap_dataframe(labels, gap_info)
        self.assertEqual(len(gdf), 1)
        row = gdf.iloc[0]
        self.assertEqual(row["start_label"], "2024-02")
        self.assertEqual(row["end_label"], "2024-09")
        self.assertEqual(row["gap_size"], 7.0)
        self.assertEqual(row["gap_number"], 1)
        for col in ["gap_number", "start_label", "end_label", "gap_size",
                    "start_idx", "end_idx"]:
            self.assertIn(col, gdf.columns)

    def test_continuous_labels_produce_empty_frame(self):
        labels = ["2024-01", "2024-02", "2024-03"]
        gdf = create_gap_dataframe(labels, detect_temporal_gaps(labels))
        self.assertEqual(len(gdf), 0)
        self.assertEqual(
            list(gdf.columns),
            ["gap_number", "start_label", "end_label", "gap_size"])


if __name__ == '__main__':
    unittest.main()
