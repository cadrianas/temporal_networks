import unittest

import igraph as ig
import numpy as np
import pandas as pd

from temporal_networks.change_points import (
    detect_change_points,
    flag_anomalous_snapshots,
)


def _simple_df(values, labels=None):
    """Build a minimal DataFrame like network_properties output."""
    if labels is None:
        labels = [str(i) for i in range(len(values))]
    return pd.DataFrame({"Graph": labels, "metric": values})


def _gap_info(segments):
    return {"segments": segments, "gaps": [], "has_gaps": False}


# ---------------------------------------------------------------------------
# detect_change_points — zscore
# ---------------------------------------------------------------------------

class TestZScore(unittest.TestCase):

    def test_spike_detected(self):
        # Central spike well outside 2-sigma
        values = [1.0] * 5 + [200.0] + [1.0] * 5
        df = _simple_df(values)
        cp = detect_change_points(df, threshold=2.0)
        self.assertFalse(cp.empty)
        self.assertIn(5, cp["index"].tolist())

    def test_flat_series_no_flags(self):
        df = _simple_df([5.0] * 10)
        cp = detect_change_points(df, threshold=3.0)
        self.assertTrue(cp.empty)

    def test_output_columns(self):
        values = [1.0] * 5 + [200.0] + [1.0] * 5
        df = _simple_df(values)
        cp = detect_change_points(df, threshold=2.0)
        self.assertEqual(list(cp.columns),
                         ["column", "index", "label", "score", "method"])

    def test_method_recorded(self):
        values = [1.0] * 5 + [200.0] + [1.0] * 5
        df = _simple_df(values)
        cp = detect_change_points(df, threshold=2.0, method="zscore")
        self.assertTrue((cp["method"] == "zscore").all())

    def test_correct_label_in_output(self):
        labels = [f"2024-{i+1:02d}" for i in range(11)]
        values = [1.0] * 5 + [200.0] + [1.0] * 5
        df = _simple_df(values, labels)
        cp = detect_change_points(df, threshold=2.0)
        spike_row = cp[cp["index"] == 5]
        self.assertEqual(spike_row["label"].iloc[0], "2024-06")

    def test_columns_none_uses_all_numeric(self):
        df = pd.DataFrame({
            "Graph": list("abcdefghijk"),
            "m1": [1.0] * 5 + [200.0] + [1.0] * 5,
            "m2": [2.0] * 11,
        })
        cp = detect_change_points(df, threshold=2.0)
        # m1 should flag, m2 should not
        self.assertIn("m1", cp["column"].tolist())
        self.assertNotIn("m2", cp["column"].tolist())

    def test_explicit_columns_subset(self):
        df = pd.DataFrame({
            "Graph": list("abcdefghijk"),
            "m1": [1.0] * 5 + [200.0] + [1.0] * 5,
            "m2": [1.0] * 5 + [200.0] + [1.0] * 5,
        })
        # Only analyse m2 — result should only mention m2
        cp = detect_change_points(df, columns=["m2"], threshold=2.0)
        self.assertNotIn("m1", cp["column"].tolist())
        self.assertIn("m2", cp["column"].tolist())


# ---------------------------------------------------------------------------
# detect_change_points — diff/MAD
# ---------------------------------------------------------------------------

class TestDiffMAD(unittest.TestCase):

    def test_step_change_detected(self):
        # Clean step change: 0→0→0→0→0→10→10→10→10→10
        values = [0.0] * 5 + [10.0] * 5
        df = _simple_df(values)
        cp = detect_change_points(df, method="diff", threshold=2.0)
        self.assertFalse(cp.empty)
        self.assertEqual(cp["method"].iloc[0], "diff")

    def test_flat_series_no_flags(self):
        df = _simple_df([7.0] * 10)
        cp = detect_change_points(df, method="diff", threshold=3.0)
        self.assertTrue(cp.empty)

    def test_constant_differences_no_flags(self):
        # Linearly increasing — all diffs equal, MAD = 0 → no flags
        df = _simple_df(list(range(10)))
        cp = detect_change_points(df, method="diff", threshold=3.0)
        self.assertTrue(cp.empty)


# ---------------------------------------------------------------------------
# Gap-awareness
# ---------------------------------------------------------------------------

class TestGapAwareness(unittest.TestCase):

    def test_jump_at_gap_not_flagged_zscore(self):
        # Values jump from 1 to 100 at index 5 — coincides with a gap.
        # With gap_info supplied, each segment is scored independently
        # so the boundary itself is not computed, and the step is split
        # across two separate z-score windows.
        values = [1.0] * 5 + [100.0] * 5
        labels = [f"2024-{i+1:02d}" for i in range(5)] + \
                 [f"2025-{i+1:02d}" for i in range(5)]
        df = _simple_df(values, labels)
        # Segment 0: [1,1,1,1,1] — constant, std=0 → no flags
        # Segment 1: [100,100,100,100,100] — constant → no flags
        gi = _gap_info([(0, 5), (5, 10)])
        cp = detect_change_points(df, threshold=2.0, gap_info=gi)
        self.assertTrue(cp.empty)

    def test_jump_at_gap_not_flagged_diff(self):
        values = [1.0] * 5 + [100.0] * 5
        labels = [str(i) for i in range(10)]
        df = _simple_df(values, labels)
        # Each segment: diffs are all 0 (constant) → MAD=0 → no flags
        gi = _gap_info([(0, 5), (5, 10)])
        cp = detect_change_points(df, method="diff", threshold=2.0,
                                   gap_info=gi)
        self.assertTrue(cp.empty)

    def test_jump_without_gap_info_is_flagged(self):
        # Same data, but no gap_info — the jump should be detected
        values = [0.0] * 5 + [100.0] * 5
        df = _simple_df(values)
        # Without gap_info zscore sees the global distribution
        cp_z = detect_change_points(df, method="zscore", threshold=1.5)
        cp_d = detect_change_points(df, method="diff", threshold=1.5)
        # At least one method should detect the jump
        self.assertTrue(not cp_z.empty or not cp_d.empty)


# ---------------------------------------------------------------------------
# PELT (optional dependency)
# ---------------------------------------------------------------------------

class TestPelt(unittest.TestCase):

    def test_pelt_missing_raises_informative_error(self):
        import sys
        import unittest.mock as mock

        df = _simple_df([1.0] * 10)
        # Simulate ruptures not being installed
        with mock.patch.dict(sys.modules, {"ruptures": None}):
            with self.assertRaises(ImportError) as ctx:
                detect_change_points(df, method="pelt")
        self.assertIn("ruptures", str(ctx.exception))
        self.assertIn("pip install", str(ctx.exception))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):

    def test_invalid_method_raises(self):
        df = _simple_df([1.0] * 5)
        with self.assertRaises(ValueError):
            detect_change_points(df, method="bogus")

    def test_empty_output_has_correct_columns(self):
        df = _simple_df([5.0] * 10)
        cp = detect_change_points(df, threshold=100.0)
        self.assertTrue(cp.empty)
        self.assertEqual(list(cp.columns),
                         ["column", "index", "label", "score", "method"])

    def test_missing_column_skipped_gracefully(self):
        df = _simple_df([1.0] * 5)
        # Ask for a column that does not exist — should produce empty result
        cp = detect_change_points(df, columns=["nonexistent"])
        self.assertTrue(cp.empty)


# ---------------------------------------------------------------------------
# flag_anomalous_snapshots
# ---------------------------------------------------------------------------

class TestFlagAnomalousSnapshots(unittest.TestCase):

    def test_detects_sparse_outlier_snapshot(self):
        g_dense = ig.Graph.Full(n=6)
        g_sparse = ig.Graph(n=6, edges=[(0, 1)])
        graphs = [g_dense] * 8 + [g_sparse] + [g_dense] * 8
        labels = [f"2024-{i+1:02d}" for i in range(17)]
        flags = flag_anomalous_snapshots(graphs, graph_labels=labels,
                                         threshold=2.0)
        self.assertFalse(flags.empty)

    def test_constant_sequence_no_flags(self):
        g = ig.Graph.Full(n=4)
        graphs = [g] * 10
        labels = [f"2024-{i+1:02d}" for i in range(10)]
        flags = flag_anomalous_snapshots(graphs, graph_labels=labels,
                                         threshold=3.0)
        self.assertTrue(flags.empty)

    def test_output_columns(self):
        g = ig.Graph.Full(n=4)
        graphs = [g] * 5
        labels = [f"2024-{i+1:02d}" for i in range(5)]
        flags = flag_anomalous_snapshots(graphs, graph_labels=labels)
        self.assertEqual(list(flags.columns),
                         ["column", "index", "label", "score", "method"])


if __name__ == "__main__":
    unittest.main()
