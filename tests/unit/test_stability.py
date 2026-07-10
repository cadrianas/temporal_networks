import math
import os
import tempfile
import unittest
from unittest.mock import patch
import io
import contextlib

import igraph as ig
import numpy as np
import pandas as pd

from temporal_networks.stability import (
    snapshot_similarity,
    temporal_correlation_coefficient,
)


class TestSnapshotSimilarity(unittest.TestCase):
    def test_starts_at_second_snapshot(self):
        g = ig.Graph(n=4, edges=[(0, 1), (1, 2)])
        graphs = [g.copy(), g.copy(), g.copy()]
        df = snapshot_similarity(graphs, graph_labels=["a", "b", "c"],
                                 report_gaps=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["Graph"]), ["b", "c"])

    def test_identical_snapshots(self):
        g = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3)])
        df = snapshot_similarity([g.copy(), g.copy()],
                                 graph_labels=["2024-01", "2024-02"],
                                 report_gaps=False)
        row = df.iloc[0]
        self.assertEqual(row["jaccard"], 1.0)
        self.assertEqual(row["edge_persistence"], 1.0)
        self.assertEqual(row["node_persistence"], 1.0)
        self.assertEqual(row["temporal_correlation"], 1.0)

    def test_edge_disjoint_snapshots(self):
        g0 = ig.Graph(n=4, edges=[(0, 1)])
        g1 = ig.Graph(n=4, edges=[(2, 3)])
        df = snapshot_similarity([g0, g1],
                                 graph_labels=["2024-01", "2024-02"],
                                 report_gaps=False)
        row = df.iloc[0]
        self.assertEqual(row["jaccard"], 0.0)
        self.assertEqual(row["edge_persistence"], 0.0)
        self.assertEqual(row["node_persistence"], 0.0)
        self.assertEqual(row["temporal_correlation"], 0.0)

    def test_partial_overlap_exact_values(self):
        g0 = ig.Graph(n=4, edges=[(0, 1), (1, 2)])
        g1 = ig.Graph(n=4, edges=[(1, 2), (2, 3)])
        df = snapshot_similarity([g0, g1],
                                 graph_labels=["2024-01", "2024-02"],
                                 report_gaps=False)
        row = df.iloc[0]
        # intersection {(1,2)}, union {(0,1),(1,2),(2,3)} -> 1/3
        self.assertAlmostEqual(row["jaccard"], 1 / 3)
        self.assertEqual(row["edge_persistence"], 0.5)  # 1 of 2 survive

    def test_node_persistence(self):
        # node 0 active in both; node 1 drops, node 2 appears
        g0 = ig.Graph(n=3, edges=[(0, 1)])
        g1 = ig.Graph(n=3, edges=[(0, 2)])
        df = snapshot_similarity([g0, g1],
                                 graph_labels=["2024-01", "2024-02"],
                                 report_gaps=False)
        # active_prev = {0, 1}, intersection with {0, 2} = {0} -> 1/2
        self.assertEqual(df.iloc[0]["node_persistence"], 0.5)


class TestGapHandling(unittest.TestCase):
    def test_pair_across_gap_is_nan(self):
        g = ig.Graph(n=4, edges=[(0, 1), (1, 2)])
        # 2024-02 missing -> a gap between the two snapshots
        df = snapshot_similarity([g.copy(), g.copy()],
                                 graph_labels=["2024-01", "2024-03"],
                                 report_gaps=False)
        row = df.iloc[0]
        self.assertTrue(math.isnan(row["jaccard"]))
        self.assertTrue(math.isnan(row["edge_persistence"]))
        self.assertTrue(math.isnan(row["temporal_correlation"]))


class TestTemporalCorrelationCoefficient(unittest.TestCase):
    def test_identical_sequence_is_one(self):
        g = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3)])
        graphs = [g.copy() for _ in range(3)]
        c = temporal_correlation_coefficient(
            graphs, graph_labels=["2024-01", "2024-02", "2024-03"])
        self.assertEqual(c, 1.0)

    def test_disjoint_sequence_is_zero(self):
        g0 = ig.Graph(n=4, edges=[(0, 1)])
        g1 = ig.Graph(n=4, edges=[(2, 3)])
        c = temporal_correlation_coefficient(
            [g0, g1], graph_labels=["2024-01", "2024-02"])
        self.assertEqual(c, 0.0)

    def test_all_gap_pairs_returns_nan(self):
        g = ig.Graph(n=4, edges=[(0, 1)])
        # single consecutive pair that straddles a gap -> no valid pair
        c = temporal_correlation_coefficient(
            [g.copy(), g.copy()], graph_labels=["2024-01", "2024-05"])
        self.assertTrue(math.isnan(c))


class TestErrorHandling(unittest.TestCase):
    def test_compute_exception_warns_and_emits_nan_row(self):
        """A failing pair warns and yields a NaN row, preserving shape."""
        g = ig.Graph(n=4, edges=[(0, 1)])
        graphs = [g.copy(), g.copy()]
        with patch("temporal_networks.stability._edge_identity_set",
                   side_effect=Exception("boom")):
            with self.assertWarns(UserWarning):
                df = snapshot_similarity(graphs, graph_labels=["a", "b"],
                                         report_gaps=False)
        # One row per consecutive pair, even on failure.
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "Graph"], "b")
        for metric in ["jaccard", "edge_persistence",
                       "node_persistence", "temporal_correlation"]:
            self.assertTrue(math.isnan(df.loc[0, metric]))
        self.assertEqual(list(df.columns),
                         ["Graph", "jaccard", "edge_persistence",
                          "node_persistence", "temporal_correlation"])

    def test_requires_two_graphs(self):
        with self.assertRaises(ValueError):
            snapshot_similarity([ig.Graph(n=2)], graph_labels=["a"])


class TestPlotting(unittest.TestCase):
    def test_save_path_writes_pdfs(self):
        g0 = ig.Graph(n=4, edges=[(0, 1), (1, 2)])
        g1 = ig.Graph(n=4, edges=[(1, 2), (2, 3)])
        with tempfile.TemporaryDirectory() as tmp:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                snapshot_similarity([g0, g1],
                                    graph_labels=["2024-01", "2024-02"],
                                    save_path=tmp, report_gaps=False)
            files = os.listdir(tmp)
        for metric in ("jaccard", "edge_persistence", "node_persistence",
                       "temporal_correlation"):
            self.assertIn(f"{metric}.pdf", files)


if __name__ == "__main__":
    unittest.main()
