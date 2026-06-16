import os
import tempfile
import unittest

import pandas as pd

from temporal_networks.io import (
    snapshots_from_events,
    snapshots_from_edgelist,
)
from temporal_networks._gap_utilities import detect_temporal_gaps


def _events():
    """Three monthly bins (Jan, Feb, Mar) with known edges."""
    return pd.DataFrame({
        "t": ["2024-01-05", "2024-01-20", "2024-02-10", "2024-03-02"],
        "u": ["a", "b", "a", "a"],
        "v": ["b", "c", "c", "c"],
    })


class TestBasicBinning(unittest.TestCase):
    def test_snapshot_count_labels_and_edges(self):
        graphs, labels = snapshots_from_events(
            _events(), "t", "u", "v", freq="M")
        self.assertEqual(labels, ["2024-01", "2024-02", "2024-03"])
        self.assertEqual([g.ecount() for g in graphs], [2, 1, 1])

    def test_union_vertex_set_is_shared(self):
        graphs, _ = snapshots_from_events(_events(), "t", "u", "v", freq="M")
        # Union of all nodes is {a, b, c}; every snapshot carries all three.
        for g in graphs:
            self.assertEqual(sorted(g.vs["name"]), ["a", "b", "c"])
            self.assertEqual(g.vcount(), 3)

    def test_output_feeds_gap_detector(self):
        _, labels = snapshots_from_events(_events(), "t", "u", "v", freq="M")
        self.assertFalse(detect_temporal_gaps(labels)["has_gaps"])


class TestGaps(unittest.TestCase):
    def test_missing_month_becomes_a_gap(self):
        df = pd.DataFrame({
            "t": ["2024-01-05", "2024-03-02"],   # February missing
            "u": ["a", "a"],
            "v": ["b", "c"],
        })
        graphs, labels = snapshots_from_events(df, "t", "u", "v", freq="M")
        self.assertEqual(labels, ["2024-01", "2024-03"])
        self.assertEqual(len(graphs), 2)
        info = detect_temporal_gaps(labels)
        self.assertTrue(info["has_gaps"])
        self.assertEqual(info["num_gaps"], 1)


class TestWeights(unittest.TestCase):
    def test_weight_col_is_summed(self):
        df = pd.DataFrame({
            "t": ["2024-01-01", "2024-01-15", "2024-01-20"],
            "u": ["a", "a", "b"],
            "v": ["b", "b", "c"],          # (a,b) appears twice
            "w": [2.0, 3.0, 1.0],
        })
        graphs, _ = snapshots_from_events(
            df, "t", "u", "v", freq="M", weight_col="w")
        g = graphs[0]
        self.assertTrue(g.is_weighted())
        weights = {
            tuple(sorted((g.vs[e.source]["name"], g.vs[e.target]["name"]))): e["weight"]
            for e in g.es
        }
        self.assertEqual(weights[("a", "b")], 5.0)
        self.assertEqual(weights[("b", "c")], 1.0)

    def test_parallel_events_collapse_to_count(self):
        df = pd.DataFrame({
            "t": ["2024-01-01", "2024-01-15"],
            "u": ["a", "a"],
            "v": ["b", "b"],               # same edge twice, no weight col
        })
        graphs, _ = snapshots_from_events(df, "t", "u", "v", freq="M")
        g = graphs[0]
        self.assertEqual(g.ecount(), 1)            # collapsed
        self.assertEqual(g.es["weight"], [2.0])    # multiplicity

    def test_simple_data_has_no_weight(self):
        graphs, _ = snapshots_from_events(_events(), "t", "u", "v", freq="M")
        self.assertNotIn("weight", graphs[0].es.attributes())


class TestDirected(unittest.TestCase):
    def test_directed_keeps_orientation(self):
        df = pd.DataFrame({
            "t": ["2024-01-01", "2024-01-02"],
            "u": ["a", "b"],
            "v": ["b", "a"],
        })
        graphs, _ = snapshots_from_events(
            df, "t", "u", "v", freq="M", directed=True)
        g = graphs[0]
        self.assertTrue(g.is_directed())
        self.assertEqual(g.ecount(), 2)   # (a->b) and (b->a) distinct

    def test_undirected_merges_orientation(self):
        df = pd.DataFrame({
            "t": ["2024-01-01", "2024-01-02"],
            "u": ["a", "b"],
            "v": ["b", "a"],
        })
        graphs, _ = snapshots_from_events(df, "t", "u", "v", freq="M")
        self.assertEqual(graphs[0].ecount(), 1)   # (a,b)==(b,a)


class TestLabelFormats(unittest.TestCase):
    def test_daily_labels(self):
        df = pd.DataFrame({
            "t": ["2024-03-01", "2024-03-02"],
            "u": ["a", "a"],
            "v": ["b", "c"],
        })
        _, labels = snapshots_from_events(df, "t", "u", "v", freq="D")
        self.assertEqual(labels, ["2024-03-01", "2024-03-02"])

    def test_quarterly_labels(self):
        df = pd.DataFrame({
            "t": ["2024-02-01", "2024-05-01"],
            "u": ["a", "a"],
            "v": ["b", "c"],
        })
        _, labels = snapshots_from_events(df, "t", "u", "v", freq="Q")
        self.assertEqual(labels, ["2024-Q1", "2024-Q2"])

    def test_coarse_format_duplicate_labels_raises(self):
        # Daily bins but a monthly label_format -> duplicate labels.
        df = pd.DataFrame({
            "t": ["2024-03-01", "2024-03-02"],
            "u": ["a", "a"],
            "v": ["b", "c"],
        })
        with self.assertRaises(ValueError):
            snapshots_from_events(
                df, "t", "u", "v", freq="D", label_format="%Y-%m")

    def test_unparseable_label_format_raises(self):
        with self.assertRaises(ValueError):
            snapshots_from_events(
                _events(), "t", "u", "v", freq="M", label_format="%d/%m/%Y")


class TestErrors(unittest.TestCase):
    def test_missing_column(self):
        with self.assertRaises(ValueError):
            snapshots_from_events(_events(), "t", "u", "MISSING", freq="M")

    def test_empty_df(self):
        empty = pd.DataFrame({"t": [], "u": [], "v": []})
        with self.assertRaises(ValueError):
            snapshots_from_events(empty, "t", "u", "v", freq="M")

    def test_unparseable_time_column(self):
        df = pd.DataFrame({
            "t": ["not-a-date", "also-bad"],
            "u": ["a", "b"],
            "v": ["b", "c"],
        })
        with self.assertRaises(ValueError):
            snapshots_from_events(df, "t", "u", "v", freq="M")

    def test_non_numeric_weight_column(self):
        df = pd.DataFrame({
            "t": ["2024-01-01"],
            "u": ["a"],
            "v": ["b"],
            "w": ["heavy"],
        })
        with self.assertRaises(ValueError):
            snapshots_from_events(
                df, "t", "u", "v", freq="M", weight_col="w")


class TestFromEdgelist(unittest.TestCase):
    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.csv")
            _events().to_csv(path, index=False)
            graphs, labels = snapshots_from_edgelist(
                path, "t", "u", "v", freq="M")
        self.assertEqual(labels, ["2024-01", "2024-02", "2024-03"])
        self.assertEqual([g.ecount() for g in graphs], [2, 1, 1])


if __name__ == "__main__":
    unittest.main()
