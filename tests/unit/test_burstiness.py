import contextlib
import io
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import igraph as ig
import pandas as pd

from temporal_networks.burstiness import (
    inter_event_times,
    burstiness_coefficient,
)


def _on():
    """An edge (0, 1) present."""
    return ig.Graph(n=2, edges=[(0, 1)])


def _off():
    """No edges."""
    return ig.Graph(n=2)


class TestInterEventTimes(unittest.TestCase):
    def test_rows_sorted_chronologically_not_lexicographically(self):
        """Default labels ("Graph 10" < "Graph 2" as strings) must come
        out in snapshot order."""
        # Edge active at snapshots 0, 1, 9, 10 -> three intervals.
        graphs = [_on(), _on()] + [_off()] * 7 + [_on(), _on()]
        df = inter_event_times(graphs)  # default "Graph N" labels
        self.assertEqual(list(df["start_label"]),
                         ["Graph 1", "Graph 2", "Graph 10"])

    def test_intervals_in_inferred_unit(self):
        graphs = [_on(), _off(), _on()]  # active at 0 and 2
        labels = ["2024-01", "2024-02", "2024-03"]
        df = inter_event_times(graphs, graph_labels=labels)
        self.assertEqual(list(df.columns),
                         ["entity", "start_label", "end_label",
                          "interval", "spans_gap"])
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["entity"], "(0, 1)")
        self.assertEqual(row["interval"], 2.0)  # Jan -> Mar = 2 months
        self.assertFalse(bool(row["spans_gap"]))

    def test_regular_every_snapshot(self):
        graphs = [_on(), _on(), _on()]
        labels = ["2024-01", "2024-02", "2024-03"]
        df = inter_event_times(graphs, graph_labels=labels)
        self.assertEqual(list(df["interval"]), [1.0, 1.0])

    def test_by_node(self):
        graphs = [_on(), _off(), _on()]
        labels = ["2024-01", "2024-02", "2024-03"]
        df = inter_event_times(graphs, graph_labels=labels, by="node")
        entities = set(df["entity"])
        self.assertEqual(entities, {"0", "1"})
        # both nodes active at 0 and 2 -> one interval each
        self.assertTrue(all(df["interval"] == 2.0))

    def test_empty_when_single_activation(self):
        # edge active only once -> no interval rows
        graphs = [_on(), _off(), _off()]
        labels = ["2024-01", "2024-02", "2024-03"]
        df = inter_event_times(graphs, graph_labels=labels)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns),
                         ["entity", "start_label", "end_label",
                          "interval", "spans_gap"])

    def test_invalid_by_raises(self):
        with self.assertRaises(ValueError):
            inter_event_times([_on(), _on()], graph_labels=["a", "b"],
                              by="bogus")

    def test_requires_two_graphs(self):
        with self.assertRaises(ValueError):
            inter_event_times([_on()], graph_labels=["a"])


class TestGapHandling(unittest.TestCase):
    def test_exclude_gaps_drops_spanning_interval(self):
        # active at 0 and 2, with 2024-02 missing -> gap between them
        graphs = [_on(), _on()]
        labels = ["2024-01", "2024-03"]
        df = inter_event_times(graphs, graph_labels=labels, exclude_gaps=True)
        self.assertTrue(df.empty)

    def test_keep_gaps_flags_spanning_interval(self):
        graphs = [_on(), _on()]
        labels = ["2024-01", "2024-03"]
        df = inter_event_times(graphs, graph_labels=labels, exclude_gaps=False)
        self.assertEqual(len(df), 1)
        self.assertTrue(bool(df.iloc[0]["spans_gap"]))
        self.assertEqual(df.iloc[0]["interval"], 2.0)


class TestBurstinessCoefficient(unittest.TestCase):
    def test_regular_pattern_is_minus_one(self):
        graphs = [_on(), _on(), _on(), _on()]
        labels = ["2024-01", "2024-02", "2024-03", "2024-04"]
        df = burstiness_coefficient(graphs, graph_labels=labels,
                                    report_gaps=False)
        self.assertEqual(list(df.columns),
                         ["entity", "n_events", "mean_interval",
                          "std_interval", "burstiness"])
        row = df.iloc[0]
        self.assertEqual(row["n_events"], 4)
        self.assertEqual(row["std_interval"], 0.0)
        self.assertEqual(row["burstiness"], -1.0)

    def test_clustered_pattern_is_positive(self):
        # active at 0,1,2 then a long quiet stretch then 12
        # intervals: 1, 1, 10 -> std > mean -> B > 0
        active = {0, 1, 2, 12}
        graphs = [_on() if i in active else _off() for i in range(13)]
        labels = [f"2024-{m:02d}" if m <= 12 else "x" for m in range(1, 14)]
        labels = [f"20{24 + (i // 12)}-{(i % 12) + 1:02d}" for i in range(13)]
        df = burstiness_coefficient(graphs, graph_labels=labels,
                                    report_gaps=False)
        row = df[df["entity"] == "(0, 1)"].iloc[0]
        self.assertEqual(row["n_events"], 4)
        self.assertGreater(row["burstiness"], 0.0)

    def test_single_activation_is_nan(self):
        graphs = [_on(), _off(), _off()]
        labels = ["2024-01", "2024-02", "2024-03"]
        df = burstiness_coefficient(graphs, graph_labels=labels,
                                    report_gaps=False)
        row = df[df["entity"] == "(0, 1)"].iloc[0]
        self.assertEqual(row["n_events"], 1)
        self.assertTrue(math.isnan(row["burstiness"]))
        self.assertTrue(math.isnan(row["mean_interval"]))

    def test_exclude_gaps_changes_statistics(self):
        # active at 0 and 2 only, with a gap (2024-02 missing).
        graphs = [_on(), _on()]
        labels = ["2024-01", "2024-03"]
        excl = burstiness_coefficient(graphs, graph_labels=labels,
                                      exclude_gaps=True, report_gaps=False)
        keep = burstiness_coefficient(graphs, graph_labels=labels,
                                      exclude_gaps=False, report_gaps=False)
        # excluding the only (gap-spanning) interval -> no usable interval
        self.assertTrue(math.isnan(excl.iloc[0]["burstiness"]))
        # keeping it -> one interval -> regular (-1)
        self.assertEqual(keep.iloc[0]["burstiness"], -1.0)

    def test_invalid_by_raises(self):
        with self.assertRaises(ValueError):
            burstiness_coefficient([_on(), _on()], graph_labels=["a", "b"],
                                   by="bogus")


class TestErrorHandling(unittest.TestCase):
    def test_compute_exception_warns_and_skips(self):
        graphs = [_on(), _on()]
        with patch("temporal_networks.burstiness._edge_identity_set",
                   side_effect=ig.InternalError("boom")):
            with self.assertWarns(UserWarning):
                df = burstiness_coefficient(graphs, graph_labels=["a", "b"],
                                            report_gaps=False)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns),
                         ["entity", "n_events", "mean_interval",
                          "std_interval", "burstiness"])


class TestExceptionPropagation(unittest.TestCase):
    def test_programming_errors_propagate(self):
        """A TypeError from a helper is a bug and must not become a warning."""
        graphs = [_on(), _on()]
        with patch("temporal_networks.burstiness._edge_identity_set",
                   side_effect=TypeError("a bug, not bad data")):
            with self.assertRaises(TypeError):
                burstiness_coefficient(graphs, graph_labels=["a", "b"],
                                       report_gaps=False)


class TestPlotting(unittest.TestCase):
    def test_save_path_writes_pdf(self):
        graphs = [_on(), _on(), _on()]
        labels = ["2024-01", "2024-02", "2024-03"]
        with tempfile.TemporaryDirectory() as tmp:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                burstiness_coefficient(graphs, graph_labels=labels,
                                       save_path=tmp, report_gaps=False)
            files = os.listdir(tmp)
        self.assertIn("burstiness_edge.pdf", files)


if __name__ == "__main__":
    unittest.main()
