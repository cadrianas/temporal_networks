import contextlib
import io
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import igraph as ig
import pandas as pd

from temporal_networks.temporal_paths import (
    temporal_reachability,
    temporal_distances,
    temporal_closeness,
    temporal_efficiency,
    temporal_betweenness,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _chain():
    """Three-snapshot chain: 0-1 at t0, 1-2 at t1. Labels a/b."""
    g0 = ig.Graph(n=3, edges=[(0, 1)])
    g1 = ig.Graph(n=3, edges=[(1, 2)])
    return [g0, g1], ["a", "b"]


def _wait_test():
    """Two snapshots with source-only edges at different steps.

    g0: 0-1    g1: 0-2
    allow_wait=True  → 0 reaches both 1 and 2
    allow_wait=False → 0 only reaches 1 (can't wait to use g1 edge)
    """
    g0 = ig.Graph(n=3, edges=[(0, 1)])
    g1 = ig.Graph(n=3, edges=[(0, 2)])
    return [g0, g1], ["a", "b"]


def _gap_chain():
    """Two snapshots with a gap: A-B at 2024-01, B-C at 2024-03."""
    g0 = ig.Graph(n=3)
    g0.vs["name"] = ["A", "B", "C"]
    g0.add_edge(0, 1)
    g1 = ig.Graph(n=3)
    g1.vs["name"] = ["A", "B", "C"]
    g1.add_edge(1, 2)
    return [g0, g1], ["2024-01", "2024-03"]


def _reach(df, src, tgt):
    row = df[(df["source"] == src) & (df["target"] == tgt)]
    return bool(row["reachable"].iloc[0])


def _latency(df, src, tgt):
    row = df[(df["source"] == src) & (df["target"] == tgt)]
    return float(row["latency"].iloc[0])


# ---------------------------------------------------------------------------
# temporal_reachability
# ---------------------------------------------------------------------------

class TestTemporalReachability(unittest.TestCase):

    def test_chain_latencies(self):
        """0→1 at step 1, 0→2 at step 2 — hand-verifiable."""
        graphs, labels = _chain()
        df = temporal_reachability(graphs, graph_labels=labels)
        self.assertTrue(_reach(df, 0, 1))
        self.assertEqual(
            df[(df["source"] == 0) & (df["target"] == 1)]
            ["first_arrival_idx"].iloc[0], 1.0)
        self.assertTrue(_reach(df, 0, 2))
        self.assertEqual(
            df[(df["source"] == 0) & (df["target"] == 2)]
            ["first_arrival_idx"].iloc[0], 2.0)

    def test_self_pairs_reachable_at_zero(self):
        graphs, labels = _chain()
        df = temporal_reachability(graphs, graph_labels=labels)
        for node in [0, 1, 2]:
            row = df[(df["source"] == node) & (df["target"] == node)]
            self.assertTrue(bool(row["reachable"].iloc[0]))
            self.assertEqual(row["first_arrival_idx"].iloc[0], 0.0)

    def test_disconnected_node_unreachable(self):
        g0 = ig.Graph(n=3, edges=[(0, 1)])  # node 2 isolated
        df = temporal_reachability([g0], graph_labels=["t"])
        self.assertFalse(_reach(df, 0, 2))
        self.assertTrue(math.isnan(
            df[(df["source"] == 0) & (df["target"] == 2)]
            ["first_arrival_idx"].iloc[0]))

    def test_output_columns(self):
        graphs, labels = _chain()
        df = temporal_reachability(graphs, graph_labels=labels)
        self.assertEqual(list(df.columns),
                         ["source", "target", "reachable",
                          "first_arrival_idx"])

    def test_undirected_both_directions(self):
        """Undirected edge: 1 can also reach 0 at step 1."""
        graphs, labels = _chain()
        df = temporal_reachability(graphs, graph_labels=labels)
        self.assertTrue(_reach(df, 1, 0))
        self.assertEqual(
            df[(df["source"] == 1) & (df["target"] == 0)]
            ["first_arrival_idx"].iloc[0], 1.0)


# ---------------------------------------------------------------------------
# allow_wait toggle
# ---------------------------------------------------------------------------

class TestAllowWait(unittest.TestCase):

    def test_wait_true_reaches_both(self):
        graphs, labels = _wait_test()
        df = temporal_reachability(graphs, graph_labels=labels,
                                   allow_wait=True)
        # source=0 can wait and use g1's edge to reach 2
        self.assertTrue(_reach(df, 0, 2))

    def test_wait_false_cannot_wait(self):
        graphs, labels = _wait_test()
        df = temporal_reachability(graphs, graph_labels=labels,
                                   allow_wait=False)
        # source=0 departs at t=0, can't use g1's edge (arrival != t=1)
        self.assertFalse(_reach(df, 0, 2))

    def test_wait_false_still_reaches_neighbor(self):
        graphs, labels = _wait_test()
        df = temporal_reachability(graphs, graph_labels=labels,
                                   allow_wait=False)
        self.assertTrue(_reach(df, 0, 1))


# ---------------------------------------------------------------------------
# One-hop-per-snapshot semantics (edge-order independence)
# ---------------------------------------------------------------------------

class TestOneHopPerSnapshot(unittest.TestCase):
    """Edges within a snapshot are simultaneous contacts.

    A path may traverse at most one edge per snapshot, and the result must
    not depend on the order edges were added to the graph (regression test:
    the single-sweep BFS used to chain through same-snapshot edges only
    when they happened to be stored in path order).
    """

    def test_no_multi_hop_within_single_snapshot(self):
        """Path 0-1-2 in ONE snapshot: 0 cannot reach 2 (one hop max)."""
        g = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
        df = temporal_reachability([g], graph_labels=["2024-01"])
        self.assertTrue(_reach(df, 0, 1))
        self.assertFalse(_reach(df, 0, 2))

    def test_reachability_independent_of_edge_order(self):
        g_fwd = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
        g_rev = ig.Graph(n=3, edges=[(1, 2), (0, 1)])
        df_fwd = temporal_reachability([g_fwd], graph_labels=["2024-01"])
        df_rev = temporal_reachability([g_rev], graph_labels=["2024-01"])
        pd.testing.assert_frame_equal(df_fwd, df_rev)

    def test_second_snapshot_completes_the_hop(self):
        """The same edge in the NEXT snapshot allows the second hop."""
        g = ig.Graph(n=3, edges=[(1, 2), (0, 1)])  # unfavourable order
        df = temporal_reachability([g, g.copy()],
                                   graph_labels=["2024-01", "2024-02"])
        self.assertTrue(_reach(df, 0, 2))
        self.assertEqual(
            df[(df["source"] == 0) & (df["target"] == 2)]
            ["first_arrival_idx"].iloc[0], 2.0)

    def test_betweenness_independent_of_edge_order(self):
        g_fwd = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
        g_rev = ig.Graph(n=3, edges=[(1, 2), (0, 1)])
        labels = ["2024-01", "2024-02"]
        bt_fwd = temporal_betweenness([g_fwd, g_fwd.copy()],
                                      graph_labels=labels, report_gaps=False)
        bt_rev = temporal_betweenness([g_rev, g_rev.copy()],
                                      graph_labels=labels, report_gaps=False)
        pd.testing.assert_frame_equal(bt_fwd, bt_rev)
        # Node 1 brokers the 0->2 and 2->0 foremost paths.
        self.assertEqual(
            float(bt_fwd[bt_fwd["node"] == 1]["betweenness"].iloc[0]), 1.0)


# ---------------------------------------------------------------------------
# Gap-blocking
# ---------------------------------------------------------------------------

class TestGapBlocking(unittest.TestCase):

    def test_cross_gaps_false_blocks_at_gap(self):
        """A→B reachable (within segment), A→C blocked (crosses gap)."""
        graphs, labels = _gap_chain()
        df = temporal_reachability(graphs, graph_labels=labels,
                                   cross_gaps=False)
        self.assertTrue(_reach(df, "A", "B"))
        self.assertFalse(_reach(df, "A", "C"))

    def test_post_gap_segment_paths_still_valid(self):
        """Paths confined to a post-gap segment are NOT blocked.

        Regression test: the BFS frontier used to die at the gap boundary
        (source included), so nothing was ever reachable after the first
        gap. A-B pre-gap; A-C then C-D post-gap: A must reach C and D via
        within-segment paths, while B->C (which would need waiting across
        the gap) stays blocked.
        """
        def g(edges):
            gr = ig.Graph(n=4)
            gr.vs["name"] = ["A", "B", "C", "D"]
            gr.add_edges(edges)
            return gr

        graphs = [g([(0, 1)]), g([(0, 2)]), g([(2, 3)])]
        labels = ["2024-01", "2024-03", "2024-04"]  # gap after 2024-01
        df = temporal_reachability(graphs, graph_labels=labels,
                                   cross_gaps=False)
        self.assertTrue(_reach(df, "A", "B"))    # pre-gap segment
        self.assertTrue(_reach(df, "A", "C"))    # post-gap segment
        self.assertTrue(_reach(df, "A", "D"))    # two hops, post-gap only
        self.assertEqual(
            df[(df["source"] == "A") & (df["target"] == "D")]
            ["first_arrival_idx"].iloc[0], 3.0)
        # B's only edge is pre-gap: reaching C would cross the gap.
        self.assertFalse(_reach(df, "B", "C"))

    def test_cross_gaps_true_allows_path_through_gap(self):
        graphs, labels = _gap_chain()
        df = temporal_reachability(graphs, graph_labels=labels,
                                   cross_gaps=True)
        self.assertTrue(_reach(df, "A", "B"))
        self.assertTrue(_reach(df, "A", "C"))
        self.assertEqual(
            df[(df["source"] == "A") & (df["target"] == "C")]
            ["first_arrival_idx"].iloc[0], 2.0)


# ---------------------------------------------------------------------------
# temporal_distances
# ---------------------------------------------------------------------------

class TestTemporalDistances(unittest.TestCase):

    def test_chain_distances(self):
        graphs, labels = _chain()
        dist = temporal_distances(graphs, graph_labels=labels)
        self.assertEqual(_latency(dist, 0, 1), 1.0)
        self.assertEqual(_latency(dist, 0, 2), 2.0)

    def test_unreachable_is_inf(self):
        g0 = ig.Graph(n=3, edges=[(0, 1)])
        dist = temporal_distances([g0], graph_labels=["t"])
        self.assertEqual(_latency(dist, 0, 2), float("inf"))

    def test_self_latency_is_zero(self):
        graphs, labels = _chain()
        dist = temporal_distances(graphs, graph_labels=labels)
        self.assertEqual(_latency(dist, 0, 0), 0.0)

    def test_output_columns(self):
        graphs, labels = _chain()
        dist = temporal_distances(graphs, graph_labels=labels)
        self.assertEqual(list(dist.columns), ["source", "target", "latency"])

    def test_gap_blocked_is_inf(self):
        graphs, labels = _gap_chain()
        dist = temporal_distances(graphs, graph_labels=labels,
                                   cross_gaps=False)
        self.assertEqual(_latency(dist, "A", "C"), float("inf"))

    def test_gap_allowed_has_finite_latency(self):
        graphs, labels = _gap_chain()
        dist = temporal_distances(graphs, graph_labels=labels,
                                   cross_gaps=True)
        self.assertEqual(_latency(dist, "A", "C"), 2.0)


# ---------------------------------------------------------------------------
# temporal_closeness
# ---------------------------------------------------------------------------

class TestTemporalCloseness(unittest.TestCase):

    def test_returns_one_row_per_node(self):
        graphs, labels = _chain()
        cl = temporal_closeness(graphs, graph_labels=labels, report_gaps=False)
        self.assertEqual(set(cl["node"]), {0, 1, 2})

    def test_output_columns(self):
        graphs, labels = _chain()
        cl = temporal_closeness(graphs, graph_labels=labels, report_gaps=False)
        self.assertEqual(list(cl.columns), ["node", "closeness"])

    def test_isolated_node_has_zero_closeness(self):
        g0 = ig.Graph(n=3, edges=[(0, 1)])  # node 2 isolated
        cl = temporal_closeness([g0], graph_labels=["t"], report_gaps=False)
        self.assertEqual(float(cl[cl["node"] == 2]["closeness"].iloc[0]), 0.0)

    def test_sorted_descending(self):
        graphs, labels = _chain()
        cl = temporal_closeness(graphs, graph_labels=labels, report_gaps=False)
        vals = list(cl["closeness"])
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_save_path_writes_pdf(self):
        graphs, labels = _chain()
        with tempfile.TemporaryDirectory() as tmp:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                temporal_closeness(graphs, graph_labels=labels,
                                    save_path=tmp, report_gaps=False)
            files = os.listdir(tmp)
        self.assertIn("temporal_closeness.pdf", files)


# ---------------------------------------------------------------------------
# temporal_efficiency
# ---------------------------------------------------------------------------

class TestTemporalEfficiency(unittest.TestCase):

    def test_single_edge_positive_efficiency(self):
        g0 = ig.Graph(n=2, edges=[(0, 1)])
        eff = temporal_efficiency([g0], graph_labels=["t"])
        self.assertGreater(eff, 0.0)
        self.assertLessEqual(eff, 1.0)

    def test_fully_connected_one_step_is_one(self):
        """All pairs reachable in 1 step → efficiency = 1.0."""
        g0 = ig.Graph.Full(n=3)
        eff = temporal_efficiency([g0], graph_labels=["t"])
        self.assertAlmostEqual(eff, 1.0)

    def test_disconnected_graph_is_partial(self):
        g0 = ig.Graph(n=4, edges=[(0, 1)])  # nodes 2,3 isolated
        eff = temporal_efficiency([g0], graph_labels=["t"])
        self.assertGreater(eff, 0.0)
        self.assertLess(eff, 1.0)

    def test_gap_reduces_efficiency(self):
        graphs, labels = _gap_chain()
        eff_blocked = temporal_efficiency(graphs, graph_labels=labels,
                                          cross_gaps=False)
        eff_allowed = temporal_efficiency(graphs, graph_labels=labels,
                                          cross_gaps=True)
        self.assertLessEqual(eff_blocked, eff_allowed)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling(unittest.TestCase):

    def test_bfs_exception_warns_and_skips(self):
        g0 = ig.Graph(n=2, edges=[(0, 1)])
        with patch("temporal_networks.temporal_paths._bfs_from_source",
                   side_effect=ig.InternalError("boom")):
            with self.assertWarns(UserWarning):
                df = temporal_reachability([g0], graph_labels=["t"])
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns),
                         ["source", "target", "reachable",
                          "first_arrival_idx"])

    def test_union_nodes_exception_returns_empty(self):
        g0 = ig.Graph(n=2, edges=[(0, 1)])
        with patch("temporal_networks.temporal_paths._union_nodes",
                   side_effect=ig.InternalError("no nodes")):
            with self.assertWarns(UserWarning):
                df = temporal_reachability([g0], graph_labels=["t"])
        self.assertTrue(df.empty)


# ---------------------------------------------------------------------------
# temporal_betweenness
# ---------------------------------------------------------------------------

def _betw(df, node):
    return float(df[df["node"] == node]["betweenness"].iloc[0])


class TestTemporalBetweenness(unittest.TestCase):

    def test_chain_broker_value(self):
        """0→1→2: node 1 is the sole broker.

        From source 0 the only brokered pair is (0, 2) routed through 1,
        contributing dependency 1. Normalised by (n-1)(n-2) = 2 → 0.5.
        Nodes 0 and 2 broker nothing.
        """
        graphs, labels = _chain()
        bt = temporal_betweenness(graphs, graph_labels=labels,
                                  report_gaps=False)
        self.assertEqual(_betw(bt, 1), 0.5)
        self.assertEqual(_betw(bt, 0), 0.0)
        self.assertEqual(_betw(bt, 2), 0.0)

    def test_raw_unnormalized(self):
        graphs, labels = _chain()
        bt = temporal_betweenness(graphs, graph_labels=labels,
                                  normalized=False, report_gaps=False)
        # One brokered ordered pair (0→2) through node 1 → raw dependency 1
        self.assertEqual(_betw(bt, 1), 1.0)

    def test_output_columns(self):
        graphs, labels = _chain()
        bt = temporal_betweenness(graphs, graph_labels=labels,
                                  report_gaps=False)
        self.assertEqual(list(bt.columns), ["node", "betweenness"])

    def test_sorted_descending(self):
        graphs, labels = _chain()
        bt = temporal_betweenness(graphs, graph_labels=labels,
                                  report_gaps=False)
        vals = list(bt["betweenness"])
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_two_nodes_all_zero(self):
        """With n < 3 no intermediary is possible — all zeros."""
        g0 = ig.Graph(n=2, edges=[(0, 1)])
        bt = temporal_betweenness([g0], graph_labels=["t"], report_gaps=False)
        self.assertTrue((bt["betweenness"] == 0.0).all())

    def test_broker_highest(self):
        """Longer chain 0→1→2→3: nodes 1 and 2 broker, ends do not."""
        g0 = ig.Graph(n=4, edges=[(0, 1)])
        g1 = ig.Graph(n=4, edges=[(1, 2)])
        g2 = ig.Graph(n=4, edges=[(2, 3)])
        bt = temporal_betweenness([g0, g1, g2],
                                  graph_labels=["a", "b", "c"],
                                  report_gaps=False)
        self.assertGreater(_betw(bt, 1), 0.0)
        self.assertGreater(_betw(bt, 2), 0.0)
        self.assertEqual(_betw(bt, 0), 0.0)
        self.assertEqual(_betw(bt, 3), 0.0)

    def test_gap_blocks_brokerage(self):
        """A broker that only works across a gap scores 0 when blocked."""
        graphs, labels = _gap_chain()  # A-B at 2024-01, B-C at 2024-03
        blocked = temporal_betweenness(graphs, graph_labels=labels,
                                       cross_gaps=False, report_gaps=False)
        allowed = temporal_betweenness(graphs, graph_labels=labels,
                                       cross_gaps=True, report_gaps=False)
        # Blocked: A cannot reach C, so B brokers nothing
        self.assertEqual(_betw(blocked, "B"), 0.0)
        # Allowed: B brokers the A→C path
        self.assertGreater(_betw(allowed, "B"), 0.0)

    def test_post_gap_brokerage_counts(self):
        """A broker acting entirely inside a post-gap segment scores.

        Regression test: the foremost-path pass used to die at the gap,
        zeroing betweenness for all post-gap activity. A-C then C-D after
        the gap: C brokers the within-segment A->D pair (raw dependency 1).
        """
        def g(edges):
            gr = ig.Graph(n=4)
            gr.vs["name"] = ["A", "B", "C", "D"]
            gr.add_edges(edges)
            return gr

        graphs = [g([(0, 1)]), g([(0, 2)]), g([(2, 3)])]
        labels = ["2024-01", "2024-03", "2024-04"]  # gap after 2024-01
        bt = temporal_betweenness(graphs, graph_labels=labels,
                                  cross_gaps=False, normalized=False,
                                  report_gaps=False)
        self.assertEqual(_betw(bt, "C"), 1.0)

    def test_pair_counted_once_across_segments(self):
        """A pair reachable in two segments is counted at its foremost
        arrival only — the second segment must not double the credit."""
        def g(edges):
            gr = ig.Graph(n=3)
            gr.vs["name"] = ["A", "B", "C"]
            gr.add_edges(edges)
            return gr

        # A-B, B-C before the gap; the same pattern again after it.
        graphs = [g([(0, 1)]), g([(1, 2)]), g([(0, 1)]), g([(1, 2)])]
        labels = ["2024-01", "2024-02", "2024-04", "2024-05"]
        bt = temporal_betweenness(graphs, graph_labels=labels,
                                  cross_gaps=False, normalized=False,
                                  report_gaps=False)
        # B brokers A->C, whose foremost arrival is in the first segment.
        # The identical second-segment route must not add credit (a
        # double-counting bug would report 2.0). C->A is unreachable in
        # both segments (edges appear in the wrong temporal order), so
        # the total is exactly 1.0.
        self.assertEqual(_betw(bt, "B"), 1.0)

    def test_save_path_writes_pdf(self):
        graphs, labels = _chain()
        with tempfile.TemporaryDirectory() as tmp:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                temporal_betweenness(graphs, graph_labels=labels,
                                     save_path=tmp, report_gaps=False)
            files = os.listdir(tmp)
        self.assertIn("temporal_betweenness.pdf", files)

    def test_source_exception_warns_and_skips(self):
        g0 = ig.Graph(n=3, edges=[(0, 1)])
        with patch("temporal_networks.temporal_paths."
                   "_foremost_paths_from_source",
                   side_effect=ig.InternalError("boom")):
            with self.assertWarns(UserWarning):
                bt = temporal_betweenness([g0], graph_labels=["t"],
                                          report_gaps=False)
        # All sources skipped → every node has zero betweenness
        self.assertTrue((bt["betweenness"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
