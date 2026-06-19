import contextlib
import io
import os
import tempfile
import unittest

import igraph as ig

from temporal_networks.community_tracking import (
    track_communities,
    plot_community_lineage,
)
from temporal_networks._gap_utilities import detect_temporal_gaps


# ---------------------------------------------------------------------------
# Graph builders — disjoint cliques give deterministic communities
# ---------------------------------------------------------------------------

def _clique(names):
    """Complete graph over the named vertices."""
    g = ig.Graph()
    g.add_vertices(list(names))
    edges = [(names[a], names[b])
             for a in range(len(names))
             for b in range(a + 1, len(names))]
    g.add_edges(edges)
    return g


def _two_cliques(group_a, group_b):
    """One graph with two disjoint cliques (two connected components)."""
    names = list(group_a) + list(group_b)
    g = ig.Graph()
    g.add_vertices(names)
    edges = []
    for grp in (group_a, group_b):
        edges += [(grp[a], grp[b])
                  for a in range(len(grp))
                  for b in range(a + 1, len(grp))]
    g.add_edges(edges)
    return g


A = ["n0", "n1", "n2"]
B = ["n3", "n4", "n5"]


def _track(graphs, labels, **kw):
    """Run track_communities with console output suppressed."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        df = track_communities(graphs, graph_labels=labels,
                               algorithm="louvain", report_gaps=False, **kw)
    return df


def _events_at(df, label):
    return sorted(df[df["Graph"] == label]["event"].tolist())


def _lineages_at(df, label):
    return set(df[df["Graph"] == label]["lineage_id"])


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

class TestMerge(unittest.TestCase):

    def test_two_communities_merge_into_one(self):
        g0 = _two_cliques(A, B)        # two triangles
        g1 = _clique(A + B)            # one big clique
        df = _track([g0, g1], ["t0", "t1"])

        # snapshot 0: two births
        self.assertEqual(_events_at(df, "t0"), ["birth", "birth"])
        # snapshot 1: a single merged community
        t1 = df[df["Graph"] == "t1"]
        self.assertEqual(len(t1), 1)
        self.assertEqual(t1.iloc[0]["event"], "merge")
        self.assertEqual(t1.iloc[0]["size"], 6)

    def test_merge_inherits_a_predecessor_lineage(self):
        g0 = _two_cliques(A, B)
        g1 = _clique(A + B)
        df = _track([g0, g1], ["t0", "t1"])
        merged_lineage = df[df["event"] == "merge"]["lineage_id"].iloc[0]
        self.assertIn(merged_lineage, _lineages_at(df, "t0"))


class TestSplit(unittest.TestCase):

    def test_one_community_splits_into_two(self):
        g0 = _clique(A + B)            # one big clique
        g1 = _two_cliques(A, B)        # two triangles
        df = _track([g0, g1], ["t0", "t1"])

        self.assertEqual(_events_at(df, "t0"), ["birth"])
        self.assertEqual(_events_at(df, "t1"), ["split", "split"])


class TestBirthDeath(unittest.TestCase):

    def test_birth_of_new_community(self):
        g0 = _clique(A)               # only n0,n1,n2 exist
        g1 = _two_cliques(A, B)       # n3,n4,n5 appear as a new community
        df = _track([g0, g1], ["t0", "t1"])
        self.assertEqual(_events_at(df, "t1"), ["birth", "continue"])

    def test_death_of_community(self):
        # B persists through t0, t1, then disappears at t2 -> dies at t1
        g0 = _two_cliques(A, B)
        g1 = _two_cliques(A, B)
        g2 = _clique(A)               # n3,n4,n5 community is gone
        df = _track([g0, g1, g2], ["t0", "t1", "t2"])
        self.assertIn("death", _events_at(df, "t1"))
        self.assertEqual(_events_at(df, "t2"), ["continue"])


# ---------------------------------------------------------------------------
# match_threshold sensitivity
# ---------------------------------------------------------------------------

class TestMatchThreshold(unittest.TestCase):

    def test_weak_match_links_at_low_threshold(self):
        # 4-cliques sharing 2 of 6 nodes -> Jaccard = 2/6 = 0.333
        g0 = _clique(["n0", "n1", "n2", "n3"])
        g1 = _clique(["n2", "n3", "n4", "n5"])
        df = _track([g0, g1], ["t0", "t1"], match_threshold=0.3)
        # Linked: lineage continues across snapshots
        self.assertEqual(_lineages_at(df, "t0"), _lineages_at(df, "t1"))
        self.assertEqual(_events_at(df, "t1"), ["continue"])

    def test_weak_match_breaks_at_high_threshold(self):
        g0 = _clique(["n0", "n1", "n2", "n3"])
        g1 = _clique(["n2", "n3", "n4", "n5"])
        df = _track([g0, g1], ["t0", "t1"], match_threshold=0.5)
        # Not linked: fresh lineage, the t1 community is a birth
        self.assertEqual(_events_at(df, "t1"), ["birth"])
        self.assertTrue(
            _lineages_at(df, "t0").isdisjoint(_lineages_at(df, "t1")))


# ---------------------------------------------------------------------------
# Gap-awareness
# ---------------------------------------------------------------------------

class TestGapAwareness(unittest.TestCase):

    def test_lineage_not_bridged_across_gap(self):
        g = _two_cliques(A, B)
        # 2024-02 missing -> gap between the two snapshots
        labels = ["2024-01", "2024-03"]
        df = _track([g, g.copy()], labels, bridge_gaps=False)
        # Communities after the gap start fresh lineages
        self.assertEqual(_events_at(df, "2024-03"), ["birth", "birth"])
        self.assertTrue(
            _lineages_at(df, "2024-01").isdisjoint(
                _lineages_at(df, "2024-03")))

    def test_lineage_bridged_when_requested(self):
        g = _two_cliques(A, B)
        labels = ["2024-01", "2024-03"]
        df = _track([g, g.copy()], labels, bridge_gaps=True)
        # Same communities -> continue, lineages preserved across the gap
        self.assertIn("continue", _events_at(df, "2024-03"))
        self.assertEqual(_lineages_at(df, "2024-01"),
                         _lineages_at(df, "2024-03"))


# ---------------------------------------------------------------------------
# Structure / validation
# ---------------------------------------------------------------------------

class TestStructure(unittest.TestCase):

    def test_output_columns(self):
        g = _two_cliques(A, B)
        df = _track([g, g.copy()], ["t0", "t1"])
        self.assertEqual(list(df.columns),
                         ["Graph", "community_id", "lineage_id", "size",
                          "event", "members"])

    def test_members_are_node_keys(self):
        g0 = _two_cliques(A, B)
        df = _track([g0, g0.copy()], ["t0", "t1"])
        all_members = set()
        for m in df["members"]:
            all_members.update(m)
        self.assertEqual(all_members, set(A) | set(B))

    def test_invalid_algorithm_raises(self):
        g = _clique(A)
        with self.assertRaises(ValueError):
            track_communities([g, g.copy()], graph_labels=["t0", "t1"],
                              algorithm="bogus", report_gaps=False)

    def test_single_snapshot_all_birth(self):
        g = _two_cliques(A, B)
        df = _track([g], ["t0"])
        self.assertEqual(_events_at(df, "t0"), ["birth", "birth"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

class TestPlotLineage(unittest.TestCase):

    def test_save_path_writes_pdf(self):
        g0 = _two_cliques(A, B)
        g1 = _clique(A + B)
        labels = ["t0", "t1"]
        df = _track([g0, g1], labels)
        gap_info = detect_temporal_gaps(labels)
        with tempfile.TemporaryDirectory() as tmp:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                plot_community_lineage(df, labels, gap_info, save_path=tmp)
            files = os.listdir(tmp)
        self.assertIn("community_lineage.pdf", files)

    def test_none_save_path_writes_nothing(self):
        g = _two_cliques(A, B)
        labels = ["t0", "t1"]
        df = _track([g, g.copy()], labels)
        gap_info = detect_temporal_gaps(labels)
        # Should simply return without error and without writing
        self.assertIsNone(
            plot_community_lineage(df, labels, gap_info, save_path=None))


if __name__ == "__main__":
    unittest.main()
