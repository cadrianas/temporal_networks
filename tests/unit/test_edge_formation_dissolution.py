import unittest
from unittest.mock import patch
import io
import contextlib
import igraph as ig
import pandas as pd
from temporal_networks.edge_formation_dissolution import compute_edge_dynamics

class TestEdgeFormationDissolutionException(unittest.TestCase):
    def test_compute_edge_dynamics_exception(self):
        """Test error handling in compute_edge_dynamics when get_edgelist fails."""
        g1 = ig.Graph.Erdos_Renyi(n=5, p=0.5)
        g2 = ig.Graph.Erdos_Renyi(n=5, p=0.5)
        graphs = [g1, g2]
        labels = ["Graph 1", "Graph 2"]

        # We patch igraph.Graph.get_edgelist to raise an Exception. The
        # failing pair must warn and yield a NaN row (shape preserved).
        with patch('igraph.Graph.get_edgelist', side_effect=Exception("Test Exception")):
            with self.assertWarns(UserWarning):
                results = compute_edge_dynamics(
                    graphs=graphs,
                    graph_labels=labels
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(results.loc[0, "Graph"], "Graph 2")
        self.assertTrue(pd.isna(results.loc[0, "Edges_Formed"]))
        self.assertTrue(pd.isna(results.loc[0, "Edges_Dissolved"]))


class TestEdgeFormationDissolution(unittest.TestCase):
    def test_gap_straddling_pair_is_nan(self):
        """A pair across a detected gap is NaN, matching snapshot_similarity.

        Regression test: edge dynamics used to compare gap-straddling
        snapshots as if they were consecutive.
        """
        g0 = ig.Graph(n=3, edges=[(0, 1)])
        g1 = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
        g2 = ig.Graph(n=3, edges=[(1, 2)])
        # 2024-03 is missing -> gap between the 2nd and 3rd snapshots.
        labels = ["2024-01", "2024-02", "2024-04"]

        df = compute_edge_dynamics([g0, g1, g2], graph_labels=labels)

        self.assertEqual(len(df), 2)
        # Normal pair still computed exactly.
        self.assertEqual(df.loc[0, "Graph"], "2024-02")
        self.assertEqual(df.loc[0, "Edges_Formed"], 1)
        self.assertEqual(df.loc[0, "Edges_Dissolved"], 0)
        # Gap-straddling pair reported as NaN in all metric columns.
        self.assertEqual(df.loc[1, "Graph"], "2024-04")
        for col in ["Edges_Formed", "Edges_Dissolved",
                    "Edges_Formed_Percent", "Edges_Dissolved_Percent"]:
            self.assertTrue(pd.isna(df.loc[1, col]))

    def test_compute_edge_dynamics_normal(self):
        """Test edge dynamics with normal graphs where edges are formed and dissolved."""
        # Graph 0: edges [(0, 1), (1, 2)]
        g0 = ig.Graph(n=4, edges=[(0, 1), (1, 2)])
        # Graph 1: edges [(0, 1), (2, 3)] -> dissolved (1, 2), formed (2, 3)
        g1 = ig.Graph(n=4, edges=[(0, 1), (2, 3)])
        # Graph 2: edges [(0, 1), (2, 3), (0, 3)] -> formed (0, 3)
        g2 = ig.Graph(n=4, edges=[(0, 1), (2, 3), (0, 3)])

        graphs = [g0, g1, g2]
        labels = ["G0", "G1", "G2"]

        df = compute_edge_dynamics(graphs, graph_labels=labels)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)  # Comparison starts from second graph

        # Check G1 vs G0
        row1 = df.iloc[0]
        self.assertEqual(row1["Graph"], "G1")
        self.assertEqual(row1["Edges_Formed"], 1)
        self.assertEqual(row1["Edges_Dissolved"], 1)
        self.assertEqual(row1["Edges_Formed_Percent"], 50.0)  # 1 / 2 * 100
        self.assertEqual(row1["Edges_Dissolved_Percent"], 50.0) # 1 / 2 * 100

        # Check G2 vs G1
        row2 = df.iloc[1]
        self.assertEqual(row2["Graph"], "G2")
        self.assertEqual(row2["Edges_Formed"], 1)
        self.assertEqual(row2["Edges_Dissolved"], 0)
        self.assertEqual(row2["Edges_Formed_Percent"], 50.0)  # 1 / 2 * 100
        self.assertEqual(row2["Edges_Dissolved_Percent"], 0.0) # 0 / 2 * 100

    def test_compute_edge_dynamics_no_changes(self):
        """Test edge dynamics when graphs are identical."""
        g0 = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
        g1 = ig.Graph(n=3, edges=[(0, 1), (1, 2)])

        graphs = [g0, g1]
        labels = ["G0", "G1"]

        df = compute_edge_dynamics(graphs, graph_labels=labels)

        self.assertEqual(len(df), 1)
        row1 = df.iloc[0]
        self.assertEqual(row1["Edges_Formed"], 0)
        self.assertEqual(row1["Edges_Dissolved"], 0)
        self.assertEqual(row1["Edges_Formed_Percent"], 0.0)
        self.assertEqual(row1["Edges_Dissolved_Percent"], 0.0)

    def test_compute_edge_dynamics_empty_graphs(self):
        """Test edge dynamics with graphs that have no edges."""
        g0 = ig.Graph(n=3, edges=[])
        g1 = ig.Graph(n=3, edges=[(0, 1)])

        graphs = [g0, g1]
        labels = ["G0", "G1"]

        df = compute_edge_dynamics(graphs, graph_labels=labels)

        self.assertEqual(len(df), 1)
        row1 = df.iloc[0]
        self.assertEqual(row1["Edges_Formed"], 1)
        self.assertEqual(row1["Edges_Dissolved"], 0)
        # Percentages when previous graph has 0 edges should be 0
        self.assertEqual(row1["Edges_Formed_Percent"], 0.0)
        self.assertEqual(row1["Edges_Dissolved_Percent"], 0.0)

    def test_compute_edge_dynamics_insufficient_graphs(self):
        """Test edge dynamics raises an error with fewer than 2 graphs."""
        g0 = ig.Graph(n=3, edges=[(0, 1)])
        graphs = [g0]

        with self.assertRaises(ValueError):
            compute_edge_dynamics(graphs)


class TestEdgeDynamicsNodeIdentity(unittest.TestCase):
    """Edges must be matched by node identity (name), not vertex index."""

    def test_same_edges_different_node_order(self):
        """Same edge A-B in both snapshots, but nodes stored in a different
        order, must yield zero formed/dissolved (regression for bug #4)."""
        g0 = ig.Graph(n=3, edges=[(0, 1)])      # edge between A and B
        g0.vs["name"] = ["A", "B", "C"]
        g1 = ig.Graph(n=3, edges=[(1, 2)])      # also A-B, but A,B at idx 1,2
        g1.vs["name"] = ["C", "A", "B"]

        df = compute_edge_dynamics([g0, g1], graph_labels=["t0", "t1"])
        row = df.iloc[0]
        # Index-based comparison would report (0,1) vs (1,2) => 1 formed, 1 dissolved
        self.assertEqual(row["Edges_Formed"], 0)
        self.assertEqual(row["Edges_Dissolved"], 0)

    def test_named_edge_change_detected(self):
        """A genuine change in named edges is still detected."""
        g0 = ig.Graph(n=3, edges=[(0, 1)])      # A-B
        g0.vs["name"] = ["A", "B", "C"]
        g1 = ig.Graph(n=3, edges=[(1, 2)])      # B-C
        g1.vs["name"] = ["A", "B", "C"]

        df = compute_edge_dynamics([g0, g1], graph_labels=["t0", "t1"])
        row = df.iloc[0]
        self.assertEqual(row["Edges_Formed"], 1)     # B-C formed
        self.assertEqual(row["Edges_Dissolved"], 1)  # A-B dissolved

    def test_unnamed_graphs_fall_back_to_indices(self):
        """Without name/label attributes, index-based comparison is preserved."""
        g0 = ig.Graph(n=4, edges=[(0, 1), (1, 2)])
        g1 = ig.Graph(n=4, edges=[(0, 1), (2, 3)])
        df = compute_edge_dynamics([g0, g1], graph_labels=["t0", "t1"])
        row = df.iloc[0]
        self.assertEqual(row["Edges_Formed"], 1)
        self.assertEqual(row["Edges_Dissolved"], 1)

if __name__ == '__main__':
    unittest.main()
