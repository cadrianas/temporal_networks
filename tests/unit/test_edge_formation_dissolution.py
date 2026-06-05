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

        # We patch igraph.Graph.get_edgelist to raise an Exception
        with patch('igraph.Graph.get_edgelist', side_effect=Exception("Test Exception")):
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                results = compute_edge_dynamics(
                    graphs=graphs,
                    graph_labels=labels
                )

            output = f.getvalue()
            # Verify that the warning message is correctly formatted and printed
            self.assertIn("Warning: Error comparing graphs 0 and 1: Test Exception", output)

            # Verify that the returned DataFrame is empty due to the exception skipping the iteration
            self.assertTrue(results.empty)
import igraph as ig
import pandas as pd

from temporal_networks.edge_formation_dissolution import compute_edge_dynamics


class TestEdgeFormationDissolution(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()
