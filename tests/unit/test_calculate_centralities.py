import unittest
from unittest.mock import patch
import io
import contextlib
import tempfile
import pandas as pd
import igraph as ig
from temporal_networks.calculate_centralities import calculate_centralities

class TestCalculateCentralities(unittest.TestCase):
    def setUp(self):
        # Create a simple graph for testing
        self.graph = ig.Graph.Erdos_Renyi(n=5, p=0.5, directed=False)
        self.graphs = [self.graph, self.graph.copy()]
        self.labels = ["Graph 1", "Graph 2"]

    def test_calculate_centralities_happy_path(self):
        """Test that calculate_centralities returns expected DataFrame structure."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            df = calculate_centralities(
                graphs=self.graphs,
                graph_labels=self.labels,
                filename=None,
                report_gaps=False,
                visualize_evolution=False,
                save_path=tmpdirname
            )

            # Check if it returns a DataFrame
            self.assertIsInstance(df, pd.DataFrame)

            # Check expected columns
            expected_columns = [
                "Graph", "Node", "Node_Index", "Degree_Centrality",
                "Closeness_Centrality", "Betweenness_Centrality",
                "Eigenvector_Centrality", "PageRank", "Harmonic_Centrality",
                "Eccentricity", "Clustering_Coefficient", "HITS_Authority",
                "HITS_Hub"
            ]
            for col in expected_columns:
                self.assertIn(col, df.columns)

            # Check shape: 2 graphs, 5 nodes each = 10 rows
            self.assertEqual(df.shape[0], 10)

            # Check graph names
            self.assertListEqual(list(df["Graph"].unique()), self.labels)

    def test_node_label_extraction_name(self):
        """Test extraction of node labels from 'name' attribute."""
        g = ig.Graph(n=3)
        g.vs["name"] = ["A", "B", "C"]

        df = calculate_centralities(
            graphs=[g],
            graph_labels=["G1"],
            filename=None,
            report_gaps=False
        )

        self.assertListEqual(list(df["Node"]), ["A", "B", "C"])

    def test_node_label_extraction_label(self):
        """Test extraction of node labels from 'label' attribute."""
        g = ig.Graph(n=3)
        g.vs["label"] = ["X", "Y", "Z"]

        df = calculate_centralities(
            graphs=[g],
            graph_labels=["G1"],
            filename=None,
            report_gaps=False
        )

        self.assertListEqual(list(df["Node"]), ["X", "Y", "Z"])

    def test_node_label_extraction_fallback(self):
        """Test fallback node label generation when no name or label exists."""
        g = ig.Graph(n=3)

        with self.assertWarns(UserWarning) as ctx:
            df = calculate_centralities(
                graphs=[g],
                graph_labels=["G1"],
                filename=None,
                report_gaps=False
            )

        self.assertListEqual(list(df["Node"]), ["Node_0", "Node_1", "Node_2"])
        self.assertIn("has no 'name' or 'label' attribute",
                      str(ctx.warning))

    def test_centrality_exception_handling(self):
        """Test handling of exceptions in centrality calculations."""
        # Patch eigenvector_centrality to raise an exception
        with patch('igraph.Graph.eigenvector_centrality', side_effect=Exception("Test Exception")):
            df = calculate_centralities(
                graphs=[self.graph],
                graph_labels=["G1"],
                filename=None,
                report_gaps=False
            )

            # Check that other centralities were computed (Degree should have valid values)
            self.assertFalse(df["Degree_Centrality"].isnull().all())

            # Check that the failed centrality has None/NaN values
            self.assertTrue(df["Eigenvector_Centrality"].isnull().all())

if __name__ == '__main__':
    unittest.main()
