import unittest
import pandas as pd
import igraph as ig
import os
import shutil
from temporal_networks.vertex_properties import vertex_properties

class TestVertexProperties(unittest.TestCase):
    def setUp(self):
        # Create graphs for testing
        self.g1 = ig.Graph.Barabasi(n=50, m=2, directed=False)
        self.g2 = ig.Graph.Barabasi(n=50, m=2, directed=False)
        self.g3 = ig.Graph.Barabasi(n=50, m=2, directed=False)

        # Add node names
        self.g1.vs["name"] = [f"Node_{i}" for i in range(50)]
        self.g2.vs["name"] = [f"Node_{i}" for i in range(50)]
        self.g3.vs["name"] = [f"Node_{i}" for i in range(50)]

        self.graphs = [self.g1, self.g2, self.g3]
        self.labels = ["2019-01", "2019-02", "2019-03"]

        # Directory for plots/files
        self.test_save_path = "test_plots/"
        os.makedirs(self.test_save_path, exist_ok=True)

    def tearDown(self):
        # Clean up test directories
        if os.path.exists(self.test_save_path):
            shutil.rmtree(self.test_save_path)

    def test_vertex_properties_standard(self):
        """Test vertex_properties standard case returns DataFrame with correct columns"""
        df = vertex_properties(self.graphs, node_name="Node_5", graph_labels=self.labels,
                               save_path=self.test_save_path, visualisation=False, report_gaps=False)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["Graph"]), self.labels)

        expected_cols = [
            "Graph", "Degree_Centrality", "Closeness_Centrality",
            "Betweenness_Centrality", "Eigenvector_Centrality", "PageRank",
            "Harmonic_Centrality", "Eccentricity", "Clustering_Coefficient",
            "Constraint", "Coreness", "Authority_Score", "Hub_Score"
        ]

        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_vertex_properties_missing_node(self):
        """Test vertex_properties handles missing nodes correctly"""
        # Remove "Node_5" from g2 by changing its name
        node_idx = self.g2.vs.find(name="Node_5").index
        self.g2.vs[node_idx]["name"] = "Node_5_changed"

        with self.assertWarns(UserWarning):
            df = vertex_properties(self.graphs, node_name="Node_5",
                                   graph_labels=self.labels,
                                   save_path=self.test_save_path,
                                   visualisation=False, report_gaps=False)

        self.assertIsInstance(df, pd.DataFrame)
        # One row per graph: the snapshot missing Node_5 warns and is
        # reported as a NaN row, keeping output aligned with the labels.
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["Graph"]), self.labels)
        missing = df[df["Graph"] == "2019-02"].iloc[0]
        self.assertTrue(pd.isna(missing["Degree_Centrality"]))

    def test_vertex_properties_visualisation_false(self):
        """Test vertex_properties executes correctly with visualisation=False"""
        # Ensure directory is empty
        shutil.rmtree(self.test_save_path)
        os.makedirs(self.test_save_path, exist_ok=True)

        df = vertex_properties(self.graphs, node_name="Node_5", graph_labels=self.labels,
                               save_path=self.test_save_path, visualisation=False, report_gaps=False)

        # Directory should still just be empty, no files created
        files = os.listdir(self.test_save_path)
        self.assertEqual(len(files), 0)
        self.assertEqual(len(df), 3)

    def test_vertex_properties_label_attribute(self):
        """Test vertex_properties matching nodes using 'label' instead of 'name'"""
        # Remove name attribute, use label instead
        g1_label = self.g1.copy()
        g2_label = self.g2.copy()
        g3_label = self.g3.copy()

        del g1_label.vs["name"]
        del g2_label.vs["name"]
        del g3_label.vs["name"]

        g1_label.vs["label"] = [f"Label_{i}" for i in range(50)]
        g2_label.vs["label"] = [f"Label_{i}" for i in range(50)]
        g3_label.vs["label"] = [f"Label_{i}" for i in range(50)]

        graphs_label = [g1_label, g2_label, g3_label]

        df = vertex_properties(graphs_label, node_name="Label_10", graph_labels=self.labels,
                               save_path=self.test_save_path, visualisation=False, report_gaps=False)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)

if __name__ == '__main__':
    unittest.main()
