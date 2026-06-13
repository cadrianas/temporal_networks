import unittest
import os
import igraph as ig
from unittest.mock import patch
from temporal_networks.plot_community_evolution import plot_community_evolution

class TestPlotCommunityEvolution(unittest.TestCase):
    def setUp(self):
        self.output_file = "test_community_evolution.html"

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_plot_community_evolution_success(self):
        # Create a small list of graphs
        graphs = [ig.Graph.Barabasi(n=10, m=2) for _ in range(3)]

        # Test that the function does not raise any exceptions
        try:
            plot_community_evolution(graphs, community_algorithm="louvain", output_file=self.output_file)
        except Exception as e:
            self.fail(f"plot_community_evolution raised Exception unexpectedly: {e}")

        # Test that the output file is created
        self.assertTrue(os.path.exists(self.output_file))

    def test_plot_community_evolution_empty_graphs(self):
        with self.assertRaises(ValueError):
            plot_community_evolution([], community_algorithm="louvain", output_file=self.output_file)

    def test_plot_community_evolution_invalid_algorithm(self):
        graphs = [ig.Graph.Barabasi(n=10, m=2) for _ in range(3)]
        with self.assertRaises(ValueError):
            plot_community_evolution(graphs, community_algorithm="invalid_algo", output_file=self.output_file)

    def test_louvain_on_directed_graphs(self):
        """Regression (bug #2): louvain must be converted to undirected, not crash."""
        graphs = [ig.Graph.Barabasi(n=20, m=2, directed=True) for _ in range(3)]
        plot_community_evolution(graphs, community_algorithm="louvain",
                                 output_file=self.output_file)
        self.assertTrue(os.path.exists(self.output_file))

    def test_edge_betweenness_on_directed_graphs(self):
        """Regression (bug #3): edge_betweenness dendrogram needs as_clustering()."""
        graphs = [ig.Graph.Barabasi(n=20, m=2, directed=True) for _ in range(3)]
        plot_community_evolution(graphs, community_algorithm="edge_betweenness",
                                 output_file=self.output_file)
        self.assertTrue(os.path.exists(self.output_file))

if __name__ == '__main__':
    unittest.main()
