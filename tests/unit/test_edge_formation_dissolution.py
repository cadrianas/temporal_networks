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

if __name__ == '__main__':
    unittest.main()
