import unittest
from unittest.mock import patch
import io
import contextlib
import tempfile
import igraph as ig
from temporal_networks.communities_measures import communities_measures

class TestCommunitiesMeasuresException(unittest.TestCase):
    def test_communities_measures_exception(self):
        """Test error handling in communities_measures when an algorithm fails."""
        g = ig.Graph.Erdos_Renyi(n=5, p=0.5)
        graphs = [g]
        labels = ["Graph 1"]

        with tempfile.TemporaryDirectory() as tmpdirname:
            # We want to patch only one algorithm to fail to see the specific warning
            with patch('igraph.Graph.community_leiden', side_effect=Exception("Test Exception")):
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    results = communities_measures(
                        graphs=graphs,
                        graph_labels=labels,
                        save_path=tmpdirname,
                        visualisation=False,
                        report_gaps=False
                    )

                output = f.getvalue()
                # Verify that the warning message is correctly formatted and printed
                self.assertIn("Warning: Algorithm leiden failed on graph Graph 1: Test Exception", output)

                # Verify that the failed algorithm results are not in the dictionary
                self.assertNotIn("leiden", results)

                # Verify that other algorithms still complete and yield results
                self.assertIn("louvain", results)
                self.assertTrue(len(results) > 0)

if __name__ == '__main__':
    unittest.main()
