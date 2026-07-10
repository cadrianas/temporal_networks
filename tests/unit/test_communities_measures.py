import unittest
from unittest.mock import patch
import io
import contextlib
import tempfile
import warnings
import igraph as ig
import pandas as pd
from temporal_networks.communities_measures import communities_measures


def _quiet(**kwargs):
    """Run communities_measures with stdout progress suppressed."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        return communities_measures(**kwargs)


class TestCommunitiesMeasuresException(unittest.TestCase):
    def test_algorithm_failure_warns_and_keeps_key(self):
        """A failing algorithm warns and maps to an empty DataFrame —
        never a silently missing key (regression: KeyError for users)."""
        g = ig.Graph.Erdos_Renyi(n=5, p=0.5)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Patch one algorithm to fail to see the specific warning
            with patch('igraph.Graph.community_leiden',
                       side_effect=Exception("Test Exception")):
                with self.assertWarns(UserWarning):
                    results = _quiet(
                        graphs=[g],
                        graph_labels=["Graph 1"],
                        save_path=tmpdirname,
                        visualisation=False,
                        report_gaps=False
                    )

        # The failed algorithm is present, with an empty DataFrame.
        self.assertIn("leiden", results)
        self.assertTrue(results["leiden"].empty)
        self.assertEqual(list(results["leiden"].columns),
                         ["Graph", "Node", "Community"])
        # Other algorithms still complete and yield results.
        self.assertIn("louvain", results)
        self.assertFalse(results["louvain"].empty)


class TestAlgorithmSelection(unittest.TestCase):
    def test_default_excludes_spinglass(self):
        """Spinglass fails on disconnected graphs, so the default run on a
        disconnected snapshot must not attempt it (regression: 1 warning
        per snapshot and a missing key)."""
        g = ig.Graph(n=6, edges=[(0, 1), (1, 2), (3, 4)])  # disconnected
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            results = _quiet(graphs=[g], graph_labels=["2024-01"],
                             save_path=None, visualisation=False,
                             report_gaps=False)
        self.assertNotIn("spinglass", results)
        self.assertEqual(
            set(results),
            {"leiden", "louvain", "walktrap", "fast_greedy",
             "label_prop", "infomap"})

    def test_spinglass_opt_in_on_connected_graph(self):
        g = ig.Graph.Full(n=6)
        results = _quiet(graphs=[g], graph_labels=["2024-01"],
                         algorithms=["spinglass"],
                         save_path=None, visualisation=False,
                         report_gaps=False)
        self.assertEqual(set(results), {"spinglass"})
        self.assertFalse(results["spinglass"].empty)

    def test_algorithms_subset_runs_only_those(self):
        g = ig.Graph.Full(n=5)
        results = _quiet(graphs=[g], graph_labels=["2024-01"],
                         algorithms=["louvain", "infomap"],
                         save_path=None, visualisation=False,
                         report_gaps=False)
        self.assertEqual(set(results), {"louvain", "infomap"})

    def test_unknown_algorithm_raises(self):
        g = ig.Graph.Full(n=5)
        with self.assertRaises(ValueError) as cm:
            communities_measures(graphs=[g], graph_labels=["2024-01"],
                                 algorithms=["louvain", "bogus"],
                                 save_path=None, visualisation=False,
                                 report_gaps=False)
        self.assertIn("bogus", str(cm.exception))


class TestCommunitiesMeasuresDirected(unittest.TestCase):
    """Regression: modularity-based algorithms must be converted to undirected,
    not silently dropped, when given directed graphs (bug #2)."""

    def test_louvain_and_leiden_run_on_directed_input(self):
        graphs = [ig.Graph.Barabasi(n=30, m=3, directed=True) for _ in range(2)]
        labels = ["2024-01", "2024-02"]
        results = _quiet(
            graphs=graphs, graph_labels=labels,
            save_path=None, visualisation=False, report_gaps=False)
        # Previously these produced "input graph must be undirected" and were
        # swallowed, so they were missing from the results dict.
        self.assertIn("louvain", results)
        self.assertIn("leiden", results)


if __name__ == '__main__':
    unittest.main()
