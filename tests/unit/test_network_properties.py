import unittest
import pandas as pd
import numpy as np
import igraph as ig
from unittest.mock import patch, MagicMock

from temporal_networks.network_properties import network_properties
import importlib

# The package shadows the `network_properties` submodule with the function
# of the same name, so string-based patch targets can't resolve. Grab the
# real module via importlib and use patch.object against it.
_np_module = importlib.import_module("temporal_networks.network_properties")

class TestNetworkProperties(unittest.TestCase):
    def setUp(self):
        # Create a basic undirected graph
        self.g1 = ig.Graph.Erdos_Renyi(n=10, p=0.5, directed=False)
        # Create a disconnected graph
        self.g2 = ig.Graph(n=5, edges=[(0, 1), (2, 3)], directed=False)
        # Create a directed graph
        self.g3 = ig.Graph.Erdos_Renyi(n=10, p=0.5, directed=True)
        # Create an empty graph
        self.g4 = ig.Graph()

    def test_basic_properties(self):
        """Test if network properties are correctly computed for a list of graphs."""
        graphs = [self.g1, self.g2]
        labels = ["G1", "G2"]

        # Run function without plotting and gap reporting
        df = network_properties(
            graphs=graphs,
            graph_labels=labels,
            save_path=None,
            visualisation=False,
            report_gaps=False
        )

        # Check return type
        self.assertIsInstance(df, pd.DataFrame)

        # Check rows and columns
        self.assertEqual(len(df), 2)
        expected_columns = [
            "Graph", "Number of Nodes", "Number of Edges", "Density",
            "Strongly Connected Components", "Girth", "Diameter",
            "Average Path Length", "Mean Degree", "Reciprocity",
            "Transitivity", "Is Bipartite", "Is Connected", "Is DAG",
            "Is Directed", "Is Named", "Is Simple", "Is Weighted", "Has Multiple"
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Check values for g1
        self.assertEqual(df.loc[0, "Graph"], "G1")
        self.assertEqual(df.loc[0, "Number of Nodes"], 10)
        self.assertEqual(df.loc[0, "Number of Edges"], self.g1.ecount())
        self.assertAlmostEqual(df.loc[0, "Density"], self.g1.density())
        self.assertEqual(df.loc[0, "Is Directed"], False)

        # Check values for g2
        self.assertEqual(df.loc[1, "Graph"], "G2")
        self.assertEqual(df.loc[1, "Number of Nodes"], 5)
        self.assertEqual(df.loc[1, "Number of Edges"], 2)
        self.assertEqual(df.loc[1, "Is Connected"], False)

    def test_default_labels(self):
        """Test behavior when no labels are provided."""
        df = network_properties(
            graphs=[self.g1, self.g2],
            graph_labels=None,
            save_path=None,
            visualisation=False,
            report_gaps=False
        )
        self.assertEqual(df.loc[0, "Graph"], "Graph 1")
        self.assertEqual(df.loc[1, "Graph"], "Graph 2")

    def test_empty_graph_list(self):
        """Test behavior when an empty list of graphs is passed."""
        with self.assertRaises(ValueError) as context:
            network_properties(
                graphs=[],
                save_path=None,
                visualisation=False,
                report_gaps=False
            )
        self.assertIn("graphs list cannot be empty", str(context.exception))

    def test_save_to_csv(self):
        """Test saving results to CSV."""
        graphs = [self.g1]
        filename = "test_output.csv"

        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            network_properties(
                graphs=graphs,
                filename=filename,
                save_path=None,
                visualisation=False,
                report_gaps=False
            )
            mock_to_csv.assert_called_once_with(filename, index=False)

    def test_visualisation_called(self):
        """Test if visualisation functions are called when visualisation=True."""
        graphs = [self.g1]

        with patch.object(_np_module, '_plot_properties') as mock_plot:
            network_properties(
                graphs=graphs,
                save_path="test_plots/",
                visualisation=True,
                report_gaps=False
            )
            mock_plot.assert_called_once()

    def test_report_gaps_called(self):
        """Test if gap reporting is triggered when report_gaps=True."""
        graphs = [self.g1]
        labels = ["2024-01"]

        with patch.object(_np_module, 'print_gap_report') as mock_print_gap:
            with patch.object(_np_module, 'detect_temporal_gaps') as mock_detect_gap:
                mock_detect_gap.return_value = {"segments": [], "gaps": []}
                network_properties(
                    graphs=graphs,
                    graph_labels=labels,
                    save_path=None,
                    visualisation=False,
                    report_gaps=True
                )
                mock_detect_gap.assert_called_once()
                mock_print_gap.assert_called_once()

    def test_graph_processing_error(self):
        """Test behavior when a graph raises an exception during processing."""
        # Create a mock graph that raises an exception when vcount() is called
        mock_graph = MagicMock()
        mock_graph.vcount.side_effect = Exception("Test Error")

        # It should warn, but continue or handle it based on the try-except logic
        # According to the code, if any exception occurs in the main block it will skip adding to arrays
        with patch('builtins.print') as mock_print:
            df = network_properties(
                graphs=[self.g1, mock_graph, self.g2],
                graph_labels=["G1", "ErrorGraph", "G2"],
                save_path=None,
                visualisation=False,
                report_gaps=False
            )

            # The properties lists are only filled for graphs that successfully pass the block.
            # In the code:
            # except Exception as e:
            #     print(f"Warning: Error processing graph {graph_labels[len(num_vertices)]}: {e}")
            #     continue

            # So the DataFrame should only have 2 rows (G1 and G2)
            self.assertEqual(len(df), 2)
            self.assertEqual(df.loc[0, "Graph"], "G1")
            # The implementation uses graph_labels[i] for each successfully
            # processed graph, so skipping the middle graph does not shift
            # labels: row 1 correctly corresponds to G2.
            self.assertEqual(df.loc[1, "Graph"], "G2")

            # Verify the warning was printed
            mock_print.assert_any_call("Warning: Error processing graph ErrorGraph: Test Error")

    def test_specific_property_exceptions(self):
        """Test behavior when specific properties (like diameter or girth) raise exceptions."""
        # An empty graph might raise exceptions for some properties like diameter or girth, or return NaN
        # igraph.Graph() has 0 vertices. avg_path_length might raise error or return NaN depending on igraph version

        # Instead of relying on igraph behavior, let's mock a graph that works for vcount etc,
        # but raises for diameter, girth, avg_path_length, and transitivity
        mock_graph = MagicMock()
        mock_graph.vcount.return_value = 10
        mock_graph.ecount.return_value = 5
        mock_graph.density.return_value = 0.1
        mock_graph.components.return_value = MagicMock(__len__=lambda self: 2)
        mock_graph.diameter.side_effect = Exception("Diameter Error")
        mock_graph.girth.side_effect = Exception("Girth Error")
        mock_graph.shortest_paths.side_effect = Exception("Shortest Paths Error")
        mock_graph.degree.return_value = [1, 2] # np.mean will be 1.5
        mock_graph.reciprocity.return_value = 0.5
        mock_graph.transitivity_undirected.side_effect = Exception("Transitivity Error")
        mock_graph.is_bipartite.return_value = False
        mock_graph.is_connected.return_value = False
        mock_graph.is_dag.return_value = False
        mock_graph.is_directed.return_value = False
        mock_graph.is_named.return_value = False
        mock_graph.is_simple.return_value = True
        mock_graph.is_weighted.return_value = False
        mock_graph.has_multiple.return_value = False

        df = network_properties(
            graphs=[mock_graph],
            graph_labels=["MockG"],
            save_path=None,
            visualisation=False,
            report_gaps=False
        )

        self.assertEqual(len(df), 1)
        self.assertTrue(np.isnan(df.loc[0, "Diameter"]))
        self.assertTrue(np.isnan(df.loc[0, "Girth"]))
        val = df.loc[0, "Average Path Length"]
        self.assertTrue(np.isnan(float(val)))
        self.assertTrue(np.isnan(df.loc[0, "Transitivity"]))

if __name__ == '__main__':
    unittest.main()
