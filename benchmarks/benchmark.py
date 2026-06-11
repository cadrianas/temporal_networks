import time
from unittest.mock import Mock
import pandas as pd
from temporal_networks import calculate_centralities

def run_benchmark():
    # create mock graphs
    graphs = []
    for i in range(1000):
        m = Mock()
        m.vcount.return_value = 100
        m.vs.attributes.return_value = ["name"]

        # Mock vs dict-like access
        mock_vs = Mock()
        mock_vs.__getitem__ = Mock(return_value=[f"n{j}" for j in range(100)])
        mock_vs.attributes.return_value = ["name"]
        m.vs = mock_vs

        # make all centrality methods raise exceptions
        m.degree.side_effect = Exception("error")
        m.closeness.side_effect = Exception("error")
        m.betweenness.side_effect = Exception("error")
        m.is_directed.return_value = False
        m.eigenvector_centrality.side_effect = Exception("error")
        m.pagerank.side_effect = Exception("error")
        m.harmonic_centrality.side_effect = Exception("error")
        m.eccentricity.side_effect = Exception("error")
        m.transitivity_local_undirected.side_effect = Exception("error")
        m.authority_score.side_effect = Exception("error")
        m.hub_score.side_effect = Exception("error")
        graphs.append(m)

    labels = [f"G{i}" for i in range(1000)]

    start = time.perf_counter()
    calculate_centralities(graphs, graph_labels=labels, filename=None, report_gaps=False)
    end = time.perf_counter()

    print(f"Time taken: {end - start:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
