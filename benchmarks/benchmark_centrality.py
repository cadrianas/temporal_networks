import time
import igraph as ig
import pandas as pd
from temporal_networks.calculate_centralities import calculate_centralities

def benchmark():
    # Create a large graph
    graph = ig.Graph.Erdos_Renyi(n=1000, p=0.01)

    # We will mock the centrality methods to force exceptions
    def raise_err(*args, **kwargs):
        raise ValueError("Force exception")

    graph.degree = raise_err
    graph.closeness = raise_err
    graph.betweenness = raise_err
    graph.eigenvector_centrality = raise_err
    graph.pagerank = raise_err
    graph.harmonic_centrality = raise_err
    graph.eccentricity = raise_err
    graph.transitivity_local_undirected = raise_err
    graph.authority_score = raise_err
    graph.hub_score = raise_err

    graphs = [graph for _ in range(100)]

    start_time = time.time()
    calculate_centralities(graphs, filename=None, report_gaps=False, visualize_evolution=False)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
