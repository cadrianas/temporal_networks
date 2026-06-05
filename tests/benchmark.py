import time
import igraph as ig
import pandas as pd
import numpy as np
from temporal_networks.network_properties import network_properties

def create_test_graphs(num_graphs=100, nodes=50, edges=100):
    graphs = []
    labels = []
    for i in range(num_graphs):
        g = ig.Graph.Erdos_Renyi(n=nodes, m=edges)
        graphs.append(g)
        labels.append(f"G_{i}")
    return graphs, labels

def run_benchmark():
    graphs, labels = create_test_graphs(200, 50, 100)

    start_time = time.time()
    for _ in range(10): # Run 10 times to average out noise
        network_properties(graphs, graph_labels=labels, visualisation=False, report_gaps=False)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Benchmark completed in {duration:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
