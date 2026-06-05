import time
import igraph as ig
from temporal_networks import calculate_centralities

def create_large_graphs():
    return [ig.Graph.Barabasi(n=2000, m=2) for _ in range(5)]

if __name__ == "__main__":
    graphs = create_large_graphs()

    start_time = time.time()
    for _ in range(3):
        calculate_centralities(graphs, filename=None, report_gaps=False)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")
