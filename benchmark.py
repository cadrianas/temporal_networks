import time
import igraph as ig
from temporal_networks.calculate_centralities import calculate_centralities

def run_benchmark():
    # Create some reasonably large graphs to measure the performance impact
    print("Generating graphs...")
    # 50 graphs with 1000 nodes each
    graphs = [ig.Graph.Barabasi(n=1000, m=2) for _ in range(50)]
    labels = [f"2024-{i+1:02d}" for i in range(50)]

    print("Running baseline...")
    start_time = time.time()
    # To trigger the exception paths or just normal paths
    # We will just run it normally
    centralities = calculate_centralities(graphs, graph_labels=labels, report_gaps=False, filename=None)
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"Time taken: {elapsed:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()