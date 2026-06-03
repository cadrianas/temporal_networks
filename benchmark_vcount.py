import time
import igraph as ig
from temporal_networks import calculate_centralities

def benchmark():
    # Create 100 random graphs with 1000 nodes each
    graphs = [ig.Graph.Erdos_Renyi(n=500, p=0.01) for _ in range(50)]

    start_time = time.time()
    centralities = calculate_centralities(graphs, filename=None, report_gaps=False, visualize_evolution=False)
    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
