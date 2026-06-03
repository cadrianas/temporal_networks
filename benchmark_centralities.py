import time
import igraph as ig
from temporal_networks import calculate_centralities

def run_benchmark(num_graphs=1000, num_nodes=100):
    labels = [f"2024-{i+1:02d}" for i in range(num_graphs)]

    class BadGraph:
        def __init__(self, vcount_val):
            self._vcount_val = vcount_val
            self.vcount_calls = 0

            # Mock vs.attributes()
            class VS:
                def attributes(self):
                    return []
            self.vs = VS()

        def vcount(self):
            self.vcount_calls += 1
            # simulate some minor work or just the function call overhead
            return self._vcount_val

        def degree(self): raise Exception()
        def closeness(self): raise Exception()
        def betweenness(self, directed=False): raise Exception()
        def eigenvector_centrality(self): raise Exception()
        def pagerank(self): raise Exception()
        def harmonic_centrality(self): raise Exception()
        def eccentricity(self): raise Exception()
        def transitivity_local_undirected(self): raise Exception()
        def authority_score(self): raise Exception()
        def hub_score(self): raise Exception()
        def is_directed(self): return False

    bad_graphs = [BadGraph(num_nodes) for _ in range(num_graphs)]

    start = time.time()
    calculate_centralities(bad_graphs, graph_labels=labels, report_gaps=False, filename=None)
    end = time.time()

    total_calls = sum(g.vcount_calls for g in bad_graphs)
    return end - start, total_calls

if __name__ == "__main__":
    t, calls = run_benchmark(1000, 100)
    print(f"Benchmark took: {t:.4f} seconds")
    print(f"Total vcount calls: {calls}")
