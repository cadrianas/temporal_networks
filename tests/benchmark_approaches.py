import time
import igraph as ig
import pandas as pd
import numpy as np

def create_test_graphs(num_graphs=1000, nodes=50, edges=100):
    graphs = []
    for i in range(num_graphs):
        g = ig.Graph.Erdos_Renyi(n=nodes, m=edges)
        graphs.append(g)
    return graphs

def method1_original(graphs):
    num_vertices = []
    num_edges = []
    density = []
    strongly_connected_components = []
    girth = []
    diameter = []
    avg_path_length = []
    mean_degree = []
    reciprocity = []
    transitivity = []

    # Additional graph type properties
    is_bipartite = []
    is_connected = []
    is_dag = []
    is_directed = []
    is_named = []
    is_simple = []
    is_weighted = []
    has_multiple = []

    for graph in graphs:
        try:
            num_vertices.append(graph.vcount())
            num_edges.append(graph.ecount())
            density.append(graph.density())
            strongly_connected_components.append(len(graph.components(mode="STRONG")))

            try:
                diameter.append(graph.diameter())
            except Exception:
                diameter.append(np.nan)

            try:
                girth.append(graph.girth())
            except Exception:
                girth.append(np.nan)

            try:
                avg_path_length.append(np.mean(graph.distances()))
            except Exception:
                avg_path_length.append(np.nan)

            mean_degree.append(np.mean(graph.degree()))
            reciprocity.append(graph.reciprocity())

            try:
                transitivity.append(graph.transitivity_undirected())
            except Exception:
                transitivity.append(np.nan)

            is_bipartite.append(graph.is_bipartite())
            is_connected.append(graph.is_connected())
            is_dag.append(graph.is_dag())
            is_directed.append(graph.is_directed())
            is_named.append(graph.is_named())
            is_simple.append(graph.is_simple())
            is_weighted.append(graph.is_weighted())
            has_multiple.append(graph.has_multiple())
        except Exception:
            continue

    return pd.DataFrame({
        "Number of Nodes": num_vertices,
        "Number of Edges": num_edges,
        "Density": density,
        "Strongly Connected Components": strongly_connected_components,
        "Girth": girth,
        "Diameter": diameter,
        "Average Path Length": avg_path_length,
        "Mean Degree": mean_degree,
        "Reciprocity": reciprocity,
        "Transitivity": transitivity,
        "Is Bipartite": is_bipartite,
        "Is Connected": is_connected,
        "Is DAG": is_dag,
        "Is Directed": is_directed,
        "Is Named": is_named,
        "Is Simple": is_simple,
        "Is Weighted": is_weighted,
        "Has Multiple": has_multiple,
    })

def method13_dict_list_of_dicts(graphs):
    def get_graph_props(graph):
        try:
            # We want to use dictionary mapping
            try: d = graph.diameter()
            except Exception: d = np.nan

            try: g = graph.girth()
            except Exception: g = np.nan

            try: apl = np.mean(graph.distances())
            except Exception: apl = np.nan

            try: t = graph.transitivity_undirected()
            except Exception: t = np.nan

            return {
                "Number of Nodes": graph.vcount(),
                "Number of Edges": graph.ecount(),
                "Density": graph.density(),
                "Strongly Connected Components": len(graph.components(mode="STRONG")),
                "Girth": g,
                "Diameter": d,
                "Average Path Length": apl,
                "Mean Degree": np.mean(graph.degree()),
                "Reciprocity": graph.reciprocity(),
                "Transitivity": t,
                "Is Bipartite": graph.is_bipartite(),
                "Is Connected": graph.is_connected(),
                "Is DAG": graph.is_dag(),
                "Is Directed": graph.is_directed(),
                "Is Named": graph.is_named(),
                "Is Simple": graph.is_simple(),
                "Is Weighted": graph.is_weighted(),
                "Has Multiple": graph.has_multiple()
            }
        except Exception:
            return None

    results = []
    for g in graphs:
        res = get_graph_props(g)
        if res is not None:
            results.append(res)

    return pd.DataFrame(results)

if __name__ == "__main__":
    graphs = create_test_graphs(5000)

    for method, name in [
        (method1_original, "Original (18 lists)"),
        (method13_dict_list_of_dicts, "List of Dicts approach")
    ]:
        start = time.time()
        for _ in range(10):
            method(graphs)
        print(f"{name}: {time.time() - start:.4f}s")
