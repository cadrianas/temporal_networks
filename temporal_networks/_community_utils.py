"""
Shared community-detection helper.

Promoted from ``plot_community_evolution`` so that both the visualization
and the lineage-tracking modules detect communities through one consistent
implementation rather than duplicating the algorithm-name mapping and the
directed/undirected and dendrogram-cut handling.
"""

import logging
import warnings

import igraph as ig
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

# Map public algorithm names to igraph functions. Names align with
# ``communities_measures`` and ``plot_community_evolution`` ("louvain"), with
# "multilevel" exposed as an alias for the same Louvain/multilevel method.
_ALGORITHM_MAP = {
    "edge_betweenness": ig.Graph.community_edge_betweenness,
    "walktrap": ig.Graph.community_walktrap,
    "fast_greedy": ig.Graph.community_fastgreedy,
    "label_prop": ig.Graph.community_label_propagation,
    "spinglass": ig.Graph.community_spinglass,
    "leiden": ig.Graph.community_leiden,
    "louvain": ig.Graph.community_multilevel,
    "multilevel": ig.Graph.community_multilevel,
    "infomap": ig.Graph.community_infomap,
}

# Algorithms that only operate on undirected graphs.
_UNDIRECTED_ONLY = {
    "walktrap", "fast_greedy", "label_prop", "spinglass",
    "louvain", "multilevel", "leiden",
}

# Algorithms whose result is a dendrogram that must be cut into a clustering.
_DENDROGRAM = {"walktrap", "fast_greedy", "edge_betweenness"}


def _detect_communities(graphs: List,
                        community_algorithm: str) -> Tuple[List[Any], str]:
    """
    Detect communities for each graph using the chosen algorithm.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Graphs on which to run community detection.
    community_algorithm : str
        Name of the community detection algorithm to apply. One of
        ``edge_betweenness``, ``walktrap``, ``fast_greedy``, ``label_prop``,
        ``spinglass``, ``leiden``, ``louvain`` / ``multilevel``, ``infomap``.

    Returns
    -------
    tuple of (list, str)
        The per-graph partitions (``None`` where detection failed) and the
        capitalized algorithm name.
    """
    algo_key = community_algorithm.lower()
    if algo_key not in _ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm: {community_algorithm}. "
                         f"Must be one of: {list(_ALGORITHM_MAP.keys())}")

    algo_func = _ALGORITHM_MAP[algo_key]
    algo_name = community_algorithm.capitalize()

    logger.info("Detecting communities with %s algorithm...", algo_name)

    communities_list: List[Any] = []
    for graph_idx, graph in enumerate(graphs):
        try:
            g = graph.copy()
            # Modularity-based algorithms only operate on undirected graphs.
            if g.is_directed() and algo_key in _UNDIRECTED_ONLY:
                g = g.as_undirected()

            try:
                # walktrap, fast_greedy and edge_betweenness return a
                # VertexDendrogram that must be cut into a flat clustering.
                if algo_key in _DENDROGRAM:
                    partition = algo_func(g).as_clustering()
                else:
                    partition = algo_func(g)
            except Exception as e:
                warnings.warn(f"Community detection failed for "
                              f"graph {graph_idx}: {e}; skipping "
                              f"this snapshot")
                communities_list.append(None)
                continue

            communities_list.append(partition)

        except Exception as e:
            warnings.warn(f"Error processing graph {graph_idx}: {e}; "
                          f"skipping this snapshot")
            communities_list.append(None)

    return communities_list, algo_name
