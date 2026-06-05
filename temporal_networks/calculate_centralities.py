"""
Temporal Network Analysis: Centrality Measures Module (WITH GAP REPORTING)

This module provides functions for computing various centrality measures that
quantify the importance and influence of individual nodes within networks.
Supports computation across temporal networks with proper handling of node labels
and temporal gaps.

KEY FEATURES:
- Computes multiple centrality measures (degree, betweenness, PageRank, etc.)
- Automatically detects and reports temporal gaps in your data
- Optionally visualizes how node centralities change over time
- Handles gapped data correctly in temporal plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Dict
from ._gap_utilities import (
    detect_temporal_gaps,
    print_gap_report,
    plot_with_gap_handling,
    validate_and_setup_graphs
)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def calculate_centralities(graphs: List,
                          graph_labels: Optional[List[str]] = None,
                          filename: Optional[str] = "centralities_results.csv",
                          report_gaps: bool = True,
                          visualize_evolution: bool = False,
                          save_path: str = "plots/") -> pd.DataFrame:
    """
    Compute comprehensive centrality measures for nodes across multiple graphs.

    Calculates various centrality metrics for each node in a collection of networks,
    including degree, closeness, betweenness, eigenvector, PageRank, harmonic,
    eccentricity, clustering coefficient, and HITS scores.

    **KEY FEATURE:** Automatically detects and reports temporal gaps in your data.
    Optionally visualizes how node centralities evolve over time with proper gap handling.

    Parameters
    ----------
    graphs : list of igraph.Graph
        List of igraph.Graph objects to analyze
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        Supports multiple formats: YYYY-MM, YYYY-MM-DD, YYYY-W##, YYYY-Q#, YYYY
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    filename : str, optional
        CSV filename for saving results (default: "centralities_results.csv")
        If None, results are not saved to file
    report_gaps : bool, optional
        If True (default), analyzes and reports temporal gaps to the console
    visualize_evolution : bool, optional
        If True, creates plots showing how centrality measures evolve over time
        (default: False). This is useful for tracking important nodes.
    save_path : str, optional
        Directory for saving visualizations (default: "plots/")

    Returns
    -------
    pandas.DataFrame
        DataFrame with centrality measures, one row per node per graph.
        Columns include: Graph, Node, and various centrality measures

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import calculate_centralities
    >>>
    >>> # Continuous data (no gaps)
    >>> graphs = [ig.Graph.Barabasi(n=50, m=2) for _ in range(12)]
    >>> labels = [f"2024-{i+1:02d}" for i in range(12)]
    >>> centralities = calculate_centralities(graphs, graph_labels=labels)
    >>>
    >>> # Gapped data (seasonal operation)
    >>> labels_seasonal = ["2024-03", "2024-04", "2024-05", "2024-06",
    ...                    "2024-07", "2024-08", "2024-11", "2024-12"]
    >>> centralities = calculate_centralities(graphs[:8], graph_labels=labels_seasonal)
    >>> # Will report: Gap between 2024-08 and 2024-11

    Notes
    -----
    Centrality measures computed:
    - Degree: Number of direct connections
    - Closeness: Average distance to all other nodes
    - Betweenness: Number of shortest paths node lies on
    - Eigenvector: Importance based on connections to important nodes
    - PageRank: Probabilistic measure of node importance
    - Harmonic: Closeness variant less sensitive to disconnected components
    - Eccentricity: Greatest distance to any other node
    - Clustering Coefficient: Tendency of neighbors to be connected
    - HITS Authority/Hub: Measures from hyperlink-induced topic search

    Temporal Gaps:
    - Automatically analyzes temporal labels for gaps
    - Reports if data has missing periods
    - Temporal plots preserve gaps as visual breaks
    """

    # Validate inputs and set up labels
    graph_labels = validate_and_setup_graphs(graphs, graph_labels)

    # Analyze temporal gaps
    gap_info = detect_temporal_gaps(graph_labels)

    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    # Create output directory if needed
    if save_path and visualize_evolution:
        os.makedirs(save_path, exist_ok=True)

    # Initialize list to store centrality measures for all nodes across all graphs
    centralities_list = []

    # Iterate over each graph
    for graph_idx, graph in enumerate(graphs):
        graph_name = graph_labels[graph_idx]
        vcount = graph.vcount()

        vcount = graph.vcount()

        vcount = graph.vcount()

        n_nodes = graph.vcount()

        # Cache graph.vcount() to avoid redundant method calls
        v_count = graph.vcount()

        # Cache the number of nodes to avoid redundant method calls
        num_nodes = graph.vcount()

        # Get node labels from graph attributes
        if "name" in graph.vs.attributes():
            node_labels = graph.vs["name"]
        elif "label" in graph.vs.attributes():
            node_labels = graph.vs["label"]
        else:
            node_labels = [f"Node_{i}" for i in range(num_nodes)]
            node_labels = [f"Node_{i}" for i in range(n_nodes)]
            node_labels = [f"Node_{i}" for i in range(vcount)]
            print(f"Warning: Graph {graph_name} has no 'name' or 'label' attribute. Using node indices.")

        # Compute centrality measures
        vcount = graph.vcount()
        none_list = [None] * vcount

        try:
            degree_centrality = graph.degree()
        except Exception:
            degree_centrality = [None] * num_nodes
            degree_centrality = [None] * n_nodes
            degree_centrality = [None] * vcount

        try:
            closeness_centrality = graph.closeness()
        except Exception:
            closeness_centrality = [None] * num_nodes
            closeness_centrality = [None] * n_nodes
            closeness_centrality = [None] * vcount

        try:
            betweenness_centrality = graph.betweenness(directed=graph.is_directed())
        except Exception:
            betweenness_centrality = [None] * num_nodes
            betweenness_centrality = [None] * n_nodes
            betweenness_centrality = [None] * vcount

        try:
            eigenvector_centrality = graph.eigenvector_centrality()
        except Exception:
            eigenvector_centrality = [None] * num_nodes
            eigenvector_centrality = [None] * n_nodes
            eigenvector_centrality = [None] * vcount

        try:
            pagerank = graph.pagerank()
        except Exception:
            pagerank = [None] * num_nodes
            pagerank = [None] * n_nodes
            pagerank = [None] * vcount

        try:
            harmonic_centrality = graph.harmonic_centrality()
        except Exception:
            harmonic_centrality = [None] * num_nodes
            harmonic_centrality = [None] * n_nodes
            harmonic_centrality = [None] * vcount

        try:
            eccentricity = graph.eccentricity()
        except Exception:
            eccentricity = [None] * num_nodes
            eccentricity = [None] * n_nodes
            eccentricity = [None] * vcount

        try:
            clustering_coefficient = graph.transitivity_local_undirected()
        except Exception:
            clustering_coefficient = [None] * num_nodes
            clustering_coefficient = [None] * n_nodes
            clustering_coefficient = [None] * vcount

        try:
            authority_score = graph.authority_score()
        except Exception:
            authority_score = [None] * num_nodes
            authority_score = [None] * n_nodes
            authority_score = [None] * vcount

        try:
            hub_score = graph.hub_score()
        except Exception:
            hub_score = [None] * num_nodes
            hub_score = [None] * n_nodes
            hub_score = [None] * vcount

        # For each node, store all centrality measures
        for node_idx, node_label in enumerate(node_labels):
            centrality_dict = {
                "Graph": graph_name,
                "Node": node_label,
                "Node_Index": node_idx,
                "Degree_Centrality": degree_centrality[node_idx],
                "Closeness_Centrality": closeness_centrality[node_idx],
                "Betweenness_Centrality": betweenness_centrality[node_idx],
                "Eigenvector_Centrality": eigenvector_centrality[node_idx],
                "PageRank": pagerank[node_idx],
                "Harmonic_Centrality": harmonic_centrality[node_idx],
                "Eccentricity": eccentricity[node_idx],
                "Clustering_Coefficient": clustering_coefficient[node_idx],
                "HITS_Authority": authority_score[node_idx],
                "HITS_Hub": hub_score[node_idx],
            }
            centralities_list.append(centrality_dict)

    # Convert to DataFrame
    centralities_df = pd.DataFrame(centralities_list)

    # Save to CSV if requested
    if filename:
        try:
            centralities_df.to_csv(filename, index=False)
            print(f"✓ Centralities results saved to {filename}")
        except Exception as e:
            print(f"Error saving centralities to CSV: {e}")

    # Optional: Visualize centrality evolution over time
    if visualize_evolution:
        _visualize_centrality_evolution(centralities_df, graph_labels, gap_info, save_path)

    return centralities_df


def _visualize_centrality_evolution(centralities_df: pd.DataFrame,
                                   graph_labels: List[str],
                                   gap_info: Dict,
                                   save_path: str) -> None:
    """
    Visualize how centrality measures evolve over time.

    Creates plots showing temporal evolution of average centrality measures
    across all nodes, with proper gap handling.
    """

    # Calculate average centrality measures per time step
    centrality_measures = [
        "Degree_Centrality",
        "Betweenness_Centrality",
        "PageRank",
        "Closeness_Centrality",
    ]

    for measure in centrality_measures:
        if measure not in centralities_df.columns:
            continue

        try:
            # Get average centrality for each graph
            avg_by_graph = centralities_df.groupby("Graph")[measure].mean().reset_index()
            # Ensure correct order based on graph_labels
            avg_by_graph = avg_by_graph.set_index("Graph").reindex(graph_labels).reset_index()
            avg_values = avg_by_graph[measure].values

            # Create plot
            fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

            # Plot with gap handling
            plot_with_gap_handling(ax, graph_labels, avg_values,
                                  gap_info["segments"],
                                  marker='o', linestyle='-', markersize=8,
                                  linewidth=2, color='#1f77b4')

            ax.set_xlabel("Year - Month", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"Average {measure.replace('_', ' ')}", fontsize=12, fontweight='bold')
            ax.set_title(f"Temporal Evolution: {measure.replace('_', ' ')}", fontsize=14, fontweight='bold')

            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            plot_filename = os.path.join(save_path, f"centrality_evolution_{measure}.pdf")
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Centrality evolution plot saved: {plot_filename}")

        except Exception as e:
            print(f"Warning: Could not plot {measure}: {e}")
