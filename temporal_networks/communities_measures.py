"""
Temporal Network Analysis: Community Detection Module (WITH GAP REPORTING)

This module provides functions for detecting and analyzing community structures
in networks using multiple algorithms. Supports temporal analysis by tracking
how communities evolve over time.

KEY FEATURES:
- Applies 7 different community detection algorithms
- Automatically detects and reports temporal gaps
- Plots correctly show gaps as visual breaks
- Tracks community structure evolution
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Dict
from ._gap_utilities import (
    detect_temporal_gaps,
    print_gap_report,
    plot_with_gap_handling,
    format_large_numbers,
    validate_and_setup_graphs
)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def communities_measures(graphs: List,
                        graph_labels: Optional[List[str]] = None,
                        save_path: Optional[str] = None,
                        visualisation: bool = True,
                        report_gaps: bool = True) -> dict:
    """
    Detect and analyze community structures using multiple algorithms.

    Applies various community detection algorithms to each graph in the temporal
    sequence and tracks how community structure evolves over time. Provides both
    detailed community membership and summary statistics.

    **KEY FEATURE:** Automatically detects and reports temporal gaps in your data.
    Plots correctly show gaps as visual breaks.

    Parameters
    ----------
    graphs : list of igraph.Graph
        List of igraph.Graph objects to analyze
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        Supports multiple formats: YYYY-MM, YYYY-MM-DD, YYYY-W##, YYYY-Q#, YYYY
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    save_path : str, optional
        Directory path for saving results and visualizations. If None
        (default), no files are saved.
    visualisation : bool, optional
        If True (default), generates plots for community evolution
    report_gaps : bool, optional
        If True (default), analyzes and reports temporal gaps to the console

    Returns
    -------
    dict
        Dictionary with results for each algorithm, mapping algorithm names to
        DataFrames containing community assignments

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import communities_measures
    >>> G1 = ig.Graph.Barabasi(n=100, m=2)
    >>> G2 = ig.Graph.Barabasi(n=100, m=2)
    >>> graphs = [G1, G2]
    >>> labels = ["2019-01", "2019-02"]
    >>> communities = communities_measures(graphs, graph_labels=labels)

    Notes
    -----
    Community detection algorithms used:
    - Leiden: Optimizes modularity with refined granularity
    - Louvain: Fast modularity optimization
    - Walktrap: Uses random walks to find communities
    - Fast Greedy: Greedy optimization of modularity
    - Label Propagation: Fast method based on label propagation
    - Spinglass: Based on statistical mechanics
    - Infomap: Information-theoretic approach

    Temporal Gaps:
    - Automatically analyzes temporal labels for gaps
    - Reports if data has missing periods
    - Community plots preserve gaps as visual breaks
    """

    # Validate inputs and set up labels
    graph_labels = validate_and_setup_graphs(graphs, graph_labels)

    # Analyze temporal gaps
    gap_info = detect_temporal_gaps(graph_labels)

    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    # Create output directory
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # Define community detection algorithms
    # Note: Some require undirected graphs, so we convert as needed
    community_algorithms = [
        ("leiden", "community_leiden"),
        ("louvain", "community_multilevel"),
        ("walktrap", "community_walktrap"),
        ("fast_greedy", "community_fastgreedy"),
        ("label_prop", "community_label_propagation"),
        ("spinglass", "community_spinglass"),
        ("infomap", "community_infomap")
    ]

    results = {}

    # Apply each algorithm
    for algo_name, algo_func in community_algorithms:
        print(f"\nProcessing algorithm: {algo_name}")
        all_communities = []
        num_communities_list = []

        # Apply algorithm to each graph
        for graph_idx, graph in enumerate(graphs):
            graph_label = graph_labels[graph_idx]

            try:
                # Convert to undirected for algorithms that require it
                g = graph.copy()
                if g.is_directed() and algo_func in {"community_walktrap",
                                                      "community_fastgreedy",
                                                      "community_label_propagation",
                                                      "community_spinglass"}:
                    g = g.as_undirected()

                # Simplify graph to remove multi-edges and loops
                g.simplify(multiple=True, loops=True, combine_edges=dict(weight="sum"))

                # Get weights if available
                weights = g.es["weight"] if "weight" in g.es.attributes() else None

                # Get node labels
                if "name" in g.vs.attributes():
                    node_labels = g.vs["name"]
                elif "label" in g.vs.attributes():
                    node_labels = g.vs["label"]
                else:
                    node_labels = [f"Node_{i}" for i in range(g.vcount())]

                # Detect communities using the specified algorithm
                try:
                    if algo_func in {"community_walktrap", "community_fastgreedy"}:
                        partition = getattr(
                            g, algo_func)(weights=weights).as_clustering()
                    elif algo_func == "community_infomap":
                        partition = getattr(g, algo_func)(edge_weights=weights)
                    else:
                        partition = getattr(g, algo_func)(weights=weights)
                except Exception as e:
                    print(f"  Warning: Algorithm {algo_name} failed on "
                          f"graph {graph_label}: {e}")
                    continue

                # Store community assignments
                for comm_idx, comm in enumerate(partition):
                    for node_idx in comm:
                        all_communities.append({
                            "Graph": graph_label,
                            "Node": node_labels[node_idx],
                            "Community": comm_idx
                        })

                # Calculate statistics for this graph
                num_communities_list.append({
                    "Graph": graph_label,
                    "Number_of_Communities": len(partition),
                    "Max_Community_Size": (
                        max(len(c) for c in partition) if partition else 0
                    ),
                    "Min_Community_Size": (
                        min(len(c) for c in partition) if partition else 0
                    ),
                    "Mean_Community_Size": (
                        sum(len(c) for c in partition) / len(partition)
                        if partition else 0
                    ),
                })

            except Exception as e:
                print(f"  Error processing graph {graph_label} with "
                      f"algorithm {algo_name}: {e}")
                continue

        # Convert to DataFrame and save
        if all_communities:
            communities_df = pd.DataFrame(all_communities)
            if save_path is not None:
                csv_filename = os.path.join(
                    save_path, f"communities_{algo_name}_assignments.csv")
                communities_df.to_csv(csv_filename, index=False)
                print(f"  ✓ Community assignments saved: {csv_filename}")
            results[algo_name] = communities_df
        else:
            print(f"  Warning: No communities detected for {algo_name}")
            continue

        # Save statistics
        if num_communities_list and save_path is not None:
            stats_df = pd.DataFrame(num_communities_list)
            stats_csv_filename = os.path.join(
                save_path, f"communities_{algo_name}_stats.csv")
            stats_df.to_csv(stats_csv_filename, index=False)
            print(f"  ✓ Community statistics saved: {stats_csv_filename}")

            # Generate visualizations with gap handling
            if visualisation:
                _plot_community_stats(stats_df, algo_name, graph_labels,
                                      gap_info, save_path)

    return results


def _plot_community_stats(stats_df: pd.DataFrame, algo_name: str,
                         graph_labels: List[str], gap_info: Dict,
                         save_path: str) -> None:
    """
    Helper function to plot community statistics with gap handling.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with community statistics
    algo_name : str
        Name of the algorithm
    graph_labels : list of str
        Temporal labels
    gap_info : dict
        Gap detection information
    save_path : str
        Path for saving plots
    """

    properties_to_plot = [
        ("Number_of_Communities", "Number of Communities"),
        ("Max_Community_Size", "Maximum Community Size"),
        ("Mean_Community_Size", "Mean Community Size"),
    ]

    for prop_col, prop_label in properties_to_plot:
        if prop_col not in stats_df.columns:
            continue

        try:
            fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

            # Ensure data covers all graph_labels in correct order
            plot_df = stats_df.set_index("Graph").reindex(graph_labels).reset_index()
            y_values = plot_df[prop_col].values

            # Use gap-aware plotting
            plot_with_gap_handling(ax, graph_labels, y_values,
                                  gap_info["segments"],
                                  marker='o', linestyle='-', markersize=10,
                                  linewidth=2, color='#2ca02c')

            ax.set_xlabel("Time", fontsize=14, fontweight='bold')
            ax.set_ylabel(prop_label, fontsize=14, fontweight='bold')
            ax.set_title(f"{prop_label} ({algo_name})", fontsize=16, fontweight='bold')

            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))
            plt.yticks(fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            plot_filename = os.path.join(
                save_path, f"communities_{algo_name}_{prop_col}.pdf")
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Plot saved: {plot_filename}")

        except Exception as e:
            print(f"  Warning: Could not plot {prop_label}: {e}")
