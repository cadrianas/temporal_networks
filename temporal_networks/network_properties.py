"""
Temporal Network Analysis: Network Properties Module (WITH GAP REPORTING)

This module provides functions for computing structural properties of networks,
including metrics like density, diameter, clustering coefficients, and more.

KEY FEATURES:
- Supports temporal analysis with automatic gap detection
- Reports where data has gaps (seasonal closures, missing measurements, etc.)
- Handles any temporal format (monthly, daily, weekly, etc.)
- Plots correctly show gaps as visual breaks, not false continuity
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional
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

def network_properties(graphs: List,
                      graph_labels: Optional[List[str]] = None,
                      filename: Optional[str] = None,
                      save_path: str = "plots/",
                      visualisation: bool = True,
                      report_gaps: bool = True) -> pd.DataFrame:
    """
    Compute comprehensive network properties for a collection of graphs.

    This function systematically analyzes a collection of networks, extracting
    multiple properties including node/edge counts, density, connectivity measures,
    and clustering coefficients.

    **KEY FEATURE:** Automatically detects and reports temporal gaps in your data,
    showing you where discontinuities occur (seasonal closures, maintenance windows,
    missing measurements, etc.). Plots correctly show gaps as visual breaks.

    Parameters
    ----------
    graphs : list of igraph.Graph
        List of igraph.Graph objects to analyze
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        Supports multiple formats: YYYY-MM, YYYY-MM-DD, YYYY-W##, YYYY-Q#, YYYY
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    filename : str, optional
        If provided, saves numerical results to CSV with this filename
    save_path : str, optional
        Directory path for saving visualizations (default: "plots/")
    visualisation : bool, optional
        If True (default), generates plots for each numerical property
    report_gaps : bool, optional
        If True (default), analyzes and reports temporal gaps to the console

    Returns
    -------
    pandas.DataFrame
        DataFrame containing network properties for each graph,
        with one row per graph and columns for each metric

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import network_properties
    >>>
    >>> # Continuous data (no gaps)
    >>> graphs = [ig.Graph.Barabasi(n=100, m=2) for _ in range(12)]
    >>> labels = [f"2024-{i+1:02d}" for i in range(12)]
    >>> props = network_properties(graphs, graph_labels=labels)
    >>>
    >>> # Gapped data (seasonal operation)
    >>> labels_seasonal = ["2024-03", "2024-04", "2024-05", "2024-06",
    ...                    "2024-07", "2024-08", "2024-11", "2024-12"]
    >>> props = network_properties(graphs[:8], graph_labels=labels_seasonal)
    >>> # Will report: Gap between 2024-08 and 2024-11

    Notes
    -----
    Properties computed include:
    - Basic metrics: node count, edge count, density
    - Connectivity: diameter, average path length, strongly connected components
    - Local structure: clustering coefficient, reciprocity
    - Graph type: directed, weighted, bipartite, etc.

    Gap Detection:
    - Automatically analyzes temporal labels
    - Reports if data has gaps (missing time periods)
    - Shows exactly where gaps occur
    - Plots preserve gaps as visual discontinuities
    """

    # Validate inputs and set up labels
    graph_labels = validate_and_setup_graphs(graphs, graph_labels)

    # Create output directory
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Analyze temporal gaps
    gap_info = detect_temporal_gaps(graph_labels)

    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    # Store results in a list of dictionaries to avoid repeated appends to multiple lists
    results = []

    # Loop over each graph and compute properties
    for i, graph in enumerate(graphs):
        try:
            # Handle properties that might raise exceptions
            try: d = graph.diameter()
            except Exception: d = np.nan

            try: g = graph.girth()
            except Exception: g = np.nan

            try: apl = np.mean(graph.shortest_paths())
            except Exception: apl = np.nan

            try: t = graph.transitivity_undirected()
            except Exception: t = np.nan

            results.append({
                "Graph": graph_labels[i],
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
            })

        except Exception as e:
            print(f"Warning: Error processing graph {graph_labels[i]}: {e}")
            continue

    # Create DataFrame with network properties
    network_data = pd.DataFrame(results)

    # If no graphs were processed successfully, return an empty DataFrame with the correct columns
    if network_data.empty:
        columns = [
            "Graph", "Number of Nodes", "Number of Edges", "Density",
            "Strongly Connected Components", "Girth", "Diameter",
            "Average Path Length", "Mean Degree", "Reciprocity", "Transitivity",
            "Is Bipartite", "Is Connected", "Is DAG", "Is Directed",
            "Is Named", "Is Simple", "Is Weighted", "Has Multiple"
        ]
        network_data = pd.DataFrame(columns=columns)

    # Save to CSV if requested
    if filename:
        try:
            network_data.to_csv(filename, index=False)
            print(f"✓ Network properties saved to {filename}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    # Generate visualizations with gap handling
    if visualisation:
        _plot_properties(network_data, gap_info, save_path)

    return network_data


def _plot_properties(network_data: pd.DataFrame, gap_info: dict, save_path: str):
    """Generate and save plots for network properties over time."""
    properties_to_plot = [
        "Number of Nodes",
        "Number of Edges",
        "Density",
        "Diameter",
        "Average Path Length",
        "Mean Degree",
        "Reciprocity",
        "Transitivity",
    ]

    for prop in properties_to_plot:
        if prop not in network_data.columns:
            continue

        try:
            fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

            y_values = network_data[prop].values
            graph_labels_subset = network_data["Graph"].values

            # Plot with gap handling
            plot_with_gap_handling(ax, list(graph_labels_subset), y_values,
                                  gap_info["segments"],
                                  marker='o', linestyle='-', markersize=10,
                                  linewidth=2, color='#1f77b4')

            ax.set_xlabel("Year - Month", fontsize=14, fontweight='bold')
            ax.set_ylabel(prop, fontsize=14, fontweight='bold')
            ax.set_title(f"{prop} Over Time", fontsize=16, fontweight='bold')

            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))
            plt.yticks(fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            plot_filename = os.path.join(save_path, f"{prop}.pdf")
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Plot saved: {plot_filename}")

        except Exception as e:
            print(f"Warning: Could not plot {prop}: {e}")
