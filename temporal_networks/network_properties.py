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
    format_large_numbers
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

    # Validate inputs
    if not graphs:
        raise ValueError("graphs list cannot be empty")

    # Set up graph labels
    if graph_labels is None:
        graph_labels = [f"Graph {i+1}" for i in range(len(graphs))]
    elif len(graph_labels) != len(graphs):
        raise ValueError(f"graph_labels length ({len(graph_labels)}) must match graphs length ({len(graphs)})")

    # Create output directory
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Analyze temporal gaps
    gap_info = detect_temporal_gaps(graph_labels)

    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    # Initialize lists to store properties
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

    # Loop over each graph and compute properties
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
                avg_path_length.append(np.mean(graph.shortest_paths()))
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

        except Exception as e:
            print(f"Warning: Error processing graph {graph_labels[len(num_vertices)]}: {e}")
            continue

    # Create DataFrame with network properties
    network_data = pd.DataFrame({
        "Graph": graph_labels[:len(num_vertices)],
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
