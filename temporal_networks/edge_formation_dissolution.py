"""
Temporal Network Analysis: Edge Dynamics Module (WITH GAP REPORTING)

This module provides functions for computing and visualizing edge formation
and dissolution patterns across temporal networks. Properly handles temporal
gaps in the data.

KEY FEATURES:
- Automatically detects and reports temporal gaps
- Plots correctly show gaps as visual breaks
- Analyzes edge dynamics (routes that form and dissolve)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Dict
from ._gap_utilities import (
    detect_temporal_gaps,
    print_gap_report,
    plot_with_gap_handling,
    format_large_numbers,
    validate_and_setup_graphs
)

__all__ = [
    "compute_edge_dynamics",
    "edge_formation",
    "edge_dissolution",
    "plot_edge_dynamics",
]


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def _edge_identity_set(graph) -> set:
    """
    Return a graph's edges as a set of identity keys.

    Edges are keyed by vertex ``name`` (or ``label``) when those attributes are
    present, so edges are compared by node identity across snapshots rather than
    by position. When no such attribute exists, vertex indices are used as a
    fallback (which assumes consistent node ordering across snapshots).
    Undirected edges are normalised so that ``(u, v)`` and ``(v, u)`` compare
    equal.

    Parameters
    ----------
    graph : igraph.Graph
        Graph whose edges to extract.

    Returns
    -------
    set
        Set of ``(source_key, target_key)`` tuples.
    """
    attrs = graph.vs.attributes()
    if "name" in attrs:
        keys = graph.vs["name"]
    elif "label" in attrs:
        keys = graph.vs["label"]
    else:
        keys = None

    directed = graph.is_directed()
    edge_set = set()
    for source, target in graph.get_edgelist():
        if keys is not None:
            u, v = keys[source], keys[target]
        else:
            u, v = source, target
        if not directed:
            u, v = tuple(sorted((u, v)))
        edge_set.add((u, v))
    return edge_set


def compute_edge_dynamics(graphs: List,
                         graph_labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute edge formation and dissolution between consecutive graphs.

    Analyzes how the edge structure changes from one temporal snapshot to
    the next, computing the number of edges that appear (formed) and disappear
    (dissolved) at each time step.

    Parameters
    ----------
    graphs : list of igraph.Graph
        List of igraph.Graph objects representing consecutive time points
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        If not provided, defaults to "Graph 1", "Graph 2", etc.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - Graph: Label for the current time point
        - Edges_Formed: Number of new edges created
        - Edges_Dissolved: Number of edges removed
        - Edges_Formed_Percent: Percentage change relative to previous graph
        - Edges_Dissolved_Percent: Percentage change relative to previous graph

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import compute_edge_dynamics
    >>> graphs = [ig.Graph.Barabasi(n=50, m=2) for _ in range(12)]
    >>> labels = [f"2019-{i+1:02d}" for i in range(12)]
    >>> dynamics = compute_edge_dynamics(graphs, graph_labels=labels)

    Notes
    -----
    Comparison starts from the second graph. The first row corresponds to
    differences between Graph 0 and Graph 1.

    Edges are matched by node ``name`` (or ``label``) when those vertex
    attributes are present, so snapshots whose nodes are stored in a different
    order are still compared correctly. If no such attribute exists, vertex
    indices are used, which assumes a consistent node ordering across snapshots.
    Undirected edges are treated as unordered pairs.
    """

    # Validate inputs and set up labels
    graph_labels = validate_and_setup_graphs(graphs, graph_labels, min_length=2)

    dynamics_data = []

    # Compare consecutive graphs
    for i in range(1, len(graphs)):
        g_prev = graphs[i-1]
        g_curr = graphs[i]

        try:
            # Edge sets keyed by node identity (name/label) when available, so
            # snapshots with different node orderings are compared correctly.
            prev_edges = _edge_identity_set(g_prev)
            curr_edges = _edge_identity_set(g_curr)

            # Compute edges formed and dissolved
            edges_formed = curr_edges - prev_edges
            edges_dissolved = prev_edges - curr_edges

            num_formed = len(edges_formed)
            num_dissolved = len(edges_dissolved)

            # Compute percentages (relative to previous graph edge count)
            prev_edge_count = len(prev_edges)
            formed_percent = (
                (num_formed / prev_edge_count * 100)
                if prev_edge_count > 0 else 0
            )
            dissolved_percent = (
                (num_dissolved / prev_edge_count * 100)
                if prev_edge_count > 0 else 0
            )

            dynamics_data.append({
                "Graph": graph_labels[i],
                "Edges_Formed": num_formed,
                "Edges_Dissolved": num_dissolved,
                "Edges_Formed_Percent": formed_percent,
                "Edges_Dissolved_Percent": dissolved_percent,
            })

        except Exception as e:
            print(f"Warning: Error comparing graphs {i-1} and {i}: {e}")
            continue

    dynamics_df = pd.DataFrame(dynamics_data)
    return dynamics_df


def plot_edge_dynamics(dynamics_df: pd.DataFrame,
                      graph_labels: List[str],
                      gap_info: Dict,
                      metric: str = "Edges_Formed",
                      output_file: Optional[str] = None,
                      save_path: Optional[str] = None) -> None:
    """
    Plot edge formation/dissolution dynamics over time with gap handling.

    Creates a line plot showing how edges are formed or dissolved at each
    time step. Properly shows gaps in temporal data as visual breaks.

    Parameters
    ----------
    dynamics_df : pd.DataFrame
        DataFrame from compute_edge_dynamics()
    graph_labels : list of str
        Temporal labels
    gap_info : dict
        Gap detection information
    metric : str, optional
        Metric to plot: "Edges_Formed", "Edges_Dissolved", etc.
        (default: "Edges_Formed")
    output_file : str, optional
        Filename for saving the plot. If None, uses metric name.
        (default: None → "{metric}.pdf")
    save_path : str, optional
        Directory for saving plots. If None (default), no file is saved.

    Returns
    -------
    None
        Saves plot to file

    Notes
    -----
    The plot uses gap-aware plotting to preserve temporal gaps.
    If you have data for Jan, Mar (Feb missing), the plot will show that
    gap visually instead of drawing a false line.

    Examples
    --------
    >>> import random
    >>> import igraph as ig
    >>> from temporal_networks import (compute_edge_dynamics, plot_edge_dynamics,
    ...                                detect_temporal_gaps)
    >>> ig.set_random_number_generator(random.Random(42))
    >>> graphs = [ig.Graph.Barabasi(n=30, m=2) for _ in range(6)]
    >>> labels = [f"2024-{i + 1:02d}" for i in range(6)]
    >>> dynamics = compute_edge_dynamics(graphs, graph_labels=labels)
    >>> gap_info = detect_temporal_gaps(labels)
    >>> plot_edge_dynamics(dynamics, labels, gap_info, metric="Edges_Formed")
    """

    if metric not in dynamics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe. "
                        f"Available: {list(dynamics_df.columns)}")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    try:
        fig, ax = plt.subplots(figsize=(14, 7), dpi=100)

        # Reindex dynamics_df to match graph_labels starting from the second label
        # Dynamics data only exists for labels[1:]
        plot_df = dynamics_df.set_index("Graph").reindex(graph_labels[1:]).reset_index()
        y_values = plot_df[metric].values

        # Adjust gap_segments for the fact that dynamics_df has 1 fewer
        # point than graph_labels
        # Actually, it's better to keep graph_labels and put None/NaN
        # for the first point
        y_values_full = np.concatenate([[np.nan], y_values])

        # Use gap-aware plotting
        plot_with_gap_handling(ax, graph_labels, y_values_full,
                              gap_info["segments"],
                              marker='o', linestyle='-', markersize=12,
                              linewidth=3, color='#1f77b4')

        ax.set_xlabel("Time", fontsize=14, fontweight='bold')
        ax.set_ylabel("Number of Edges", fontsize=14, fontweight='bold')
        ax.set_title(f"{metric.replace('_', ' ')} Over Time",
                     fontsize=16, fontweight='bold')

        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))
        plt.yticks(fontsize=12, fontweight='bold')

        # Grid and layout
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        if save_path is not None:
            if output_file is None:
                output_file = f"{metric}.pdf"

            plot_path = os.path.join(save_path, output_file)
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {plot_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error creating plot: {e}")


def edge_formation(graphs: List,
                  graph_labels: Optional[List[str]] = None,
                  save_path: Optional[str] = None,
                  report_gaps: bool = True) -> pd.DataFrame:
    """
    Analyze and plot edge formation over time.

    Convenience function combining edge dynamics computation and visualization
    for edge formation specifically. Automatically detects and reports temporal
    gaps.

    Parameters
    ----------
    graphs : list of igraph.Graph
        List of igraph.Graph objects
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    save_path : str, optional
        Directory for saving plots. If None (default), no file is saved.
    report_gaps : bool, optional
        If True (default), analyzes and reports temporal gaps to the console

    Returns
    -------
    pandas.DataFrame
        DataFrame with edge dynamics

    Examples
    --------
    >>> import random
    >>> import igraph as ig
    >>> from temporal_networks import edge_formation
    >>> ig.set_random_number_generator(random.Random(42))
    >>> graphs = [ig.Graph.Barabasi(n=30, m=2) for _ in range(6)]
    >>> labels = [f"2024-{i + 1:02d}" for i in range(6)]
    >>> dynamics = edge_formation(graphs, graph_labels=labels, report_gaps=False)
    Computing edge formation...
    Plotting edge formation...
    >>> dynamics.shape
    (5, 5)
    """

    # Validate inputs and set up labels
    graph_labels = validate_and_setup_graphs(graphs, graph_labels, min_length=2)

    # Analyze temporal gaps
    gap_info = detect_temporal_gaps(graph_labels)

    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    print("Computing edge formation...")
    dynamics_df = compute_edge_dynamics(graphs, graph_labels=graph_labels)

    print("Plotting edge formation...")
    plot_edge_dynamics(dynamics_df, graph_labels, gap_info,
                      metric="Edges_Formed",
                      output_file="edges_formed.pdf", save_path=save_path)

    return dynamics_df


def edge_dissolution(graphs: List,
                    graph_labels: Optional[List[str]] = None,
                    save_path: Optional[str] = None,
                    report_gaps: bool = True) -> pd.DataFrame:
    """
    Analyze and plot edge dissolution over time.

    Convenience function combining edge dynamics computation and visualization
    for edge dissolution specifically. Automatically detects and reports temporal
    gaps.

    Parameters
    ----------
    graphs : list of igraph.Graph
        List of igraph.Graph objects
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    save_path : str, optional
        Directory for saving plots. If None (default), no file is saved.
    report_gaps : bool, optional
        If True (default), analyzes and reports temporal gaps to the console

    Returns
    -------
    pandas.DataFrame
        DataFrame with edge dynamics

    Examples
    --------
    >>> import random
    >>> import igraph as ig
    >>> from temporal_networks import edge_dissolution
    >>> ig.set_random_number_generator(random.Random(42))
    >>> graphs = [ig.Graph.Barabasi(n=30, m=2) for _ in range(6)]
    >>> labels = [f"2024-{i + 1:02d}" for i in range(6)]
    >>> dynamics = edge_dissolution(graphs, graph_labels=labels, report_gaps=False)
    Computing edge dissolution...
    Plotting edge dissolution...
    >>> dynamics.shape
    (5, 5)
    """

    # Validate inputs and set up labels
    graph_labels = validate_and_setup_graphs(graphs, graph_labels, min_length=2)

    # Analyze temporal gaps
    gap_info = detect_temporal_gaps(graph_labels)

    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    print("Computing edge dissolution...")
    dynamics_df = compute_edge_dynamics(graphs, graph_labels=graph_labels)

    print("Plotting edge dissolution...")
    plot_edge_dynamics(dynamics_df, graph_labels, gap_info,
                      metric="Edges_Dissolved",
                      output_file="edges_dissolved.pdf", save_path=save_path)

    return dynamics_df
