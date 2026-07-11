"""
Temporal Network Analysis: Snapshot Stability Module (WITH GAP REPORTING)

This module quantifies how much structure persists between consecutive
snapshots: how similar adjacent graphs are, how many edges/nodes survive, and
how strongly each node keeps the same neighbours over time.

KEY FEATURES:
- Set-based similarity (Jaccard) and persistence between consecutive snapshots.
- Node-level temporal correlation (topological overlap of neighbourhoods).
- Gap-aware: pairs that straddle a detected temporal gap are reported as NaN,
  so the package never compares across missing data.
"""

import logging
import math
import os
import warnings
import igraph as ig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Set
from ._gap_utilities import (
    GapInfo,
    NodeKey,
    detect_temporal_gaps,
    print_gap_report,
    plot_with_gap_handling,
    validate_and_setup_graphs,
    _vertex_keys,
    _active_nodes,
)
from .edge_formation_dissolution import _edge_identity_set

__all__ = [
    "snapshot_similarity",
    "temporal_correlation_coefficient",
]

_METRICS = ["jaccard", "edge_persistence", "node_persistence",
            "temporal_correlation"]

logger = logging.getLogger(__name__)


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _neighbor_sets(graph: ig.Graph) -> Dict[NodeKey, Set[NodeKey]]:
    """
    Return a mapping ``node_key -> set of neighbour keys`` for a graph.

    Built from node identity keys (``_vertex_keys``) so neighbourhoods are
    comparable across snapshots. Edges are treated as undirected connections
    and self-loops are ignored.
    """
    keys = _vertex_keys(graph)
    nb: Dict[NodeKey, Set[NodeKey]] = {k: set() for k in keys}
    for source, target in graph.get_edgelist():
        u, v = keys[source], keys[target]
        if u == v:
            continue
        nb[u].add(v)
        nb[v].add(u)
    return nb


def _temporal_correlation_pair(nb_prev: Dict[NodeKey, Set[NodeKey]],
                               nb_curr: Dict[NodeKey, Set[NodeKey]]) -> float:
    """
    Average topological overlap between two consecutive snapshots.

    For each node, the overlap is
    ``|N_t ∩ N_{t+1}| / sqrt(|N_t| * |N_{t+1}|)`` (0 when the node is isolated
    in either snapshot). The result averages this over nodes that are active in
    at least one of the two snapshots, or NaN if neither snapshot has edges.
    """
    nodes: Set[NodeKey] = set()
    for n, neighbours in nb_prev.items():
        if neighbours:
            nodes.add(n)
    for n, neighbours in nb_curr.items():
        if neighbours:
            nodes.add(n)
    if not nodes:
        return float("nan")

    total = 0.0
    for n in nodes:
        a = nb_prev.get(n, set())
        b = nb_curr.get(n, set())
        if a and b:
            total += len(a & b) / math.sqrt(len(a) * len(b))
    return total / len(nodes)


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def snapshot_similarity(graphs: List[ig.Graph],
                        graph_labels: Optional[List[str]] = None,
                        save_path: Optional[str] = None,
                        report_gaps: bool = False) -> pd.DataFrame:
    """
    Measure structural similarity between consecutive snapshots.

    For each consecutive pair of graphs, computes how similar their edge sets
    are (Jaccard), what fraction of edges and active nodes survive, and the
    node-averaged temporal correlation. Comparison starts from the second
    graph, so the result has one row per consecutive pair.

    **Gap-aware:** consecutive pairs that straddle a detected temporal gap are
    reported as NaN, so structure is never compared across missing data, and
    gap-aware plots break the line there.

    Parameters
    ----------
    graphs : list of igraph.Graph
        List of igraph.Graph objects representing consecutive time points.
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...]).
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    save_path : str, optional
        Directory for saving plots. If None (default), no file is saved.
    report_gaps : bool, optional
        If True, print a temporal gap report to the console
        (default: False).

    Returns
    -------
    pandas.DataFrame
        One row per consecutive pair, with columns:

        - ``Graph``: label of the current (later) snapshot
        - ``jaccard``: |E_t ∩ E_{t+1}| / |E_t ∪ E_{t+1}|
        - ``edge_persistence``: fraction of E_t surviving into E_{t+1}
        - ``node_persistence``: fraction of active nodes surviving
        - ``temporal_correlation``: node-averaged topological overlap

        Undefined ratios (e.g. an empty edge set) and gap-straddling pairs are
        ``NaN``.

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import snapshot_similarity
    >>> g0 = ig.Graph(n=4, edges=[(0, 1), (1, 2)])
    >>> g1 = ig.Graph(n=4, edges=[(1, 2), (2, 3)])
    >>> labels = ["2024-01", "2024-02"]
    >>> sim = snapshot_similarity([g0, g1], graph_labels=labels,
    ...                           report_gaps=False)
    >>> round(float(sim.loc[0, "jaccard"]), 3)
    0.333
    >>> float(sim.loc[0, "edge_persistence"])
    0.5
    """
    graph_labels = validate_and_setup_graphs(graphs, graph_labels, min_length=2)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    gap_info = detect_temporal_gaps(graph_labels)
    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    gap_ends = {g["end_idx"] for g in gap_info.get("gaps", [])}

    def _nan_row(i: int) -> Dict[str, Any]:
        return {"Graph": graph_labels[i],
                **{metric: np.nan for metric in _METRICS}}

    rows = []
    for i in range(1, len(graphs)):
        if i in gap_ends:
            rows.append(_nan_row(i))
            continue

        try:
            e_prev = _edge_identity_set(graphs[i - 1])
            e_curr = _edge_identity_set(graphs[i])
            inter = e_prev & e_curr
            union = e_prev | e_curr

            jaccard = len(inter) / len(union) if union else np.nan
            edge_pers = len(inter) / len(e_prev) if e_prev else np.nan

            act_prev = _active_nodes(e_prev)
            act_curr = _active_nodes(e_curr)
            node_pers = (len(act_prev & act_curr) / len(act_prev)
                         if act_prev else np.nan)

            tc = _temporal_correlation_pair(_neighbor_sets(graphs[i - 1]),
                                            _neighbor_sets(graphs[i]))

            rows.append({"Graph": graph_labels[i],
                         "jaccard": jaccard,
                         "edge_persistence": edge_pers,
                         "node_persistence": node_pers,
                         "temporal_correlation": tc})

        except Exception as e:
            # Emit a NaN row so the output keeps one row per consecutive
            # pair even when a comparison fails.
            warnings.warn(
                f"Error comparing snapshots {i - 1} and {i} "
                f"({graph_labels[i]}): {e}; reporting NaN for this pair")
            rows.append(_nan_row(i))
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["Graph"] + _METRICS)

    if save_path is not None:
        _plot_similarity(df, graph_labels, gap_info, save_path)

    return df


def temporal_correlation_coefficient(
        graphs: List[ig.Graph],
        graph_labels: Optional[List[str]] = None) -> float:
    """
    Average temporal correlation coefficient over the whole sequence.

    Averages the node-averaged topological overlap across all consecutive
    pairs, skipping pairs that straddle a detected temporal gap. A value near 1
    means neighbourhoods are highly stable over time; near 0 means they
    reshuffle from snapshot to snapshot.

    Parameters
    ----------
    graphs : list of igraph.Graph
        List of igraph.Graph objects representing consecutive time points.
    graph_labels : list of str, optional
        Labels for each graph. If not provided, defaults to "Graph 1", etc.

    Returns
    -------
    float
        The sequence-level temporal correlation coefficient, or NaN if no valid
        (non-gap, non-empty) consecutive pair exists.

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import temporal_correlation_coefficient
    >>> g = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3)])
    >>> # Two identical snapshots -> perfectly correlated
    >>> temporal_correlation_coefficient([g, g.copy()])
    1.0
    """
    graph_labels = validate_and_setup_graphs(graphs, graph_labels, min_length=2)

    gap_info = detect_temporal_gaps(graph_labels)
    gap_ends = {g["end_idx"] for g in gap_info.get("gaps", [])}

    values = []
    for i in range(1, len(graphs)):
        if i in gap_ends:
            continue
        try:
            c = _temporal_correlation_pair(_neighbor_sets(graphs[i - 1]),
                                           _neighbor_sets(graphs[i]))
        except Exception as e:
            warnings.warn(
                f"Error comparing snapshots {i - 1} and {i}: {e}; "
                f"skipping this pair")
            continue
        if not math.isnan(c):
            values.append(c)

    if not values:
        return float("nan")
    return float(np.mean(values))


def _plot_similarity(df: pd.DataFrame, graph_labels: List[str],
                     gap_info: GapInfo, save_path: str) -> None:
    """
    Plot each similarity metric over time with gap handling.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`snapshot_similarity`.
    graph_labels : list of str
        Temporal labels for all snapshots.
    gap_info : dict
        Temporal gap information from ``detect_temporal_gaps``.
    save_path : str
        Directory where the metric plots are saved.

    Returns
    -------
    None
    """
    for metric in _METRICS:
        if metric not in df.columns:
            continue
        try:
            plot_df = (df.set_index("Graph")
                         .reindex(graph_labels[1:])
                         .reset_index())
            y_values = plot_df[metric].values
            # Similarity is undefined for the first snapshot (no predecessor).
            y_full = np.concatenate([[np.nan], y_values])

            fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
            plot_with_gap_handling(ax, graph_labels, y_full,
                                   gap_info["segments"],
                                   marker='o', linestyle='-', markersize=10,
                                   linewidth=2, color='#1f77b4')

            ax.set_xlabel("Time", fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(),
                          fontsize=14, fontweight='bold')
            ax.set_title(f"{metric.replace('_', ' ').title()} Over Time",
                         fontsize=16, fontweight='bold')
            plt.yticks(fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_filename = os.path.join(save_path, f"{metric}.pdf")
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info("Plot saved: %s", plot_filename)

        except Exception as e:
            warnings.warn(f"Could not plot {metric}: {e}")
