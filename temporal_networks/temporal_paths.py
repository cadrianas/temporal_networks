"""
Temporal Network Analysis: Time-respecting Path Metrics (GAP-AWARE)

This module computes path metrics that respect the ordering of snapshots:
a path from source s to target t is valid only if each hop uses an edge
that appears in a LATER snapshot than the previous hop (or the same one,
when waiting is permitted).

KEY FEATURES:
- Forward BFS per source over the snapshot sequence: O(T * E) per source.
- ``allow_wait=True`` (default): a reached node can wait at a position and
  still use edges in later snapshots.
- ``cross_gaps=False`` (default, the differentiator): time-respecting paths
  cannot cross a detected temporal gap. A closure is not assumed to be
  transparent to transmission. Pass ``cross_gaps=True`` for standard
  contact-sequence behaviour.
- Output uses ``NaN`` for unreachable entries in DataFrames and ``inf`` in
  distance tables so downstream arithmetic works naturally (1/inf = 0).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Set, Tuple
from ._gap_utilities import (
    detect_temporal_gaps,
    print_gap_report,
    validate_and_setup_graphs,
    _vertex_keys,
)

__all__ = [
    "temporal_reachability",
    "temporal_distances",
    "temporal_closeness",
    "temporal_efficiency",
]

_REACH_COLS = ["source", "target", "reachable", "first_arrival_idx"]
_DIST_COLS = ["source", "target", "latency"]
_CLOSE_COLS = ["node", "closeness"]


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _union_nodes(graphs: List) -> List:
    """
    Return all node identity keys present in at least one snapshot.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Temporal snapshots.

    Returns
    -------
    list
        Sorted list of unique node keys across all snapshots.
    """
    nodes: Set = set()
    for g in graphs:
        nodes.update(_vertex_keys(g))
    return sorted(nodes, key=str)


def _edges_keyed(graph) -> List[Tuple]:
    """
    Return edges as ``(key_u, key_v)`` pairs using vertex identity keys.

    For undirected graphs both directions are returned; self-loops are skipped.

    Parameters
    ----------
    graph : igraph.Graph
        A single snapshot.

    Returns
    -------
    list of tuple
        One ``(u, v)`` pair per directed traversal direction.
    """
    keys = _vertex_keys(graph)
    directed = graph.is_directed()
    edges: List[Tuple] = []
    for src, tgt in graph.get_edgelist():
        u, v = keys[src], keys[tgt]
        if u == v:
            continue
        edges.append((u, v))
        if not directed:
            edges.append((v, u))
    return edges


def _bfs_from_source(
    graphs: List,
    source,
    gap_ends: Set[int],
    allow_wait: bool,
    cross_gaps: bool,
) -> Dict:
    """
    Forward BFS from ``source`` over the snapshot sequence.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Temporal snapshots in order.
    source : node key
        Starting node (reached at step 0).
    gap_ends : set of int
        Snapshot indices that start a new segment after a detected gap.
    allow_wait : bool
        If True, a reached node can use edges in later snapshots.
        If False, a reached node can only use edges in the snapshot
        immediately after it was reached.
    cross_gaps : bool
        If False, reachability does not propagate across gap boundaries.

    Returns
    -------
    dict
        ``{node_key: first_arrival_step}`` for every reached node.
        ``source`` always maps to 0.
    """
    first_arrival: Dict = {source: 0}
    # reachable tracks which nodes can still propagate in the current segment.
    reachable: Set = {source}

    for t, graph in enumerate(graphs):
        if not cross_gaps and t in gap_ends:
            # Hard barrier: nothing from the previous segment carries over.
            reachable = set()

        if allow_wait:
            propagators: Set = reachable
        else:
            # Strict: only nodes that arrived at exactly this step may move.
            propagators = {n for n in reachable
                           if first_arrival.get(n) == t}

        if not propagators:
            continue

        for u, v in _edges_keyed(graph):
            if u in propagators and v not in first_arrival:
                first_arrival[v] = t + 1
                reachable.add(v)

    return first_arrival


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

def temporal_reachability(
    graphs: List,
    graph_labels: Optional[List[str]] = None,
    allow_wait: bool = True,
    cross_gaps: bool = False,
) -> pd.DataFrame:
    """
    Source–target reachability and first-arrival index for every node pair.

    Uses a forward BFS over the snapshot sequence: a node is reachable from
    ``source`` if there exists a time-respecting path — a sequence of hops
    each using an edge that appears in a later snapshot than the previous hop.

    **Gap-aware (default):** with ``cross_gaps=False``, reachability does not
    propagate across detected temporal gaps. A closure in the data is not
    assumed to be transparent to transmission. Pass ``cross_gaps=True`` for
    standard contact-sequence behaviour.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Temporal snapshots in chronological order.
    graph_labels : list of str, optional
        Labels for each snapshot. If None, defaults to "Graph 1", etc.
    allow_wait : bool, optional
        If True (default), a reached node may wait and use edges in later
        snapshots. If False, paths are strict: each node must move at the
        next available snapshot after it is reached.
    cross_gaps : bool, optional
        If False (default), time-respecting paths are blocked at detected
        temporal gaps. If True, gaps are ignored and the sequence is treated
        as contiguous.

    Returns
    -------
    pandas.DataFrame
        One row per ordered ``(source, target)`` pair — including self-pairs
        — with columns:

        - ``source``, ``target``: node identity keys
        - ``reachable``: ``True`` if a time-respecting path exists
        - ``first_arrival_idx``: snapshot step of first arrival, or ``NaN``

        Self-pairs have ``first_arrival_idx = 0``. Returns an empty DataFrame
        with the correct columns on empty or all-failing input.

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import temporal_reachability
    >>> g0 = ig.Graph(n=3, edges=[(0, 1)])
    >>> g1 = ig.Graph(n=3, edges=[(1, 2)])
    >>> df = temporal_reachability([g0, g1], graph_labels=["t0", "t1"])
    >>> bool(df[(df.source == 0) & (df.target == 2)]["reachable"].iloc[0])
    True
    >>> int(df[(df.source == 0) & (df.target == 2)]["first_arrival_idx"].iloc[0])
    2
    """
    graph_labels = validate_and_setup_graphs(graphs, graph_labels,
                                             min_length=1)
    gap_info = detect_temporal_gaps(graph_labels)
    gap_ends = {g["end_idx"] for g in gap_info.get("gaps", [])}

    try:
        all_nodes = _union_nodes(graphs)
    except Exception as e:
        print(f"Warning: Could not collect node set: {e}")
        return pd.DataFrame(columns=_REACH_COLS)

    rows = []
    for source in all_nodes:
        try:
            fa = _bfs_from_source(graphs, source, gap_ends,
                                   allow_wait, cross_gaps)
            for target in all_nodes:
                reached = target in fa
                rows.append({
                    "source": source,
                    "target": target,
                    "reachable": reached,
                    "first_arrival_idx": (
                        float(fa[target]) if reached else float("nan")),
                })
        except Exception as e:
            print(f"Warning: Error computing reachability from "
                  f"{source}: {e}")
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=_REACH_COLS)
    return (df.sort_values(["source", "target"], key=lambda c: c.map(str))
              .reset_index(drop=True))


def temporal_distances(
    graphs: List,
    graph_labels: Optional[List[str]] = None,
    allow_wait: bool = True,
    cross_gaps: bool = False,
) -> pd.DataFrame:
    """
    Time-respecting latency (in snapshot steps) for every source–target pair.

    Latency is the snapshot step at which the target is first reached when
    BFS starts at step 0. Unreachable pairs have latency ``inf``; self-pairs
    have latency ``0``.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Temporal snapshots in chronological order.
    graph_labels : list of str, optional
        Labels for each snapshot. If None, defaults to "Graph 1", etc.
    allow_wait : bool, optional
        If True (default), waiting at intermediate nodes is permitted.
    cross_gaps : bool, optional
        If False (default), paths are blocked at detected temporal gaps.

    Returns
    -------
    pandas.DataFrame
        One row per ``(source, target)`` pair with columns:

        - ``source``, ``target``: node identity keys
        - ``latency``: float snapshot steps (``0`` for self, ``inf`` if
          unreachable)

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import temporal_distances
    >>> g0 = ig.Graph(n=3, edges=[(0, 1)])
    >>> g1 = ig.Graph(n=3, edges=[(1, 2)])
    >>> dist = temporal_distances([g0, g1], graph_labels=["t0", "t1"])
    >>> float(dist[(dist.source == 0) & (dist.target == 2)]["latency"].iloc[0])
    2.0
    """
    reach = temporal_reachability(graphs, graph_labels, allow_wait, cross_gaps)
    if reach.empty:
        return pd.DataFrame(columns=_DIST_COLS)

    result = reach[["source", "target"]].copy()
    result["latency"] = reach.apply(
        lambda r: float(r["first_arrival_idx"]) if r["reachable"]
        else float("inf"),
        axis=1,
    )
    return result.reset_index(drop=True)


def temporal_closeness(
    graphs: List,
    graph_labels: Optional[List[str]] = None,
    cross_gaps: bool = False,
    save_path: Optional[str] = None,
    report_gaps: bool = True,
) -> pd.DataFrame:
    """
    Harmonic temporal closeness per node.

    For each node ``v``, closeness is the normalised sum of inverse latencies
    to all other nodes::

        closeness(v) = (1 / (n-1)) * sum_{u != v} 1 / latency(v, u)

    where ``1 / inf = 0`` (unreachable targets contribute zero). Values range
    from 0 (no outgoing time-respecting paths) to 1 (all nodes reachable in
    one snapshot step).

    Parameters
    ----------
    graphs : list of igraph.Graph
        Temporal snapshots in chronological order.
    graph_labels : list of str, optional
        Labels for each snapshot. If None, defaults to "Graph 1", etc.
    cross_gaps : bool, optional
        If False (default), paths are blocked at detected temporal gaps.
    save_path : str, optional
        Directory for saving the closeness bar-chart PDF. If None (default),
        no file is saved.
    report_gaps : bool, optional
        If True (default), prints a temporal gap report to the console.

    Returns
    -------
    pandas.DataFrame
        One row per node with columns ``node`` and ``closeness``.
        Sorted descending by closeness.

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import temporal_closeness
    >>> g0 = ig.Graph(n=3, edges=[(0, 1)])
    >>> g1 = ig.Graph(n=3, edges=[(1, 2)])
    >>> cl = temporal_closeness([g0, g1], graph_labels=["t0", "t1"],
    ...                          report_gaps=False)
    >>> float(cl[cl["node"] == 0]["closeness"].iloc[0]) > 0
    True
    """
    graph_labels = validate_and_setup_graphs(graphs, graph_labels,
                                             min_length=1)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    gap_info = detect_temporal_gaps(graph_labels)
    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    dist = temporal_distances(graphs, graph_labels,
                               allow_wait=True, cross_gaps=cross_gaps)
    if dist.empty:
        return pd.DataFrame(columns=_CLOSE_COLS)

    all_nodes = sorted(dist["source"].unique(), key=str)
    n = len(all_nodes)

    rows = []
    for node in all_nodes:
        out = dist[(dist["source"] == node) & (dist["target"] != node)]
        finite = out[out["latency"] < float("inf")]
        closeness = (
            float((1.0 / finite["latency"]).sum()) / (n - 1)
            if n > 1 and not finite.empty
            else 0.0
        )
        rows.append({"node": node, "closeness": closeness})

    df = (pd.DataFrame(rows)
            .sort_values("closeness", ascending=False)
            .reset_index(drop=True))

    if save_path is not None:
        _plot_closeness(df, save_path)

    return df


def temporal_efficiency(
    graphs: List,
    graph_labels: Optional[List[str]] = None,
    cross_gaps: bool = False,
) -> float:
    """
    Global temporal efficiency of the snapshot sequence.

    Defined as the mean of ``1 / latency`` over all ordered source–target
    pairs with source ≠ target, where ``1 / inf = 0``. Ranges from 0 (no
    time-respecting paths) to 1 (every pair reachable in one step).

    Parameters
    ----------
    graphs : list of igraph.Graph
        Temporal snapshots in chronological order.
    graph_labels : list of str, optional
        Labels for each snapshot. If None, defaults to "Graph 1", etc.
    cross_gaps : bool, optional
        If False (default), paths are blocked at detected temporal gaps.

    Returns
    -------
    float
        Global temporal efficiency, or ``NaN`` if there are no non-self
        pairs.

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import temporal_efficiency
    >>> g0 = ig.Graph(n=2, edges=[(0, 1)])
    >>> eff = temporal_efficiency([g0], graph_labels=["t0"])
    >>> float(eff) > 0
    True
    """
    graph_labels = validate_and_setup_graphs(graphs, graph_labels,
                                             min_length=1)
    dist = temporal_distances(graphs, graph_labels,
                               allow_wait=True, cross_gaps=cross_gaps)
    if dist.empty:
        return float("nan")

    others = dist[dist["source"] != dist["target"]]
    if others.empty:
        return float("nan")

    inv = others["latency"].apply(
        lambda x: 0.0 if x == float("inf") else 1.0 / x)
    return float(inv.mean())


def _plot_closeness(df: pd.DataFrame, save_path: str) -> None:
    """
    Save a horizontal bar chart of temporal closeness values.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`temporal_closeness`, sorted descending.
    save_path : str
        Directory where the PDF is saved.

    Returns
    -------
    None
    """
    try:
        nodes = [str(n) for n in df["node"]]
        values = df["closeness"].values

        fig, ax = plt.subplots(figsize=(8, max(4, len(nodes) * 0.4)), dpi=100)
        y_pos = np.arange(len(nodes))
        ax.barh(y_pos, values, color='#1f77b4', edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(nodes, fontsize=10)
        ax.set_xlabel("Temporal Closeness", fontsize=13, fontweight='bold')
        ax.set_title("Temporal Closeness per Node",
                     fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(values.max() * 1.1, 0.01))
        plt.tight_layout()

        path = os.path.join(save_path, "temporal_closeness.pdf")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Plot saved: {path}")

    except Exception as e:
        print(f"Warning: Could not plot temporal closeness: {e}")
