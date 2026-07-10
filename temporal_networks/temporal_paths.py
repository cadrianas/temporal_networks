"""
Temporal Network Analysis: Time-respecting Path Metrics (GAP-AWARE)

This module computes path metrics that respect the ordering of snapshots:
a path from source s to target t is valid only if consecutive hops use
edges from strictly later snapshots. Each hop takes one time step, so a
path traverses AT MOST ONE edge per snapshot — edges within a snapshot are
treated as simultaneous contacts and cannot be chained.

KEY FEATURES:
- Forward BFS per source over the snapshot sequence: O(T * E) per source.
- ``allow_wait=True`` (default): a reached node can wait at a position and
  still use edges in later snapshots.
- ``cross_gaps=False`` (default, the differentiator): time-respecting paths
  cannot cross a detected temporal gap — a closure is not assumed to be
  transparent to transmission. Paths confined to a single continuous
  segment remain valid, so post-gap segments are still analysed (each
  source starts fresh paths at every segment). Pass ``cross_gaps=True``
  for standard contact-sequence behaviour.
- Output uses ``NaN`` for unreachable entries in DataFrames and ``inf`` in
  distance tables so downstream arithmetic works naturally (1/inf = 0).
"""

import logging
import os
import warnings

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
    "temporal_betweenness",
]

_REACH_COLS = ["source", "target", "reachable", "first_arrival_idx"]
_DIST_COLS = ["source", "target", "latency"]
_CLOSE_COLS = ["node", "closeness"]
_BETW_COLS = ["node", "betweenness"]

logger = logging.getLogger(__name__)


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

    With ``cross_gaps=False``, a time-respecting path is valid only if it
    lies entirely within one continuous segment (no hop, and no waiting,
    across a detected gap). The source starts fresh paths at the beginning
    of every segment, so nodes connected to it purely by post-gap edges are
    still reachable; only paths that would *cross* a gap are blocked.

    Edges within one snapshot are simultaneous: a path traverses at most
    one edge per snapshot, so a node reached during snapshot ``t`` can only
    move again from snapshot ``t + 1`` onwards. New arrivals are therefore
    collected during each snapshot's sweep and merged afterwards, which
    makes the result independent of edge storage order.

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
        If False, paths are confined to single continuous segments.

    Returns
    -------
    dict
        ``{node_key: first_arrival_step}`` for every reached node, where
        the arrival is the earliest over all valid paths (all segments).
        ``source`` always maps to 0.
    """
    first_arrival: Dict = {source: 0}
    # Arrival steps of the paths confined to the current segment. Reset at
    # every gap boundary so no path (or wait) crosses a gap.
    segment_arrival: Dict = {source: 0}

    for t, graph in enumerate(graphs):
        if not cross_gaps and t in gap_ends:
            # New segment: paths from earlier segments cannot cross the
            # gap, but the source starts fresh paths within this segment.
            segment_arrival = {source: t}

        # Only nodes reached BEFORE this snapshot may move, so arrivals
        # are buffered and merged after the sweep.
        new_arrivals: Dict = {}
        for u, v in _edges_keyed(graph):
            if v in segment_arrival or v in new_arrivals:
                continue
            if allow_wait:
                can_move = u in segment_arrival
            else:
                # Strict: u may only move at the step right after arrival.
                can_move = segment_arrival.get(u) == t
            if can_move:
                new_arrivals[v] = t + 1
                if v not in first_arrival:
                    first_arrival[v] = t + 1
        segment_arrival.update(new_arrivals)

    return first_arrival


def _foremost_paths_from_source(
    graphs: List,
    source,
) -> Tuple[List, Dict, Dict]:
    """
    Forward pass of temporal Brandes from ``source`` (foremost paths).

    Sweeps a *contiguous* snapshot sequence once, recording — for the
    earliest-arrival (foremost) time-respecting paths only — how many such
    paths reach each node and which immediate predecessors lie on them.
    Waiting at a node is permitted (``allow_wait`` is implicitly True),
    matching :func:`temporal_closeness` / :func:`temporal_efficiency`.

    Gap handling lives in the caller: :func:`temporal_betweenness` runs this
    pass independently on each continuous segment, so no path crosses a gap.

    Parameters
    ----------
    graphs : list of igraph.Graph
        A contiguous run of temporal snapshots in order.
    source : node key
        Starting node (reached at step 0).

    Returns
    -------
    order : list
        Node keys in the order their foremost arrival was finalised
        (non-decreasing arrival step). Used to drive the backward pass.
    preds : dict
        ``{node: [predecessor keys]}`` on foremost paths.
    sigma : dict
        ``{node: number of foremost paths from source}`` as floats.
    """
    arrival: Dict = {source: 0}
    sigma: Dict = {source: 1.0}
    preds: Dict = {source: []}
    order: List = [source]

    for t, graph in enumerate(graphs):
        # Same one-hop-per-snapshot rule as _bfs_from_source: only nodes
        # reached before this snapshot may move. Buffering the layer's
        # arrivals also finalises sigma[u] before u is ever used as a
        # predecessor, making path counts independent of edge order.
        new_arrivals: Dict = {}
        for u, v in _edges_keyed(graph):
            if u not in arrival:
                continue
            if v in arrival:
                continue
            if v not in new_arrivals:
                new_arrivals[v] = t + 1
                sigma[v] = sigma[u]
                preds[v] = [u]
                order.append(v)
            else:
                # Another foremost path to v in the same arrival layer.
                sigma[v] += sigma[u]
                preds[v].append(u)
        arrival.update(new_arrivals)

    return order, preds, sigma


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
    each using an edge that appears in a later snapshot than the previous
    hop. Edges within one snapshot are simultaneous contacts, so a path
    traverses at most one edge per snapshot.

    **Gap-aware (default):** with ``cross_gaps=False``, a time-respecting
    path is valid only if it lies entirely within one continuous segment —
    no hop, and no waiting, across a detected temporal gap. A closure in the
    data is not assumed to be transparent to transmission. Paths within the
    post-gap segments still count (the source starts fresh paths in every
    segment), and ``first_arrival_idx`` is the earliest arrival over all
    segments. Pass ``cross_gaps=True`` for standard contact-sequence
    behaviour.

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
        If False (default), time-respecting paths cannot cross detected
        temporal gaps; paths within a single continuous segment remain
        valid. If True, gaps are ignored and the sequence is treated as
        contiguous.

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
        warnings.warn(f"Could not collect node set: {e}")
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
            warnings.warn(f"Error computing reachability from "
                          f"{source}: {e}; skipping this source")
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
        If False (default), paths cannot cross detected temporal gaps;
        paths within a single continuous segment remain valid.

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
    result["latency"] = np.where(reach["reachable"].to_numpy(),
                                 reach["first_arrival_idx"].to_numpy(),
                                 np.inf)
    return result.reset_index(drop=True)


def temporal_closeness(
    graphs: List,
    graph_labels: Optional[List[str]] = None,
    cross_gaps: bool = False,
    save_path: Optional[str] = None,
    report_gaps: bool = False,
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
        If False (default), paths cannot cross detected temporal gaps;
        paths within a single continuous segment remain valid.
    save_path : str, optional
        Directory for saving the closeness bar-chart PDF. If None (default),
        no file is saved.
    report_gaps : bool, optional
        If True, print a temporal gap report to the console
        (default: False).

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

    # Vectorised: one pass over the n^2 pair table instead of one boolean
    # mask per node. 1/inf == 0.0, so unreachable targets contribute zero.
    if n > 1:
        others = dist[dist["source"] != dist["target"]]
        inv = 1.0 / others["latency"].to_numpy()
        sums = (pd.Series(inv, index=others["source"].to_numpy())
                  .groupby(level=0).sum())
        closeness = sums.reindex(all_nodes, fill_value=0.0) / (n - 1)
    else:
        closeness = pd.Series(0.0, index=all_nodes)

    df = (closeness.rename("closeness").rename_axis("node").reset_index()
            .sort_values("closeness", ascending=False, kind="stable")
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
        If False (default), paths cannot cross detected temporal gaps;
        paths within a single continuous segment remain valid.

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

    # 1/inf == 0.0, so unreachable pairs contribute zero.
    inv = 1.0 / others["latency"].to_numpy()
    return float(inv.mean())


def temporal_betweenness(
    graphs: List,
    graph_labels: Optional[List[str]] = None,
    cross_gaps: bool = False,
    normalized: bool = True,
    save_path: Optional[str] = None,
    report_gaps: bool = False,
) -> pd.DataFrame:
    """
    Per-node temporal betweenness over time-respecting paths.

    For every ordered source–target pair, the fraction of foremost
    (earliest-arrival) time-respecting paths passing through each
    intermediate node is accumulated, using a temporal adaptation of
    Brandes' algorithm. A high value means a node frequently lies on the
    quickest time-respecting routes between other nodes — a temporal
    bottleneck or broker.

    **Gap-aware (default):** with ``cross_gaps=False``, foremost paths
    cannot cross a detected temporal gap, so a data closure never inflates
    a node's brokerage. Brokerage on paths confined to a single continuous
    segment still counts (each source–target pair is counted once, at its
    earliest arrival over all segments). Pass ``cross_gaps=True`` to treat
    the sequence as contiguous.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Temporal snapshots in chronological order.
    graph_labels : list of str, optional
        Labels for each snapshot. If None, defaults to "Graph 1", etc.
    cross_gaps : bool, optional
        If False (default), paths cannot cross detected temporal gaps;
        paths within a single continuous segment remain valid.
    normalized : bool, optional
        If True (default), divide by ``(n - 1) * (n - 2)`` (the number of
        ordered pairs not involving the node), giving values in ``[0, 1]``.
        If False, report raw pair-dependency sums.
    save_path : str, optional
        Directory for saving the betweenness bar-chart PDF. If None
        (default), no file is saved.
    report_gaps : bool, optional
        If True, print a temporal gap report to the console
        (default: False).

    Returns
    -------
    pandas.DataFrame
        One row per node with columns ``node`` and ``betweenness``,
        sorted descending by betweenness. Returns an empty DataFrame with
        the correct columns on empty input.

    Notes
    -----
    Temporal paths are inherently directed by time, so each ordered
    ``(source, target)`` pair is counted once (no factor-of-two halving as
    in undirected static betweenness). "Foremost" paths are those achieving
    the earliest arrival, consistent with the latency used by
    :func:`temporal_distances` and :func:`temporal_closeness`.

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import temporal_betweenness
    >>> g0 = ig.Graph(n=3, edges=[(0, 1)])
    >>> g1 = ig.Graph(n=3, edges=[(1, 2)])
    >>> bt = temporal_betweenness([g0, g1], graph_labels=["t0", "t1"],
    ...                            report_gaps=False)
    >>> # Node 1 brokers the 0 -> 2 path; others broker nothing
    >>> float(bt[bt["node"] == 1]["betweenness"].iloc[0])
    0.5
    >>> float(bt[bt["node"] == 0]["betweenness"].iloc[0])
    0.0
    """
    graph_labels = validate_and_setup_graphs(graphs, graph_labels,
                                             min_length=1)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    gap_info = detect_temporal_gaps(graph_labels)
    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    # With cross_gaps=False, foremost paths are confined to a single
    # continuous segment, so the forward pass runs per segment.
    if cross_gaps:
        segments = [(0, len(graphs))]
    else:
        segments = gap_info.get("segments", [(0, len(graphs))])

    try:
        all_nodes = _union_nodes(graphs)
    except Exception as e:
        warnings.warn(f"Could not collect node set: {e}")
        return pd.DataFrame(columns=_BETW_COLS)

    betweenness: Dict = {n: 0.0 for n in all_nodes}

    for source in all_nodes:
        try:
            # Nodes whose global foremost arrival is already fixed by an
            # earlier segment (arrivals in later segments are strictly
            # later, so the first segment reaching a node is foremost).
            seen: Set = {source}
            for seg_start, seg_end in segments:
                order, preds, sigma = _foremost_paths_from_source(
                    graphs[seg_start:seg_end], source)
                # Only pairs whose foremost arrival lies in this segment
                # count as targets; already-seen nodes may still appear as
                # intermediates on paths to new targets.
                counted = {w for w in order if w not in seen}
                seen.update(order)

                delta: Dict = {n: 0.0 for n in order}
                # Reverse arrival order: accumulate dependencies
                # leaf-to-root.
                for w in reversed(order):
                    credit = (1.0 if w in counted else 0.0) + delta[w]
                    for u in preds.get(w, []):
                        delta[u] += (sigma[u] / sigma[w]) * credit
                    if w != source:
                        betweenness[w] += delta[w]
        except Exception as e:
            warnings.warn(f"Error computing betweenness from "
                          f"{source}: {e}; skipping this source")
            continue

    n = len(all_nodes)
    if normalized and n > 2:
        scale = 1.0 / ((n - 1) * (n - 2))
        betweenness = {k: v * scale for k, v in betweenness.items()}

    df = (pd.DataFrame([{"node": k, "betweenness": v}
                        for k, v in betweenness.items()])
            .sort_values("betweenness", ascending=False)
            .reset_index(drop=True))
    if df.empty:
        return pd.DataFrame(columns=_BETW_COLS)

    if save_path is not None:
        _plot_betweenness(df, save_path)

    return df


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
        logger.info("Plot saved: %s", path)

    except Exception as e:
        warnings.warn(f"Could not plot temporal closeness: {e}")


def _plot_betweenness(df: pd.DataFrame, save_path: str) -> None:
    """
    Save a horizontal bar chart of temporal betweenness values.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`temporal_betweenness`, sorted descending.
    save_path : str
        Directory where the PDF is saved.

    Returns
    -------
    None
    """
    try:
        nodes = [str(n) for n in df["node"]]
        values = df["betweenness"].values

        fig, ax = plt.subplots(figsize=(8, max(4, len(nodes) * 0.4)), dpi=100)
        y_pos = np.arange(len(nodes))
        ax.barh(y_pos, values, color='#d62728', edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(nodes, fontsize=10)
        ax.set_xlabel("Temporal Betweenness", fontsize=13, fontweight='bold')
        ax.set_title("Temporal Betweenness per Node",
                     fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(values.max() * 1.1, 0.01))
        plt.tight_layout()

        path = os.path.join(save_path, "temporal_betweenness.pdf")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("Plot saved: %s", path)

    except Exception as e:
        warnings.warn(f"Could not plot temporal betweenness: {e}")
