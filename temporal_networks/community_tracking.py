"""
Temporal Network Analysis: Community Tracking Across Time (GAP-AWARE)

This module turns per-snapshot community detection into analyzable lineage
tables: it matches communities between consecutive snapshots and labels how
each one evolves — born, dies, continues, merges, or splits (Palla et al.;
Greene et al.).

KEY FEATURES:
- Reuses the shared ``_detect_communities`` helper, so detection is identical
  to ``plot_community_evolution`` / ``communities_measures``.
- Matches communities by Jaccard overlap of member node-keys
  (``_vertex_keys``), so identity is consistent across snapshots regardless of
  vertex ordering.
- Assigns a persistent ``lineage_id`` and a lifecycle ``event`` per community.
- Gap-aware: with ``bridge_gaps=False`` (default), lineages are not linked
  across a detected temporal gap — a community after a closure starts fresh
  unless it re-matches later.
"""

import logging
import os
import warnings

import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Set
from ._gap_utilities import (
    GapInfo,
    NodeKey,
    detect_temporal_gaps,
    print_gap_report,
    validate_and_setup_graphs,
    _vertex_keys,
)
from ._community_utils import _detect_communities

logger = logging.getLogger(__name__)

__all__ = [
    "track_communities",
    "plot_community_lineage",
]

_COLUMNS = ["Graph", "community_id", "lineage_id", "size", "event", "members"]


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _communities_as_keysets(partition, graph: ig.Graph) -> List[Set[NodeKey]]:
    """
    Return each community as a set of vertex identity keys.

    Parameters
    ----------
    partition : igraph.VertexClustering or None
        Detected partition for one snapshot, or None if detection failed.
    graph : igraph.Graph
        The snapshot the partition belongs to.

    Returns
    -------
    list of set
        One member-key set per community, indexed by a stable local id
        (communities are ordered by their original membership id).
    """
    if partition is None:
        return []
    keys = _vertex_keys(graph)
    groups: Dict[int, Set[NodeKey]] = {}
    for vidx, cid in enumerate(partition.membership):
        groups.setdefault(cid, set()).add(keys[vidx])
    return [groups[cid] for cid in sorted(groups)]


def _jaccard(a: Set[NodeKey], b: Set[NodeKey]) -> float:
    """Jaccard overlap of two sets (0.0 when both are empty)."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _dominant(mapping: Dict[int, float]) -> int:
    """Return the key with the highest value; ties broken by smallest key."""
    return max(mapping, key=lambda k: (mapping[k], -k))


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

def track_communities(
    graphs: List[ig.Graph],
    graph_labels: Optional[List[str]] = None,
    algorithm: str = "multilevel",
    match_threshold: float = 0.3,
    bridge_gaps: bool = False,
    report_gaps: bool = False,
) -> pd.DataFrame:
    """
    Match communities across snapshots and label lifecycle events.

    Communities are detected per snapshot, then matched between consecutive
    snapshots by Jaccard overlap of their member node-keys. Each community
    receives a persistent ``lineage_id`` and a lifecycle ``event``.

    **Gap-aware (default):** with ``bridge_gaps=False``, communities are not
    matched across a detected temporal gap, so a community appearing after a
    closure starts a fresh lineage. Pass ``bridge_gaps=True`` to link across
    gaps.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Temporal snapshots in chronological order.
    graph_labels : list of str, optional
        Labels for each snapshot. If None, defaults to "Graph 1", etc.
    algorithm : str, optional
        Community detection algorithm (default ``"multilevel"``). One of
        ``edge_betweenness``, ``walktrap``, ``fast_greedy``, ``label_prop``,
        ``spinglass``, ``leiden``, ``louvain`` / ``multilevel``, ``infomap``.
    match_threshold : float, optional
        Minimum Jaccard overlap for two communities in consecutive snapshots
        to be considered the same lineage (default ``0.3``).
    bridge_gaps : bool, optional
        If False (default), lineages are not linked across a detected gap.
    report_gaps : bool, optional
        If True, print a temporal gap report to the console
        (default: False).

    Returns
    -------
    pandas.DataFrame
        One row per (snapshot, community), with columns:

        - ``Graph``: snapshot label
        - ``community_id``: local id within the snapshot
        - ``lineage_id``: persistent id across snapshots
        - ``size``: member count
        - ``event``: ``birth`` / ``death`` / ``continue`` / ``merge`` /
          ``split``
        - ``members``: sorted list of member node-keys

        Returns an empty DataFrame with these columns on empty input.

    Notes
    -----
    Events are classified by match multiplicity, in priority order: a
    community with two or more predecessors is a ``merge``; one whose single
    predecessor also produced other communities is a ``split``; one with no
    predecessor is a ``birth``; one with no successor is a ``death``;
    otherwise it is a ``continue``. The taxonomy is intentionally minimal —
    size-based ``grow`` / ``shrink`` are left to the ``size`` column.

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import track_communities
    >>> # Two clear clusters that persist across two snapshots
    >>> edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    >>> g = ig.Graph(n=6, edges=edges)
    >>> df = track_communities([g, g.copy()], graph_labels=["t0", "t1"],
    ...                         algorithm="louvain", report_gaps=False)
    >>> sorted(df["event"].unique())
    ['birth', 'continue']
    """
    graph_labels = validate_and_setup_graphs(graphs, graph_labels,
                                             min_length=1)
    gap_info = detect_temporal_gaps(graph_labels)
    if report_gaps:
        print_gap_report(graph_labels, gap_info)
    gap_ends = {g["end_idx"] for g in gap_info.get("gaps", [])}

    partitions, _ = _detect_communities(graphs, algorithm)

    # Per-snapshot communities as member-key sets.
    comms: List[List[Set[NodeKey]]] = [
        _communities_as_keysets(partitions[i], graphs[i])
        for i in range(len(graphs))
    ]

    t_count = len(graphs)
    # preds[i][d] = {c: jaccard}; succs[i][c] = {d: jaccard} for i -> i+1.
    preds: List[Dict[int, Dict[int, float]]] = [dict() for _ in range(t_count)]
    succs: List[Dict[int, Dict[int, float]]] = [dict() for _ in range(t_count)]

    for i in range(1, t_count):
        if not bridge_gaps and i in gap_ends:
            continue  # gap boundary: no matches carry across it
        for c, cset in enumerate(comms[i - 1]):
            for d, dset in enumerate(comms[i]):
                j = _jaccard(cset, dset)
                if j >= match_threshold:
                    succs[i - 1].setdefault(c, {})[d] = j
                    preds[i].setdefault(d, {})[c] = j

    # Assign lineage ids: a lineage continues through the dominant successor
    # of its dominant predecessor, so it threads at most one community per
    # snapshot.
    lineage: List[Dict[int, int]] = [dict() for _ in range(t_count)]
    next_lineage = 0
    for i in range(t_count):
        for d in range(len(comms[i])):
            p = preds[i].get(d, {})
            eligible = {c: j for c, j in p.items()
                        if succs[i - 1].get(c) and
                        _dominant(succs[i - 1][c]) == d}
            if eligible:
                lineage[i][d] = lineage[i - 1][_dominant(eligible)]
            else:
                lineage[i][d] = next_lineage
                next_lineage += 1

    def _has_successor_snapshot(i: int) -> bool:
        """Whether snapshot ``i`` has a matchable following snapshot."""
        if i + 1 >= t_count:
            return False
        if not bridge_gaps and (i + 1) in gap_ends:
            return False
        return True

    rows = []
    for i in range(t_count):
        for d, dset in enumerate(comms[i]):
            p = preds[i].get(d, {})
            s = succs[i].get(d, {})
            n_pred, n_succ = len(p), len(s)

            is_split = False
            if n_pred == 1:
                c = next(iter(p))
                if len(succs[i - 1].get(c, {})) > 1:
                    is_split = True

            if n_pred >= 2:
                event = "merge"
            elif is_split:
                event = "split"
            elif n_pred == 0:
                event = "birth"
            elif _has_successor_snapshot(i) and n_succ == 0:
                # A following snapshot exists but the lineage finds no match
                # there: the community dies out (rather than simply being the
                # final observation).
                event = "death"
            else:
                event = "continue"

            rows.append({
                "Graph": graph_labels[i],
                "community_id": d,
                "lineage_id": lineage[i][d],
                "size": len(dset),
                "event": event,
                "members": sorted(dset, key=str),
            })

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.DataFrame(rows, columns=_COLUMNS)


def plot_community_lineage(
    tracking_df: pd.DataFrame,
    graph_labels: List[str],
    gap_info: GapInfo,
    save_path: Optional[str] = None,
) -> None:
    """
    Timeline view of community lineages over the snapshot sequence.

    Each community is plotted at ``(snapshot, lineage_id)``, sized by member
    count and coloured by lifecycle event. Communities sharing a lineage are
    connected across consecutive snapshots; connections are not drawn across a
    detected temporal gap.

    Parameters
    ----------
    tracking_df : pandas.DataFrame
        Output of :func:`track_communities`.
    graph_labels : list of str
        Temporal labels for all snapshots (defines the x-axis order).
    gap_info : dict
        Gap information from :func:`~temporal_networks.detect_temporal_gaps`;
        connections are suppressed across each gap boundary.
    save_path : str, optional
        Directory for saving the PDF. If None (default), no file is written.

    Returns
    -------
    None
    """
    if save_path is None:
        return
    try:
        os.makedirs(save_path, exist_ok=True)
        label_to_idx = {lbl: i for i, lbl in enumerate(graph_labels)}
        gap_ends = {g["end_idx"] for g in gap_info.get("gaps", [])}

        event_colors = {
            "birth": "#2ca02c", "death": "#d62728", "continue": "#1f77b4",
            "merge": "#9467bd", "split": "#ff7f0e",
        }

        fig, ax = plt.subplots(figsize=(max(8, len(graph_labels) * 1.2), 6),
                               dpi=100)

        # Lines connecting communities that share a lineage.
        for lineage_id, grp in tracking_df.groupby("lineage_id"):
            pts = sorted((label_to_idx[r["Graph"]], r["lineage_id"])
                         for _, r in grp.iterrows())
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if x1 == x0 + 1 and x1 not in gap_ends:
                    ax.plot([x0, x1], [y0, y1], color="#cccccc",
                            linewidth=1, zorder=1)

        for event, color in event_colors.items():
            sub = tracking_df[tracking_df["event"] == event]
            if sub.empty:
                continue
            xs = [label_to_idx[lbl] for lbl in sub["Graph"]]
            ys = sub["lineage_id"].tolist()
            sizes = [max(40, s * 20) for s in sub["size"]]
            ax.scatter(xs, ys, s=sizes, c=color, label=event,
                       edgecolors="black", alpha=0.85, zorder=2)

        ax.set_xticks(range(len(graph_labels)))
        ax.set_xticklabels(graph_labels, rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Time", fontsize=13, fontweight="bold")
        ax.set_ylabel("Lineage", fontsize=13, fontweight="bold")
        ax.set_title("Community Lineages Over Time", fontsize=15,
                     fontweight="bold")
        ax.legend(title="Event", loc="best", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(save_path, "community_lineage.pdf")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved: %s", path)

    except Exception as e:
        warnings.warn(f"Could not plot community lineage: {e}")
