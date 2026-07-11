"""
Temporal Network Analysis: Burstiness & Inter-event Module (GAP-AWARE)

This module characterises the *timing* of edge/node activity across snapshots:
whether activations are regular, Poisson-like, or bursty.

KEY FEATURES:
- Long-form inter-event intervals between consecutive activations of each
  edge or node, in the inferred time unit of the labels.
- Goh-Barabasi burstiness coefficient B = (sigma - mu) / (sigma + mu) per
  entity, where B = -1 is perfectly regular, 0 is Poisson-like, and B -> 1 is
  bursty.
- Gap-aware: intervals that straddle a detected temporal gap are flagged via
  ``spans_gap`` and (by default) excluded, so a data closure is never mistaken
  for an entity being inactive.
"""

import logging
import os
import warnings

import igraph as ig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from ._gap_utilities import (
    GapDict,
    _COMPUTE_ERRORS,
    detect_temporal_gaps,
    print_gap_report,
    validate_and_setup_graphs,
    parse_flexible_datetime,
    calculate_time_difference,
    _infer_unit_and_threshold,
    _active_nodes,
)
from .edge_formation_dissolution import _edge_identity_set

logger = logging.getLogger(__name__)

__all__ = [
    "inter_event_times",
    "burstiness_coefficient",
]

_IET_COLUMNS = ["entity", "start_label", "end_label", "interval", "spans_gap"]
_BURST_COLUMNS = ["entity", "n_events", "mean_interval", "std_interval",
                  "burstiness"]


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _entity_active_indices(graphs: List[ig.Graph],
                           by: str) -> Dict[str, List[int]]:
    """
    Map each entity to the ordered snapshot indices where it is active.

    An edge is active in a snapshot if it is present; a node is active if it is
    an endpoint of at least one edge. Entities are keyed by their string form so
    the result is a tidy ``str -> list of int`` mapping.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Consecutive temporal snapshots.
    by : str
        ``"edge"`` to track edges, ``"node"`` to track nodes.

    Returns
    -------
    dict
        Mapping ``entity_key -> sorted list of snapshot indices``.
    """
    active: Dict[str, List[int]] = {}
    for i, graph in enumerate(graphs):
        try:
            edges = _edge_identity_set(graph)
            if by == "edge":
                keys = {str(e) for e in edges}
            else:
                keys = {str(n) for n in _active_nodes(edges)}
            for key in keys:
                active.setdefault(key, []).append(i)
        except _COMPUTE_ERRORS as e:
            warnings.warn(f"Error processing snapshot {i}: {e}; "
                          f"skipping this snapshot")
            continue
    return active


def _interval_duration(i_a: int, i_b: int, graph_labels: List[str],
                       unit: str) -> float:
    """
    Duration between two snapshots in the inferred time unit.

    Uses :func:`calculate_time_difference` on the parsed labels when both parse;
    otherwise falls back to the difference in snapshot index, so unlabelled or
    non-datetime sequences still yield meaningful relative intervals.
    """
    date_a = parse_flexible_datetime(graph_labels[i_a])
    date_b = parse_flexible_datetime(graph_labels[i_b])
    if date_a is not None and date_b is not None:
        return calculate_time_difference(date_a, date_b, unit=unit)
    return float(i_b - i_a)


def _spans_gap(i_a: int, i_b: int, gaps: List[GapDict]) -> bool:
    """Whether the interval ``[i_a, i_b]`` straddles any detected gap break."""
    for gap in gaps:
        if i_a <= gap["start_idx"] and i_b >= gap["end_idx"]:
            return True
    return False


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def inter_event_times(graphs: List[ig.Graph],
                      graph_labels: Optional[List[str]] = None,
                      by: str = "edge",
                      exclude_gaps: bool = True) -> pd.DataFrame:
    """
    Durations between consecutive activations of each edge or node.

    For every entity, the snapshots in which it is active are ordered and each
    consecutive pair contributes one inter-event interval, measured in the time
    unit inferred from the labels (falling back to snapshot-index distance for
    non-datetime labels).

    **Gap-aware:** intervals whose span crosses a detected temporal gap are
    flagged via ``spans_gap``. When ``exclude_gaps`` is True (default) those
    intervals are dropped, so a data closure is not counted as a long quiet
    period.

    Parameters
    ----------
    graphs : list of igraph.Graph
        List of igraph.Graph objects representing consecutive time points.
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...]).
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    by : str, optional
        ``"edge"`` (default) to track edge activations, ``"node"`` to track
        node activations.
    exclude_gaps : bool, optional
        If True (default), drop intervals that straddle a detected temporal gap.
        If False, keep them (still flagged via ``spans_gap``).

    Returns
    -------
    pandas.DataFrame
        One row per inter-event interval, with columns:

        - ``entity``: edge key ``"(u, v)"`` or node key
        - ``start_label`` / ``end_label``: bounding snapshots of the interval
        - ``interval``: duration in the inferred time unit
        - ``spans_gap``: whether the interval crosses a detected data gap

        Sorted by ``entity`` then snapshot order (chronological, not
        lexicographic). Empty input returns an empty DataFrame with these
        columns.

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import inter_event_times
    >>> on = ig.Graph(n=2, edges=[(0, 1)])
    >>> off = ig.Graph(n=2)
    >>> graphs = [on, off, on.copy()]  # edge active at snapshots 0 and 2
    >>> labels = ["2024-01", "2024-02", "2024-03"]
    >>> iet = inter_event_times(graphs, graph_labels=labels)
    >>> list(iet["interval"])
    [2.0]
    >>> bool(iet.loc[0, "spans_gap"])
    False
    """
    if by not in ("edge", "node"):
        raise ValueError(f"by must be 'edge' or 'node', got {by!r}")

    graph_labels = validate_and_setup_graphs(graphs, graph_labels, min_length=2)

    gap_info = detect_temporal_gaps(graph_labels)
    gaps = gap_info.get("gaps", [])
    unit, _ = _infer_unit_and_threshold(graph_labels)

    active = _entity_active_indices(graphs, by)

    rows = []
    for entity in sorted(active):
        idxs = active[entity]
        for i_a, i_b in zip(idxs, idxs[1:]):
            try:
                spans = _spans_gap(i_a, i_b, gaps)
                if spans and exclude_gaps:
                    continue
                rows.append({
                    "entity": entity,
                    "start_label": graph_labels[i_a],
                    "end_label": graph_labels[i_b],
                    "interval": _interval_duration(i_a, i_b, graph_labels,
                                                   unit),
                    "spans_gap": spans,
                    # snapshot index, for chronological sorting (string
                    # labels like "Graph 10" sort before "Graph 2")
                    "_start_idx": i_a,
                })
            except _COMPUTE_ERRORS as e:
                warnings.warn(f"Error timing entity {entity} between "
                              f"snapshots {i_a} and {i_b}: {e}; "
                              f"skipping this interval")
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=_IET_COLUMNS)
    return (df.sort_values(["entity", "_start_idx"])
              .drop(columns="_start_idx")
              .reset_index(drop=True))


def burstiness_coefficient(graphs: List[ig.Graph],
                           graph_labels: Optional[List[str]] = None,
                           by: str = "edge",
                           exclude_gaps: bool = True,
                           save_path: Optional[str] = None,
                           report_gaps: bool = False) -> pd.DataFrame:
    """
    Goh-Barabasi burstiness coefficient per edge or node.

    For each entity, the inter-event intervals are summarised by their mean and
    standard deviation, and the burstiness coefficient
    ``B = (sigma - mu) / (sigma + mu)`` is reported. ``B = -1`` indicates a
    perfectly regular (periodic) activation pattern, ``B = 0`` a Poisson-like
    one, and ``B`` approaching 1 an increasingly bursty (clustered) one.

    **Gap-aware:** when ``exclude_gaps`` is True (default), intervals straddling
    a detected temporal gap are left out of the statistics, so a data closure is
    not read as a long inactive stretch.

    Parameters
    ----------
    graphs : list of igraph.Graph
        List of igraph.Graph objects representing consecutive time points.
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...]).
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    by : str, optional
        ``"edge"`` (default) to score edges, ``"node"`` to score nodes.
    exclude_gaps : bool, optional
        If True (default), exclude gap-straddling intervals from the statistics.
    save_path : str, optional
        Directory for saving the burstiness-distribution plot. If None
        (default), no file is saved.
    report_gaps : bool, optional
        If True, print a temporal gap report to the console
        (default: False).

    Returns
    -------
    pandas.DataFrame
        One row per entity, with columns:

        - ``entity``: edge/node key
        - ``n_events``: number of active snapshots
        - ``mean_interval`` / ``std_interval``: of the inter-event times
        - ``burstiness``: B in [-1, 1]; ``NaN`` for entities with fewer than two
          events or no usable intervals

        Empty input returns an empty DataFrame with these columns.

    Notes
    -----
    The standard deviation is the population standard deviation (``ddof=0``), so
    an entity with a single usable interval has ``std_interval == 0`` and
    ``burstiness == -1`` (maximally regular).

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import burstiness_coefficient
    >>> g_on = ig.Graph(n=2, edges=[(0, 1)])
    >>> graphs = [g_on, g_on.copy(), g_on.copy()]  # active every snapshot
    >>> labels = ["2024-01", "2024-02", "2024-03"]
    >>> bdf = burstiness_coefficient(graphs, graph_labels=labels,
    ...                              report_gaps=False)
    >>> int(bdf.loc[0, "n_events"])
    3
    >>> float(bdf.loc[0, "burstiness"])  # perfectly regular -> -1
    -1.0
    """
    if by not in ("edge", "node"):
        raise ValueError(f"by must be 'edge' or 'node', got {by!r}")

    graph_labels = validate_and_setup_graphs(graphs, graph_labels, min_length=2)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    gap_info = detect_temporal_gaps(graph_labels)
    if report_gaps:
        print_gap_report(graph_labels, gap_info)

    gaps = gap_info.get("gaps", [])
    unit, _ = _infer_unit_and_threshold(graph_labels)

    active = _entity_active_indices(graphs, by)

    rows = []
    for entity in sorted(active):
        idxs = active[entity]
        try:
            intervals = []
            for i_a, i_b in zip(idxs, idxs[1:]):
                if exclude_gaps and _spans_gap(i_a, i_b, gaps):
                    continue
                intervals.append(
                    _interval_duration(i_a, i_b, graph_labels, unit))

            if intervals:
                mean = float(np.mean(intervals))
                std = float(np.std(intervals))
                denom = std + mean
                burst = (std - mean) / denom if denom > 0 else float("nan")
            else:
                mean = std = burst = float("nan")

            rows.append({
                "entity": entity,
                "n_events": len(idxs),
                "mean_interval": mean,
                "std_interval": std,
                "burstiness": burst,
            })
        except _COMPUTE_ERRORS as e:
            warnings.warn(f"Error scoring entity {entity}: {e}; "
                          f"skipping this entity")
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=_BURST_COLUMNS)

    if save_path is not None:
        _plot_burstiness(df, by, save_path)

    return df


def _plot_burstiness(df: pd.DataFrame, by: str, save_path: str) -> None:
    """
    Plot the distribution of burstiness coefficients.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`burstiness_coefficient`.
    by : str
        ``"edge"`` or ``"node"``; used for axis/title wording and filename.
    save_path : str
        Directory where the histogram PDF is saved.

    Returns
    -------
    None
    """
    try:
        values = df["burstiness"].dropna().values if "burstiness" in df.columns \
            else np.array([])

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        if len(values) > 0:
            ax.hist(values, bins=20, range=(-1, 1), color='#1f77b4',
                    edgecolor='black', alpha=0.8)
        ax.axvline(0.0, color='grey', linestyle='--', linewidth=1.5)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Burstiness B", fontsize=14, fontweight='bold')
        ax.set_ylabel(f"Number of {by}s", fontsize=14, fontweight='bold')
        ax.set_title(f"Burstiness Distribution ({by})", fontsize=16,
                     fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_filename = os.path.join(save_path, f"burstiness_{by}.pdf")
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("Plot saved: %s", plot_filename)

    except Exception as e:
        warnings.warn(f"Could not plot burstiness distribution: {e}")
