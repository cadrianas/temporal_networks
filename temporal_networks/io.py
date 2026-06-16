"""
Temporal Network Analysis: Event-Stream Ingestion Module

Most temporal-network data arrives as a *timestamped edge list* (an event
stream): one row per interaction, with a time column and the two endpoints.
This module bins such a stream into the ``(graphs, graph_labels)`` pair that
every analysis function in :mod:`temporal_networks` expects.

KEY FEATURES:
- Bins events into snapshots at a chosen frequency (monthly, daily, ...).
- Uses a shared (union) vertex set across all snapshots, so node identity is
  stable for downstream cross-snapshot analysis.
- Produces labels that the package's gap detector understands, so missing
  periods (skipped bins) are detected automatically.
"""

import numpy as np
import pandas as pd
import igraph as ig
from typing import List, Optional, Tuple

from ._gap_utilities import parse_flexible_datetime

__all__ = [
    "snapshots_from_events",
    "snapshots_from_edgelist",
]


# Bare pandas offset aliases that emit a deprecation warning in pandas >= 2.2;
# normalised to their period-end spellings to keep the default API friendly.
_FREQ_ALIAS = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}


def _period_to_label(ts: pd.Timestamp, freq: str,
                     label_format: Optional[str]) -> str:
    """
    Convert a bin's timestamp to a gap-detector-compatible label.

    When ``label_format`` is given it is applied verbatim via ``strftime``.
    Otherwise a label is derived from ``freq`` so that it round-trips through
    :func:`temporal_networks._gap_utilities.parse_flexible_datetime` (e.g.
    ``YYYY-Qn`` for quarterly and ``YYYY-Wnn`` for weekly, which ``strftime``
    cannot express directly).

    Parameters
    ----------
    ts : pandas.Timestamp
        Representative timestamp for the bin.
    freq : str
        The (normalised) binning frequency.
    label_format : str or None
        Explicit ``strftime`` format, or None to derive from ``freq``.

    Returns
    -------
    str
        The snapshot label.
    """
    if label_format is not None:
        return ts.strftime(label_format)

    base = "".join(ch for ch in str(freq).upper() if ch.isalpha())
    if base.startswith("Q"):
        return f"{ts.year}-Q{(ts.month - 1) // 3 + 1}"
    if base.startswith("W"):
        iso = ts.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if base in ("Y", "YE", "YS", "A", "AS"):
        return ts.strftime("%Y")
    if base in ("D", "B"):
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m")


def snapshots_from_events(
    df: pd.DataFrame,
    time_col: str,
    source_col: str,
    target_col: str,
    freq: str = "M",
    directed: bool = False,
    weight_col: Optional[str] = None,
    label_format: Optional[str] = None,
) -> Tuple[List[ig.Graph], List[str]]:
    """
    Bin a timestamped edge list into a sequence of igraph snapshots.

    Groups the events by time bin (``freq``) and builds one
    :class:`igraph.Graph` per non-empty bin. All snapshots share the same
    (union) vertex set, keyed by node ``name``, so the same node can be tracked
    across snapshots. Skipped bins (periods with no events) are simply absent
    from the output, which the package's gap detector then reports as gaps.

    Parameters
    ----------
    df : pandas.DataFrame
        Event table with at least the time, source, and target columns.
    time_col : str
        Name of the timestamp column (parseable by ``pandas.to_datetime``).
    source_col, target_col : str
        Names of the endpoint columns. Values are cast to ``str`` and used as
        vertex names.
    freq : str, optional
        Pandas offset alias for the bin width (default ``"M"`` = monthly).
        Common values: ``"D"`` daily, ``"W"`` weekly, ``"M"`` monthly,
        ``"Q"`` quarterly, ``"Y"`` yearly.
    directed : bool, optional
        If True, build directed graphs and treat ``(u, v)`` and ``(v, u)`` as
        distinct (default False).
    weight_col : str, optional
        If given, edge weights are the sum of this (numeric) column over the
        events that map to each edge in each bin. If None (default), parallel
        events are collapsed and, when any collapsing occurs, edge weights are
        the event multiplicity.
    label_format : str, optional
        Explicit ``strftime`` format for the snapshot labels. If None
        (default), a label is derived from ``freq`` that the gap detector can
        parse (e.g. ``"%Y-%m"`` monthly, ``"%Y-%m-%d"`` daily, ``YYYY-Qn``
        quarterly, ``YYYY-Wnn`` weekly).

    Returns
    -------
    graphs : list of igraph.Graph
        One graph per non-empty time bin, in chronological order. Vertices
        carry a ``name`` attribute; edges carry a ``weight`` attribute when
        weights are meaningful (see ``weight_col``).
    graph_labels : list of str
        Chronological labels, one per graph.

    Raises
    ------
    ValueError
        If a required column is missing, ``df`` is empty, the time column
        cannot be parsed, no non-empty bins are produced, or ``label_format``
        produces duplicate or unparseable labels.

    Examples
    --------
    >>> import pandas as pd
    >>> from temporal_networks.io import snapshots_from_events
    >>> events = pd.DataFrame({
    ...     "t": ["2024-01-05", "2024-01-20", "2024-03-02"],
    ...     "u": ["a", "b", "a"],
    ...     "v": ["b", "c", "c"],
    ... })
    >>> graphs, labels = snapshots_from_events(events, "t", "u", "v", freq="M")
    >>> labels
    ['2024-01', '2024-03']
    >>> [g.ecount() for g in graphs]
    [2, 1]
    """
    required = [time_col, source_col, target_col]
    if weight_col is not None:
        required.append(weight_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in df: {missing}")

    if df.empty:
        raise ValueError("df is empty; nothing to bin into snapshots")

    try:
        times = pd.to_datetime(df[time_col])
    except Exception as e:
        raise ValueError(
            f"Could not parse '{time_col}' as datetimes: {e}") from e

    # Union vertex set across the whole stream, in deterministic order, so
    # every snapshot shares the same node identities.
    node_series = pd.concat([df[source_col], df[target_col]], ignore_index=True)
    node_ids = sorted({str(n) for n in node_series})
    index_of = {n: i for i, n in enumerate(node_ids)}

    work = pd.DataFrame({
        "t": times.values,
        "src": df[source_col].astype(str).values,
        "dst": df[target_col].astype(str).values,
    })
    if weight_col is not None:
        try:
            work["w"] = pd.to_numeric(df[weight_col]).values
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"weight_col '{weight_col}' is not numeric: {e}") from e

    norm_freq = _FREQ_ALIAS.get(str(freq).upper(), freq)

    # First pass: aggregate edges per non-empty bin and learn whether any
    # parallel events were collapsed (which decides if weights are meaningful).
    periods = []  # list of (label, agg_series)
    any_parallel = False
    for period, sub in work.groupby(pd.Grouper(key="t", freq=norm_freq)):
        if sub.empty:
            continue  # skipped period -> becomes a gap downstream

        src = sub["src"].values
        dst = sub["dst"].values
        if directed:
            a, b = src, dst
        else:
            swap = src > dst
            a = np.where(swap, dst, src)
            b = np.where(swap, src, dst)

        tmp = pd.DataFrame({"a": a, "b": b})
        if weight_col is not None:
            tmp["w"] = sub["w"].values
            grouped = tmp.groupby(["a", "b"], sort=True)
            sizes = grouped.size()
            agg = grouped["w"].sum()
        else:
            grouped = tmp.groupby(["a", "b"], sort=True)
            sizes = grouped.size()
            agg = sizes

        if (sizes > 1).any():
            any_parallel = True

        label = _period_to_label(period, norm_freq, label_format)
        periods.append((label, agg))

    if not periods:
        raise ValueError(
            "No non-empty time bins were produced; check 'freq' and the "
            "values in 'time_col'.")

    labels = [label for label, _ in periods]

    if len(set(labels)) != len(labels):
        raise ValueError(
            "label_format produced duplicate labels for distinct time bins; "
            "choose a finer label_format for this freq.")
    for label in labels:
        if parse_flexible_datetime(label) is None:
            raise ValueError(
                f"Generated label {label!r} is not parseable by the gap "
                f"detector. Pass a compatible label_format (e.g. '%Y-%m' "
                f"monthly, '%Y-%m-%d' daily).")

    # Second pass: build graphs. Weights are attached only when meaningful,
    # and then on every snapshot for consistency.
    weighted = (weight_col is not None) or any_parallel
    graphs = []
    for _, agg in periods:
        g = ig.Graph(directed=directed)
        g.add_vertices(node_ids)
        edges = [(index_of[u], index_of[v]) for (u, v) in agg.index]
        g.add_edges(edges)
        if weighted:
            g.es["weight"] = [float(w) for w in agg.values]
        graphs.append(g)

    return graphs, labels


def snapshots_from_edgelist(
    path: str,
    time_col: str,
    source_col: str,
    target_col: str,
    **kwargs,
) -> Tuple[List[ig.Graph], List[str]]:
    """
    Load a timestamped edge list from a CSV file into snapshots.

    Thin convenience wrapper that reads ``path`` with ``pandas.read_csv`` and
    forwards to :func:`snapshots_from_events`.

    Parameters
    ----------
    path : str
        Path to a CSV file containing the event table.
    time_col, source_col, target_col : str
        Column names passed through to :func:`snapshots_from_events`.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`snapshots_from_events` (e.g. ``freq``, ``directed``,
        ``weight_col``, ``label_format``).

    Returns
    -------
    graphs : list of igraph.Graph
        One graph per non-empty time bin.
    graph_labels : list of str
        Chronological labels, one per graph.
    """
    df = pd.read_csv(path)
    return snapshots_from_events(
        df, time_col, source_col, target_col, **kwargs)
