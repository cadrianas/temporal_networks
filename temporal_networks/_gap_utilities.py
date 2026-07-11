"""
Temporal Gap Utilities Module

This module provides shared logic for temporal gap detection, datetime parsing,
and gap-aware plotting across the temporal_networks package.
"""

import re
import warnings
from collections import Counter
from datetime import datetime
from typing import List, Optional, Set, Tuple, TypedDict, Union

import igraph as ig
import numpy as np
import pandas as pd

# A vertex identity key: the vertex "name" (or "label") when present —
# usually a string — otherwise the integer vertex index. See _vertex_keys.
NodeKey = Union[str, int]


class GapDict(TypedDict):
    """One detected temporal gap (an entry of ``GapInfo["gaps"]``)."""

    start_idx: int
    end_idx: int
    start_label: str
    end_label: str
    gap_size: float


class GapInfo(TypedDict):
    """Return schema of :func:`detect_temporal_gaps`."""

    has_gaps: bool
    num_gaps: int
    gaps: List[GapDict]
    segments: List[Tuple[int, int]]
    report: str


# Pattern of the auto-generated placeholder labels produced by
# ``validate_and_setup_graphs`` when the user supplies no labels. These
# intentionally carry no temporal information, so gap detection stays
# silently disabled for them (no warning).
_DEFAULT_LABEL_RE = re.compile(r"^Graph \d+$")


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_and_setup_graphs(graphs: List[ig.Graph],
                              graph_labels: Optional[List[str]] = None,
                              min_length: int = 1) -> List[str]:
    """
    Validate the graphs list and setup/validate graph labels.

    Parameters
    ----------
    graphs : list
        List of graph objects
    graph_labels : list of str, optional
        Labels for each graph
    min_length : int, optional
        Minimum number of graphs required (default: 1)

    Returns
    -------
    list of str
        The validated graph labels
    """
    if len(graphs) < min_length:
        if min_length == 1:
            raise ValueError("graphs list cannot be empty")
        else:
            raise ValueError(f"At least {min_length} graphs required")

    if graph_labels is None:
        graph_labels = [f"Graph {i+1}" for i in range(len(graphs))]
    elif len(graph_labels) != len(graphs):
        raise ValueError(
            f"graph_labels length ({len(graph_labels)}) must match "
            f"graphs length ({len(graphs)})")

    return graph_labels


def _vertex_keys(graph: ig.Graph) -> List[NodeKey]:
    """
    Return per-vertex identity keys for a graph.

    Vertices are keyed by their ``name`` attribute when present, otherwise by
    ``label``, otherwise by integer position. This gives a stable identity for
    comparing the same node across temporal snapshots even when snapshots store
    their vertices in a different order.

    Parameters
    ----------
    graph : igraph.Graph
        Graph whose vertex keys to extract.

    Returns
    -------
    list
        One identity key per vertex, in vertex-index order. Keys are the
        ``name`` values, the ``label`` values, or ``list(range(vcount))`` as a
        fallback when neither attribute is present.

    Raises
    ------
    ValueError
        If two vertices share the same ``name`` (or ``label``) key.
        Duplicate keys would silently merge distinct vertices in every
        identity-based comparison (edge sets, neighbourhoods, temporal
        paths), corrupting results without any visible error.

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks._gap_utilities import _vertex_keys
    >>> g = ig.Graph(n=3)
    >>> g.vs["name"] = ["a", "b", "c"]
    >>> _vertex_keys(g)
    ['a', 'b', 'c']
    >>> _vertex_keys(ig.Graph(n=2))
    [0, 1]
    """
    attrs = graph.vs.attributes()
    if "name" in attrs:
        keys, attr = list(graph.vs["name"]), "name"
    elif "label" in attrs:
        keys, attr = list(graph.vs["label"]), "label"
    else:
        return list(range(graph.vcount()))

    if len(set(keys)) != len(keys):
        dupes = sorted((k for k, c in Counter(keys).items() if c > 1),
                       key=str)
        raise ValueError(
            f"Duplicate vertex {attr!r} keys {dupes[:5]} in graph: node "
            f"identity must be unique for cross-snapshot comparison. "
            f"Rename the duplicated vertices before analysis.")
    return keys


def _active_nodes(
        edge_set: Set[Tuple[NodeKey, NodeKey]]) -> Set[NodeKey]:
    """Return the set of node keys that are an endpoint of at least one edge."""
    nodes: Set[NodeKey] = set()
    for u, v in edge_set:
        nodes.add(u)
        nodes.add(v)
    return nodes


# ============================================================================
# DATETIME PARSING
# ============================================================================

def parse_flexible_datetime(label: str) -> Optional[datetime]:
    """
    Parse a datetime label in multiple formats.

    Supports:
    - "YYYY-MM" (e.g., "2024-03") - RECOMMENDED for temporal networks
    - "YYYY-MM-DD" (e.g., "2024-03-15")
    - "YYYY-W##" (e.g., "2024-W12" for week 12)
    - "YYYY-Q#" (e.g., "2024-Q2" for quarter 2)
    - "YYYY" (e.g., "2024" for year only)

    Parameters
    ----------
    label : str
        Datetime label in any supported format

    Returns
    -------
    datetime or None
        Parsed datetime object, or None if parsing fails

    Examples
    --------
    >>> parse_flexible_datetime("2024-03")
    datetime.datetime(2024, 3, 1, 0, 0)

    >>> parse_flexible_datetime("2024-03-15")
    datetime.datetime(2024, 3, 15, 0, 0)
    """
    # Try YYYY-MM first (most common)
    try:
        return datetime.strptime(label.strip(), "%Y-%m")
    except ValueError:
        pass

    # Try YYYY-MM-DD
    try:
        return datetime.strptime(label.strip(), "%Y-%m-%d")
    except ValueError:
        pass

    # Try YYYY-W## (ISO 8601 week format). %G/%V/%u are the ISO week-date
    # directives; the non-ISO %Y/%W pair drifts by up to a week around
    # year boundaries (e.g. ISO 2025-W01 starts 2024-12-30), which would
    # make continuous weekly data report a false gap every January.
    try:
        year, week = label.strip().split('-W')
        # Monday of the given ISO week
        return datetime.strptime(f"{int(year)}-W{int(week):02d}-1",
                                 "%G-W%V-%u")
    except (ValueError, AttributeError, IndexError):
        pass

    # Try YYYY-Q# (quarter format)
    try:
        year, quarter = label.strip().split('-Q')
        month = (int(quarter) - 1) * 3 + 1
        return datetime(int(year), month, 1)
    except (ValueError, IndexError):
        pass

    # Try YYYY only
    try:
        return datetime.strptime(label.strip(), "%Y")
    except ValueError:
        pass

    return None


def calculate_time_difference(date1: datetime, date2: datetime,
                              unit: str = "months") -> float:
    """
    Calculate time difference between two dates in specified units.

    Parameters
    ----------
    date1 : datetime
        First date
    date2 : datetime
        Second date (should be after date1)
    unit : str, optional
        Time unit: "days", "weeks", "months", "years" (default: "months")

    Returns
    -------
    float
        Time difference in specified units

    Examples
    --------
    >>> d1 = datetime(2024, 3, 1)
    >>> d2 = datetime(2024, 5, 1)
    >>> calculate_time_difference(d1, d2, unit="months")
    2.0
    """
    if date1 > date2:
        date1, date2 = date2, date1

    if unit == "days":
        return float((date2 - date1).days)
    elif unit == "weeks":
        return (date2 - date1).days / 7.0
    elif unit == "months":
        return float((date2.year - date1.year) * 12 + (date2.month - date1.month))
    elif unit == "years":
        return (date2.year - date1.year) + (date2.month - date1.month) / 12.0
    else:
        raise ValueError(
            f"Unknown unit: {unit}. Use 'days', 'weeks', 'months', "
            f"or 'years'")


# ============================================================================
# GAP DETECTION
# ============================================================================

def _infer_unit_and_threshold(graph_labels: List[str]) -> Tuple[str, int]:
    """
    Infer the natural time unit and gap threshold from the label format.

    Inspects the first parseable label and returns the ``(unit, gap_threshold)``
    pair for which a regularly spaced series of that format reports *no* gaps
    (a gap is only flagged when a period is skipped):

    - ``YYYY-W##`` (weekly)    -> ``("weeks", 1)``
    - ``YYYY-Q#``  (quarterly) -> ``("months", 3)`` (one quarter is 3 months)
    - ``YYYY-MM-DD`` (daily)   -> ``("days", 1)``
    - ``YYYY-MM`` (monthly)    -> ``("months", 1)``
    - ``YYYY`` (yearly)        -> ``("years", 1)``

    Falls back to ``("months", 1)`` if no label can be parsed.

    Parameters
    ----------
    graph_labels : list of str
        Temporal labels to inspect.

    Returns
    -------
    tuple of (str, int)
        The inferred ``(unit, gap_threshold)``.
    """
    for label in graph_labels:
        if parse_flexible_datetime(label) is None:
            continue
        s = label.strip().upper()
        if "-W" in s:
            return "weeks", 1
        if "-Q" in s:
            return "months", 3
        n_parts = len(s.split("-"))
        if n_parts == 3:      # YYYY-MM-DD
            return "days", 1
        if n_parts == 2:      # YYYY-MM
            return "months", 1
        return "years", 1     # YYYY
    return "months", 1


def detect_temporal_gaps(graph_labels: List[str],
                         gap_threshold: Optional[int] = None,
                         unit: Optional[str] = None,
                         verbose: bool = False) -> GapInfo:
    """
    Detect temporal gaps in a list of labels with detailed reporting.

    A gap occurs when consecutive labels are more than `gap_threshold` apart
    (in specified units). This is useful for identifying seasonal closures,
    maintenance windows, data collection gaps, etc.

    Parameters
    ----------
    graph_labels : list of str
        Temporal labels (e.g., ["2024-03", "2024-04", "2024-11"])
    gap_threshold : int, optional
        A gap is flagged when consecutive labels are more than ``gap_threshold``
        ``unit`` apart. If None (default), it is inferred from the label format
        (1 for monthly/daily/weekly/yearly, 3 for quarterly) so that a regularly
        spaced series reports no gaps. Pass an explicit value to override.
    unit : str, optional
        Time unit for the gap calculation: "days", "weeks", "months", or
        "years". If None (default), inferred from the label format (e.g.
        "months" for ``YYYY-MM``, "days" for ``YYYY-MM-DD``, "years" for
        ``YYYY``, "weeks" for ``YYYY-W##``). Pass an explicit value to override.
    verbose : bool, optional
        If True, print detailed gap report (default: False)

    Returns
    -------
    dict
        Dictionary with keys:

        - "has_gaps" (bool): Whether gaps were detected
        - "num_gaps" (int): Number of gaps found
        - "gaps" (list): List of gap information dicts with:

          - "start_idx" (int): Index of last point before gap
          - "end_idx" (int): Index of first point after gap
          - "start_label" (str): Label before gap
          - "end_label" (str): Label after gap
          - "gap_size" (float): Size of gap in specified units

        - "segments" (list): List of (start_idx, end_idx) tuples for
          continuous segments
        - "report" (str): Human-readable gap report

    Warns
    -----
    UserWarning
        If any label cannot be parsed as a date, gap detection is disabled
        (the result reports no gaps and a single continuous segment) and a
        ``UserWarning`` is emitted naming the offending labels. No warning
        is emitted for the auto-generated placeholder labels
        (``"Graph 1"``, ``"Graph 2"``, ...) used when no labels are given.

    Examples
    --------
    >>> labels = ["2024-03", "2024-04", "2024-05", "2024-11", "2024-12"]
    >>> result = detect_temporal_gaps(labels, verbose=False)
    >>> result["has_gaps"]
    True
    >>> result["num_gaps"]
    1
    >>> result["gaps"][0]["gap_size"]
    6.0
    """

    # Infer the natural cadence from the label format so quarterly/yearly/daily/
    # weekly series are handled correctly, not just monthly. Explicit arguments
    # always take precedence.
    inferred_unit, inferred_threshold = _infer_unit_and_threshold(graph_labels)
    if unit is None:
        unit = inferred_unit
    if gap_threshold is None:
        gap_threshold = inferred_threshold

    if len(graph_labels) < 2:
        report = "No gaps: Data is continuous or contains fewer than 2 points."
        return {
            "has_gaps": False,
            "num_gaps": 0,
            "gaps": [],
            "segments": [(0, len(graph_labels))],
            "report": report
        }

    # Parse all labels
    parsed_dates = [parse_flexible_datetime(label) for label in graph_labels]

    # Check if all parsing was successful
    if any(d is None for d in parsed_dates):
        bad = [label for label, d in zip(graph_labels, parsed_dates)
               if d is None]
        # Auto-generated placeholders ("Graph 1", ...) mean the user never
        # supplied temporal labels — disable silently. Anything else is a
        # likely formatting mistake the user must hear about, because every
        # gap-aware function silently degrades when detection is off.
        if not all(_DEFAULT_LABEL_RE.match(str(label))
                   for label in graph_labels):
            warnings.warn(
                f"Could not parse {len(bad)} of {len(graph_labels)} "
                f"labels as dates (e.g. {bad[:3]}); temporal gap "
                f"detection is DISABLED and gap-aware functions will "
                f"treat the sequence as continuous. Use label formats "
                f"'YYYY-MM', 'YYYY-MM-DD', 'YYYY-W##', 'YYYY-Q#' or "
                f"'YYYY' to enable gap detection.")
        report = "Warning: Could not parse all labels. Gap detection disabled."
        return {
            "has_gaps": False,
            "num_gaps": 0,
            "gaps": [],
            "segments": [(0, len(graph_labels))],
            "report": report
        }

    # Find gaps
    gaps: List[GapDict] = []
    segments: List[Tuple[int, int]] = []
    segment_start = 0

    for i in range(1, len(parsed_dates)):
        date_prev = parsed_dates[i - 1]
        date_curr = parsed_dates[i]
        # Guaranteed non-None by the all-parsed check above (narrows for typing)
        assert date_prev is not None and date_curr is not None

        time_diff = calculate_time_difference(date_prev, date_curr, unit=unit)

        # Gap detected if difference exceeds threshold
        if time_diff > gap_threshold:
            gaps.append({
                "start_idx": i - 1,
                "end_idx": i,
                "start_label": graph_labels[i - 1],
                "end_label": graph_labels[i],
                "gap_size": time_diff,
            })

            segments.append((segment_start, i))
            segment_start = i

    # Add final segment
    segments.append((segment_start, len(graph_labels)))

    # Generate report
    report = _generate_gap_report_text(graph_labels, gaps, unit=unit)

    result: GapInfo = {
        "has_gaps": len(gaps) > 0,
        "num_gaps": len(gaps),
        "gaps": gaps,
        "segments": segments,
        "report": report
    }

    if verbose:
        print(report)

    return result


def _generate_gap_report_text(graph_labels: List[str], gaps: List[GapDict],
                              unit: str = "months") -> str:
    """
    Generate human-readable gap report string.

    Parameters
    ----------
    graph_labels : list of str
        Original temporal labels
    gaps : list of dict
        Detected gaps
    unit : str, optional
        Time unit for reporting (default: "months")

    Returns
    -------
    str
        Human-readable report
    """
    has_gaps = len(gaps) > 0
    lines = [
        "=" * 80,
        "TEMPORAL DATA STRUCTURE ANALYSIS",
        "=" * 80,
        "\nDataset Overview:",
        f"  Number of observations: {len(graph_labels)}",
        f"  Time unit: {unit}",
        f"  Date range: {graph_labels[0]} to {graph_labels[-1]}"
    ]

    if not has_gaps:
        lines.append("\n✓ Data is CONTINUOUS (no gaps detected)")
        lines.append("  All observations are sequential with no missing periods.")
    else:
        lines.append(f"\n⚠ Data has GAPS: {len(gaps)} gap(s) detected\n")

        for i, gap in enumerate(gaps, 1):
            lines.append(f"  Gap #{i}:")
            lines.append(f"    From: {gap['start_label']} (index {gap['start_idx']})")
            lines.append(f"    To:   {gap['end_label']} (index {gap['end_idx']})")
            lines.append(f"    Size: {gap['gap_size']:.1f} {unit}\n")

    lines.append("Impact on Temporal Visualization:")
    if has_gaps:
        lines.append(
            "  ✓ Plots show SEPARATE LINE SEGMENTS for each "
            "continuous period")
        lines.append("  ✓ No lines are drawn across gaps")
        lines.append("  ✓ Visual breaks indicate where data is missing")
        lines.append("\nPossible causes for gaps:")
        lines.append("  - Seasonal operation (system open only part of year)")
        lines.append("  - System maintenance or downtime")
        lines.append("  - Data collection interruptions")
        lines.append("  - Multi-phase study (Phase 1, break, Phase 2)")
    else:
        lines.append("  ✓ Plots show CONTINUOUS LINES connecting all points")
        lines.append("  ✓ All time periods are represented sequentially")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


# ============================================================================
# REPORTING & PUBLIC UTILITIES
# ============================================================================

def print_gap_report(graph_labels: List[str], gap_info: GapInfo,
                     unit: str = "months") -> None:
    """
    Print human-readable gap analysis report to console.

    Parameters
    ----------
    graph_labels : list of str
        Original temporal labels
    gap_info : dict
        Output from detect_temporal_gaps()
    unit : str, optional
        Time unit for reporting (default: "months")

    Returns
    -------
    None
    """
    if "report" in gap_info:
        print(gap_info["report"])
    else:
        # Fallback if gap_info doesn't have the report key
        print(_generate_gap_report_text(graph_labels, gap_info.get("gaps", []), unit))


def create_gap_dataframe(graph_labels: List[str],
                         gap_info: GapInfo) -> pd.DataFrame:
    """
    Create a DataFrame summarizing detected gaps.

    Parameters
    ----------
    graph_labels : list of str
        Original temporal labels
    gap_info : dict
        Output from detect_temporal_gaps()

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per gap, including:
        - start_label, end_label, gap_size
    """
    if not gap_info.get("gaps"):
        return pd.DataFrame(
            columns=["gap_number", "start_label", "end_label", "gap_size"])

    gap_rows = []
    for i, gap in enumerate(gap_info["gaps"], 1):
        gap_rows.append({
            "gap_number": i,
            "start_label": gap["start_label"],
            "end_label": gap["end_label"],
            "gap_size": gap["gap_size"],
            "start_idx": gap["start_idx"],
            "end_idx": gap["end_idx"],
        })

    return pd.DataFrame(gap_rows)


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def format_large_numbers(x: float, pos: int) -> str:
    """
    Format large numbers with appropriate units (k, M, B).

    Parameters
    ----------
    x : float
        The number to format
    pos : int
        The position

    Returns
    -------
    str
        Formatted number string
    """
    if x >= 1_000_000_000:
        return f'{x / 1_000_000_000:.1f}B'
    elif x >= 1_000_000:
        return f'{x / 1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x / 1_000:.1f}k'
    else:
        return f'{x:.0f}'


def plot_with_gap_handling(ax, graph_labels: List[str], y_values,
                           gap_segments: List[Tuple[int, int]],
                           **kwargs) -> None:
    """
    Plot data with proper handling of temporal gaps.

    Instead of drawing a continuous line across gaps, this function detects gaps
    and draws separate line segments for each continuous temporal period. This
    creates visual breaks where gaps occur, accurately representing the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    graph_labels : list of str
        Temporal labels (e.g., ["2024-03", "2024-04", "2024-11"])
    y_values : array-like
        Y-values to plot (one per label)
    gap_segments : list of tuple
        Continuous segments as (start_idx, end_idx) tuples
        (from detect_temporal_gaps()["segments"])
    **kwargs : dict, optional
        Additional keyword arguments passed to `ax.plot()`.
        Common options include `marker`, `linestyle`, `markersize`,
        `linewidth`, `color`, and `label`. Default styling is applied
        if these are not provided.

    Returns
    -------
    None

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from temporal_networks._gap_utilities import (
    ...     plot_with_gap_handling, detect_temporal_gaps)
    >>>
    >>> labels = ["2024-03", "2024-04", "2024-05", "2024-11", "2024-12"]
    >>> y_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    >>> gap_info = detect_temporal_gaps(labels)
    >>>
    >>> fig, ax = plt.subplots()
    >>> plot_with_gap_handling(ax, labels, y_values, gap_info["segments"])
    >>> _ = ax.set_title("Example with Gap")
    """
    # Set default plot styles
    default_kwargs = {
        'marker': 'o',
        'linestyle': '-',
        'markersize': 10,
        'linewidth': 2,
        'color': '#1f77b4'
    }

    # Extract label if provided, as it needs special handling
    label = kwargs.pop('label', None)

    # Update defaults with any user-provided kwargs
    plot_kwargs = {**default_kwargs, **kwargs}

    for i, (segment_start, segment_end) in enumerate(gap_segments):
        x_indices = np.arange(segment_start, segment_end)
        y_segment = [y_values[idx] for idx in x_indices]

        # Only add label for first segment (for legend)
        segment_kwargs = plot_kwargs.copy()
        if label and i == 0:
            segment_kwargs['label'] = label

        ax.plot(x_indices, y_segment, **segment_kwargs)

    # Set x-ticks and labels to show all data points
    ax.set_xticks(range(len(graph_labels)))
    ax.set_xticklabels(graph_labels, rotation=45, ha='right', fontsize=10)
