"""
Temporal Network Analysis: Change-point & Anomaly Detection (GAP-AWARE)

This module flags snapshots where a network metric shifts structurally.
It complements gap detection — gaps are *missing* data, change-points are
*structural change* in the data that is present.

KEY FEATURES:
- Operates on the output DataFrame of any existing analysis function
  (e.g. ``network_properties()``, ``snapshot_similarity()``), so it
  composes with the package rather than recomputing metrics.
- Two dependency-free methods: z-score (spike detection) and first-
  difference MAD (step/level-shift detection).
- Optional PELT method via the ``ruptures`` package (install with
  ``pip install temporal_networks[changepoint]``).
- Gap-aware: never computes a score across a detected gap boundary, so a
  data closure is not flagged as an anomaly.
"""

import warnings

import igraph as ig
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ._gap_utilities import (
    GapInfo,
    detect_temporal_gaps,
    validate_and_setup_graphs,
)
from .network_properties import network_properties

__all__ = [
    "detect_change_points",
    "flag_anomalous_snapshots",
]

_OUTPUT_COLS = ["column", "index", "label", "score", "method"]

# Normalising constant: MAD × 1.4826 ≈ std for a normal distribution.
_MAD_SCALE = 1.4826


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _zscore_flags(series: np.ndarray, segments: List[Tuple[int, int]],
                  threshold: float) -> List[Dict[str, float]]:
    """
    Flag indices whose per-segment z-score exceeds ``threshold``.

    Parameters
    ----------
    series : numpy.ndarray
        1-D float array of metric values.
    segments : list of tuple
        ``(start, end)`` index pairs for each continuous segment.
    threshold : float
        Absolute z-score threshold.

    Returns
    -------
    list of dict
        Each dict has keys ``index`` and ``score``.
    """
    flags = []
    for start, end in segments:
        seg = series[start:end]
        if len(seg) < 2 or np.all(np.isnan(seg)):
            continue
        mean = float(np.nanmean(seg))
        std = float(np.nanstd(seg))
        if std == 0:
            continue
        with np.errstate(invalid="ignore"):
            z = (seg - mean) / std
        for offset, zval in enumerate(z):
            if np.isnan(zval):
                continue
            if abs(zval) > threshold:
                flags.append({"index": start + offset, "score": float(zval)})
    return flags


def _diff_mad_flags(series: np.ndarray, segments: List[Tuple[int, int]],
                    threshold: float) -> List[Dict[str, float]]:
    """
    Flag indices where the first-difference is a MAD outlier.

    Within each segment, first differences are computed and scored as
    ``|diff - median| / (1.4826 * MAD)``. Points whose score exceeds
    ``threshold`` are flagged at the *later* index of the pair. Scores are
    NaN when MAD = 0 (perfectly regular differences within the segment).

    Parameters
    ----------
    series : numpy.ndarray
        1-D float array of metric values.
    segments : list of tuple
        ``(start, end)`` index pairs for each continuous segment.
    threshold : float
        Score threshold (in normalised MAD units, equivalent to z-score
        for normal distributions).

    Returns
    -------
    list of dict
        Each dict has keys ``index`` and ``score``.
    """
    flags = []
    for start, end in segments:
        seg = series[start:end]
        if len(seg) < 2:
            continue
        diffs = np.diff(seg.astype(float))
        median = float(np.nanmedian(diffs))
        mad = float(np.nanmedian(np.abs(diffs - median)))
        if mad == 0:
            # Fall back to std when MAD is zero (e.g. a single step in an
            # otherwise flat series — most diffs are 0, MAD collapses).
            std = float(np.nanstd(diffs))
            if std == 0:
                continue
            denom = _MAD_SCALE * std
        else:
            denom = _MAD_SCALE * mad
        scores = np.abs(diffs - median) / denom
        for offset, score in enumerate(scores):
            if np.isnan(score):
                continue
            if score > threshold:
                # diff[i] is the change *to* index start+i+1
                flags.append({"index": start + offset + 1,
                               "score": float(score)})
    return flags


def _pelt_flags(series: np.ndarray, segments: List[Tuple[int, int]],
                threshold: float) -> List[Dict[str, float]]:
    """
    Detect change points with PELT (requires the ``ruptures`` package).

    Parameters
    ----------
    series : numpy.ndarray
        1-D float array of metric values.
    segments : list of tuple
        ``(start, end)`` index pairs for each continuous segment.
    threshold : float
        Penalty value forwarded to ``ruptures.Pelt.predict(pen=threshold)``.

    Returns
    -------
    list of dict
        Each dict has keys ``index`` and ``score`` (``NaN`` for PELT).

    Raises
    ------
    ImportError
        When ``ruptures`` is not installed.
    """
    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError(
            "The 'pelt' method requires the 'ruptures' package. "
            "Install it with: pip install temporal_networks[changepoint]"
        )

    flags = []
    for start, end in segments:
        seg = series[start:end]
        if len(seg) < 2:
            continue
        try:
            algo = rpt.Pelt(model="rbf").fit(seg.reshape(-1, 1))
            cps = algo.predict(pen=threshold)
            # ruptures always appends len(seg) as the last element
            for cp in cps:
                abs_idx = start + cp
                if abs_idx < end:
                    flags.append({"index": abs_idx,
                                  "score": float("nan")})
        except Exception as e:
            warnings.warn(
                f"PELT failed for segment [{start}, {end}): {e}")
    return flags


def _segments_for_frame(series_df: pd.DataFrame, label_col: str,
                        gap_info: Optional[GapInfo]
                        ) -> List[Tuple[int, int]]:
    """
    Map detected temporal gaps to row positions of ``series_df``.

    ``gap_info["segments"]`` indexes the original *label sequence*, which
    does not necessarily align with the rows of ``series_df`` — e.g.
    ``snapshot_similarity`` output has one row per consecutive pair, so it
    is one row shorter and starts at the second label. Segments are
    therefore re-derived from ``label_col``: every row whose label is the
    first label after a detected gap starts a new segment.

    Parameters
    ----------
    series_df : pandas.DataFrame
        The frame being analysed.
    label_col : str
        Column holding the snapshot label for each row.
    gap_info : dict or None
        Output of :func:`~temporal_networks.detect_temporal_gaps`.

    Returns
    -------
    list of tuple
        ``(start, end)`` row-index pairs covering ``series_df``. A single
        ``(0, len(series_df))`` segment when there is nothing to split on.
        When the gaps carry no matchable labels (or ``label_col`` is
        absent), falls back to ``gap_info["segments"]`` clipped to the
        frame length.
    """
    n = len(series_df)
    if gap_info is None:
        return [(0, n)]

    gaps = gap_info.get("gaps") or []
    if gaps and label_col in series_df.columns:
        end_labels = {g["end_label"] for g in gaps}
        labels = list(series_df[label_col])
        boundaries = [i for i, lab in enumerate(labels)
                      if lab in end_labels and i > 0]
        segments = []
        start = 0
        for b in boundaries:
            segments.append((start, b))
            start = b
        segments.append((start, n))
        return segments

    # No gap labels to match on: use the provided index segments, clipped
    # to the frame length so a shorter frame cannot over-slice.
    segments = [(s, min(e, n))
                for s, e in gap_info.get("segments", [(0, n)]) if s < n]
    return segments or [(0, n)]


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

def detect_change_points(
    series_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "zscore",
    threshold: float = 3.0,
    label_col: str = "Graph",
    gap_info: Optional[GapInfo] = None,
) -> pd.DataFrame:
    """
    Detect change points in one or more metric time series.

    Analyses the output DataFrame of any temporal-network analysis function
    (e.g. :func:`~temporal_networks.network_properties`,
    :func:`~temporal_networks.snapshot_similarity`) and flags snapshots
    where a metric shifts unusually.

    **Gap-aware:** when ``gap_info`` is supplied, rolling statistics and
    first differences are computed independently within each continuous
    segment so a data closure is never mistaken for an anomaly.

    Parameters
    ----------
    series_df : pandas.DataFrame
        Input data, typically the output of an analysis function. Must
        contain a label column (see ``label_col``) and at least one
        numeric column.
    columns : list of str, optional
        Numeric columns to analyse. If None (default), all numeric columns
        except ``label_col`` are used.
    method : str, optional
        Detection method:

        - ``"zscore"`` (default): flag points whose per-segment z-score
          exceeds ``threshold``.
        - ``"diff"``: flag points where the first difference is a MAD
          outlier exceeding ``threshold`` normalised MAD units.
        - ``"pelt"``: PELT change-point detection via the ``ruptures``
          package (must be installed separately; raises ``ImportError``
          with an install hint otherwise).
    threshold : float, optional
        Detection threshold. For ``"zscore"`` and ``"diff"``, the standard
        3-sigma rule (``threshold=3.0``) is a common starting point. For
        ``"pelt"``, this is the penalty value forwarded to
        ``ruptures.Pelt.predict``.
    label_col : str, optional
        Column in ``series_df`` that contains the snapshot label (default
        ``"Graph"``). Used to populate the ``label`` column in the output.
    gap_info : dict, optional
        Gap information from :func:`~temporal_networks.detect_temporal_gaps`.
        When provided, statistics are reset at each gap boundary. Gap
        boundaries are matched to rows through ``label_col``, so frames
        that do not have one row per label (e.g.
        :func:`~temporal_networks.snapshot_similarity` output, which has
        one row per consecutive pair) are still split at the correct rows.

    Returns
    -------
    pandas.DataFrame
        One row per detected change point with columns:

        - ``column``: name of the metric where the change was detected
        - ``index``: row index in ``series_df``
        - ``label``: value from ``label_col`` at that row
        - ``score``: detection score (z-score, normalised MAD multiple, or
          ``NaN`` for PELT)
        - ``method``: the detection method used

        Returns an empty DataFrame with the correct columns when no change
        points are found.

    Examples
    --------
    >>> import pandas as pd
    >>> from temporal_networks import detect_change_points
    >>> values = [1.0] * 5 + [20.0] + [1.0] * 5
    >>> df = pd.DataFrame({"Graph": [str(i) for i in range(11)],
    ...                    "metric": values})
    >>> cp = detect_change_points(df, threshold=2.0)
    >>> len(cp) > 0
    True
    >>> cp.loc[0, "column"]
    'metric'
    """
    if method not in ("zscore", "diff", "pelt"):
        raise ValueError(
            f"method must be 'zscore', 'diff', or 'pelt'; got {method!r}")

    if columns is None:
        numeric = series_df.select_dtypes(include=np.number).columns.tolist()
        columns = [c for c in numeric if c != label_col]

    # Segments are aligned to the ROWS of series_df via label_col, so
    # frames that do not have one row per label (e.g. snapshot_similarity
    # output, which starts at the second label) are still split at the
    # right rows.
    segments = _segments_for_frame(series_df, label_col, gap_info)

    rows = []
    for col in columns:
        if col not in series_df.columns:
            continue
        series = series_df[col].values.astype(float)

        try:
            if method == "zscore":
                flags = _zscore_flags(series, segments, threshold)
            elif method == "diff":
                flags = _diff_mad_flags(series, segments, threshold)
            else:
                flags = _pelt_flags(series, segments, threshold)
        except ImportError:
            raise
        except (ValueError, TypeError) as e:
            warnings.warn(
                f"Error detecting change points in '{col}': {e}")
            continue

        for flag in flags:
            idx = flag["index"]
            label = (series_df[label_col].iloc[idx]
                     if label_col in series_df.columns
                     else str(idx))
            rows.append({
                "column": col,
                "index": idx,
                "label": label,
                "score": flag["score"],
                "method": method,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return df.sort_values(["column", "index"]).reset_index(drop=True)


def flag_anomalous_snapshots(
    graphs: List[ig.Graph],
    graph_labels: Optional[List[str]] = None,
    method: str = "zscore",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect structurally anomalous snapshots from a graph sequence.

    Convenience wrapper: computes :func:`~temporal_networks.network_properties`
    internally and passes the result to :func:`detect_change_points`. Gap
    information is inferred from the labels and used so that metric jumps
    at gap boundaries are not falsely flagged.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Temporal snapshots in chronological order.
    graph_labels : list of str, optional
        Labels for each snapshot. If None, defaults to "Graph 1", etc.
    method : str, optional
        Detection method passed through to :func:`detect_change_points`
        (``"zscore"`` by default).
    threshold : float, optional
        Detection threshold passed through to :func:`detect_change_points`.

    Returns
    -------
    pandas.DataFrame
        Same schema as :func:`detect_change_points`: one row per detected
        anomalous snapshot per metric.

    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import flag_anomalous_snapshots
    >>> g_normal = ig.Graph.Full(n=5)
    >>> g_sparse = ig.Graph(n=5, edges=[(0, 1)])
    >>> graphs = [g_normal] * 8 + [g_sparse] + [g_normal] * 8
    >>> labels = [f"{2024 + i // 12}-{i % 12 + 1:02d}" for i in range(17)]
    >>> flags = flag_anomalous_snapshots(graphs, graph_labels=labels,
    ...                                   threshold=2.0)
    >>> len(flags) > 0
    True
    """
    graph_labels = validate_and_setup_graphs(graphs, graph_labels,
                                             min_length=1)
    gap_info = detect_temporal_gaps(graph_labels)
    props = network_properties(graphs, graph_labels=graph_labels,
                               visualisation=False, report_gaps=False)
    return detect_change_points(props, method=method, threshold=threshold,
                                label_col="Graph", gap_info=gap_info)


