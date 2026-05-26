"""
Temporal Gap Analysis and Reporting

This module provides comprehensive gap detection and reporting for temporal networks.
It re-exports consolidated logic from _gap_utilities for backward compatibility
and provides high-level gap analysis tools.
"""

from ._gap_utilities import (
    parse_flexible_datetime,
    calculate_time_difference,
    detect_temporal_gaps,
    print_gap_report,
    create_gap_dataframe
)

__all__ = [
    "parse_flexible_datetime",
    "calculate_time_difference",
    "detect_temporal_gaps",
    "print_gap_report",
    "create_gap_dataframe",
]
