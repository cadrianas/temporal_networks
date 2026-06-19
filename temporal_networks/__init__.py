"""
temporal_networks: A Python package for analyzing temporal network evolution
with automatic gap detection.
"""

__version__ = "0.1.0"
__author__ = "Adriana-Stefania Ciupeanu, Julien Arino"

__all__ = [
    "network_properties",
    "calculate_centralities",
    "communities_measures",
    "vertex_properties",
    "compute_edge_dynamics",
    "edge_formation",
    "edge_dissolution",
    "plot_edge_dynamics",
    "plot_community_evolution",
    "detect_temporal_gaps",
    "snapshots_from_events",
    "snapshots_from_edgelist",
    "snapshot_similarity",
    "temporal_correlation_coefficient",
    "inter_event_times",
    "burstiness_coefficient",
    "temporal_reachability",
    "temporal_distances",
    "temporal_closeness",
    "temporal_efficiency",
    "temporal_betweenness",
    "detect_change_points",
    "flag_anomalous_snapshots",
    "track_communities",
    "plot_community_lineage",
]

from .network_properties import network_properties
from .calculate_centralities import calculate_centralities
from .communities_measures import communities_measures
from .vertex_properties import vertex_properties
from .edge_formation_dissolution import (
    compute_edge_dynamics,
    edge_formation,
    edge_dissolution,
    plot_edge_dynamics,
)
from .plot_community_evolution import plot_community_evolution
from ._gap_utilities import detect_temporal_gaps
from .io import snapshots_from_events, snapshots_from_edgelist
from .stability import (
    snapshot_similarity,
    temporal_correlation_coefficient,
)
from .burstiness import (
    inter_event_times,
    burstiness_coefficient,
)
from .temporal_paths import (
    temporal_reachability,
    temporal_distances,
    temporal_closeness,
    temporal_efficiency,
    temporal_betweenness,
)
from .change_points import (
    detect_change_points,
    flag_anomalous_snapshots,
)
from .community_tracking import (
    track_communities,
    plot_community_lineage,
)
