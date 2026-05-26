"""
temporal_networks: A Python package for analyzing temporal network evolution with automatic gap detection.
"""

from .network_properties import network_properties
from .calculate_centralities import calculate_centralities
from .communities_measures import communities_measures
from .vertex_properties import vertex_properties
from .edge_formation_dissolution import edge_formation, edge_dissolution
from .plot_community_evolution import plot_community_evolution
from .temporal_gap_analysis import detect_temporal_gaps

__version__ = "0.1.0"
__author__ = "Adriana-Stefania Ciupeanu, Julien Arino"

__all__ = [
    "network_properties",
    "calculate_centralities",
    "communities_measures",
    "vertex_properties",
    "edge_formation",
    "edge_dissolution",
    "plot_community_evolution",
    "detect_temporal_gaps",
]
