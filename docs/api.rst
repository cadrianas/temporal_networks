API Reference
=============

.. currentmodule:: temporal_networks

.. autosummary::

   network_properties
   calculate_centralities
   communities_measures
   vertex_properties
   compute_edge_dynamics
   edge_formation
   edge_dissolution
   plot_edge_dynamics
   plot_community_evolution
   detect_temporal_gaps
   snapshots_from_events
   snapshots_from_edgelist
   snapshot_similarity
   temporal_correlation_coefficient
   inter_event_times
   burstiness_coefficient
   temporal_reachability
   temporal_distances
   temporal_closeness
   temporal_efficiency
   temporal_betweenness
   detect_change_points
   flag_anomalous_snapshots

Data ingestion
--------------

.. autofunction:: snapshots_from_events

.. autofunction:: snapshots_from_edgelist

Network-level analysis
----------------------

.. autofunction:: network_properties

.. autofunction:: calculate_centralities

Community analysis
------------------

.. autofunction:: communities_measures

.. autofunction:: plot_community_evolution

Vertex-level analysis
---------------------

.. autofunction:: vertex_properties

Edge dynamics
-------------

.. autofunction:: compute_edge_dynamics

.. autofunction:: edge_formation

.. autofunction:: edge_dissolution

.. autofunction:: plot_edge_dynamics

Snapshot stability
------------------

.. autofunction:: snapshot_similarity

.. autofunction:: temporal_correlation_coefficient

Burstiness & inter-event analysis
---------------------------------

.. autofunction:: inter_event_times

.. autofunction:: burstiness_coefficient

Time-respecting paths
---------------------

.. note::
   ``cross_gaps=False`` (the default) is the key differentiator: paths cannot
   cross a detected temporal gap, so a data closure is not assumed to be
   transparent to transmission. Pass ``cross_gaps=True`` for standard
   contact-sequence behaviour.

.. autofunction:: temporal_reachability

.. autofunction:: temporal_distances

.. autofunction:: temporal_closeness

.. autofunction:: temporal_efficiency

.. autofunction:: temporal_betweenness

Change-point & anomaly detection
---------------------------------

.. autofunction:: detect_change_points

.. autofunction:: flag_anomalous_snapshots

Temporal gaps
-------------

.. autofunction:: detect_temporal_gaps
