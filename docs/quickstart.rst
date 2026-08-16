Quickstart
==========

The example below analyzes twelve monthly network snapshots whose labels
contain a gap between June and October. The gap is detected and handled
automatically.

.. code-block:: python

   import random
   import igraph as ig
   from temporal_networks import network_properties

   # Seed igraph's RNG so the snapshots are reproducible
   # (random.seed alone does NOT affect igraph's generators)
   ig.set_random_number_generator(random.Random(42))

   # Create 12 monthly network snapshots with a gap in summer
   graphs = [ig.Graph.Barabasi(n=50, m=2) for _ in range(12)]
   labels = ["2024-01", "2024-02", "2024-03",
             "2024-04", "2024-05", "2024-06",
             "2024-10", "2024-11", "2024-12",
             "2025-01", "2025-02", "2025-03"]

   # Analyze structural properties across time steps.
   # Gap between June and October is detected and handled automatically.
   props = network_properties(graphs, graph_labels=labels)
   print(props)

This prints a :class:`pandas.DataFrame` with one row per time step and
columns for density, diameter, clustering coefficient, and connectivity.
No files are written unless you pass ``save_path``.

Going further
-------------

- :func:`temporal_networks.calculate_centralities` tracks 13 centrality
  measures for every node across time steps.
- :func:`temporal_networks.communities_measures` applies up to 7 community
  detection algorithms and tracks community evolution.
- :func:`temporal_networks.vertex_properties` follows a single node's
  importance and local structure over time.
- :func:`temporal_networks.compute_edge_dynamics` analyzes edge formation
  and dissolution between consecutive snapshots.
- :func:`temporal_networks.detect_temporal_gaps` is the standalone gap
  analysis utility used throughout the package.

Worked examples
---------------

Two end-to-end tutorials apply the package to a full dataset rather than a
single function:

- :doc:`social_contacts` — a face-to-face proximity study with an
  unobserved holiday break: ingestion from a contact log, community
  tracking across the break, and burstiness of ties.
- :doc:`epidemic_abm` — contact data from an agent-based SIR model with a
  suspended tracing programme: change-point detection around an
  intervention, and time-respecting paths that do not assume the
  unobserved period away.

See the :doc:`api` for full parameter documentation.
