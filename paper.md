---
title: 'temporal_networks: A Python package for analyzing temporal network evolution with automatic gap detection'
tags:
  - Python
  - temporal networks
  - network analysis
  - temporal gaps
  - time series networks
  - community detection
  - centrality measures
authors:
  - name: Adriana-Stefania Ciupeanu
    orcid: 0000-0003-0833-2176
    equal-contrib: true
    affiliation: 1
  - name: Julien Arino
    orcid: 0000-0001-6409-5027
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: Department of Mathematics, University of Manitoba, Winnipeg, MB, Canada
    index: 1
date: 9 March 2026
bibliography: paper.bib
---

# Summary

`temporal_networks` is a Python package for **temporal network analysis** — the
study of how network structure and properties evolve over time [@holme2012temporal].
A temporal network is represented as an ordered sequence of graph snapshots, each
carrying a time label (for example `2024-03`). Unlike general-purpose graph
libraries such as `python-igraph` [@csardi2006igraph] and NetworkX
[@hagberg2008networkx], which operate on a single static graph at a time, the
package treats time as a first-class dimension: it ingests raw event streams into
snapshots, tracks node identity across snapshots whose vertex order may differ, and
computes temporal metrics that summarise change over the whole sequence.

Its distinguishing feature is the explicit, automatic handling of **temporal gaps**
— periods that are missing from the observation window (seasonal closures,
maintenance windows, crisis-driven interruptions, or multi-phase study designs).
The package detects gaps directly from the time labels and propagates that
information into every metric and plot, so that visualisations show genuine breaks
rather than drawing misleading continuous lines across the missing interval. It
provides per-snapshot structural properties, node-level centralities, seven
community-detection algorithms and a community-tracking layer, vertex trajectory
tracking, edge formation/dissolution dynamics, inter-event burstiness, temporal
reachability and betweenness metrics over time-respecting paths, and change-point
and anomaly detection across the sequence.

# Statement of Need

Temporal network analysis is increasingly central to understanding dynamic systems
across domains: epidemiological contact networks observed across phases with
different containment measures, transportation networks with seasonal service or
crisis-driven disruptions, infrastructure systems with maintenance windows, and
biological or social networks observed across discrete stages. Yet the analysis of
a *sequence* of networks differs fundamentally from the analysis of a single graph,
and widely used tools do not close that gap on their own.

## The temporal-paradigm problem

Libraries such as `python-igraph` [@csardi2006igraph] and NetworkX
[@hagberg2008networkx] are designed around individual graphs. Given a temporal
sequence, a researcher must build the temporal scaffolding themselves: align node
identity across snapshots, track a metric across time, aggregate per-snapshot
results, and produce trajectory plots. Dedicated temporal-network tools exist —
`teneto` targets time-varying connectivity with a focus on neuroimaging
[@thompson2017teneto], and `dynetx` models dynamic graphs as a stream of
interactions [@rossetti2020dynetx] — but neither makes the handling of *missing
observation periods* an explicit, default part of every metric and visualisation,
which is the recurring practical obstacle in applied temporal-network work.

## The concrete failure mode: false continuity across gaps

A subtle but consequential problem arises in visualisation. When a metric is
plotted against snapshot index, naive plotting connects consecutive snapshots even
when a real temporal gap separates them. A bike-sharing network operating
March–August and again November–February, for instance, has a three-month
operational gap; an index-based line plot draws an August-to-November segment that
implies a transition which never occurred. The same issue distorts any trajectory
metric and any animation. Standard graph libraries do not guard against it — the
burden falls on the user to detect the gap and special-case the plotting.

## The approach taken here

`temporal_networks` makes gap handling automatic and transparent:

- **Time-aware by construction.** Snapshot labels (`2024-03`, `2024-08`,
  `2024-11`) drive the analysis rather than serving as decoration.
- **Automatic gap detection** from the labels, with support for monthly, daily,
  weekly, quarterly, and annual formats and a configurable threshold.
- **Gap-aware plotting** in every visualisation: continuous segments are drawn
  separately, so missing periods appear as breaks.
- **Identity-aware metrics** that match vertices by name across snapshots whose
  vertex order may be shuffled, so node-level trajectories remain correct.
- **Transparent reporting** that prints where gaps occur and how they affect
  interpretation, rather than silently hiding them.

This yields a systematic workflow in which gaps are handled correctly by default
and the user is always told about the structure of their data.

# Functionality

The public API is organised into ingestion, per-snapshot metrics, time-spanning
metrics, change detection, and visualisation. All names below are exported from the
top-level package.

**Ingestion and gap detection.** `snapshots_from_events` and
`snapshots_from_edgelist` build a snapshot sequence from a raw event table or
edge list, binning by a chosen time frequency. `detect_temporal_gaps` parses the
resulting labels, computes the spacing between consecutive snapshots, and returns
the gaps together with the continuous segments used downstream for gap-aware
plotting.

**Per-snapshot structure and centrality.** `network_properties` returns
structural metrics per snapshot (counts, density, diameter, average path length,
clustering, reciprocity, components). `calculate_centralities` computes a suite of
node-level centralities (degree, closeness, betweenness, eigenvector, PageRank,
harmonic, eccentricity, Burt's constraint, hub and authority scores) for every
snapshot, and `vertex_properties` tracks the full trajectory of a named node.

**Communities.** `communities_measures` applies seven community-detection
algorithms — including Leiden [@traag2019leiden], Louvain [@blondel2008louvain],
Walktrap, fast-greedy, label propagation, spinglass, and Infomap — to each
snapshot and reports assignments, summary statistics, and modularity.
`track_communities` and `plot_community_lineage` follow communities across
snapshots to expose merges, splits, births, and deaths.

**Edge and stability dynamics.** `compute_edge_dynamics`, `edge_formation`,
`edge_dissolution`, and `plot_edge_dynamics` quantify which edges appear and
disappear between consecutive snapshots. `snapshot_similarity` and
`temporal_correlation_coefficient` measure how much structure persists from one
snapshot to the next.

**Burstiness.** `inter_event_times` and `burstiness_coefficient` characterise
whether activity is regular, Poisson-like, or bursty, using the burstiness measure
of @goh2008burstiness.

**Time-respecting paths.** `temporal_reachability`, `temporal_distances`,
`temporal_closeness`, `temporal_efficiency`, and `temporal_betweenness` operate
over time-ordered paths, so that reachability and brokerage respect the arrow of
time rather than treating the aggregated graph as static.

**Change and anomaly detection.** `detect_change_points` locates structural shifts
in a metric trajectory (optionally using the `ruptures` change-point methods of
@truong2020ruptures), and `flag_anomalous_snapshots` highlights snapshots that
deviate from the surrounding sequence.

**Gap-aware visualisation.** Every plotting routine — including
`plot_community_evolution`, `plot_edge_dynamics`, and `plot_community_lineage` —
draws separate line segments for each continuous period, so visual breaks indicate
missing data rather than implying continuity across a gap.

# Example usage

```python
import random
import pandas as pd
import igraph as ig
from temporal_networks import (
    snapshots_from_events,
    snapshot_similarity,
    burstiness_coefficient,
)

# A raw event stream; the month 2024-05 is absent from the data.
events = pd.DataFrame({
    "time":   ["2024-01", "2024-01", "2024-02", "2024-03",
               "2024-03", "2024-04", "2024-06", "2024-06"],
    "source": ["A", "B", "A", "B", "A", "A", "B", "A"],
    "target": ["B", "C", "C", "C", "C", "B", "C", "C"],
})

# Build snapshots; the May gap is inferred automatically from the labels.
graphs, labels = snapshots_from_events(
    events, time_col="time", source_col="source", target_col="target"
)

# How much structure persists between consecutive snapshots?
sim = snapshot_similarity(graphs, graph_labels=labels)

# Is activity bursty or regular?
bdf = burstiness_coefficient(graphs, graph_labels=labels)
```

The similarity and burstiness tables returned here leave the across-gap pair
uncompared (reported as `NaN`) rather than fabricating a transition across the
missing month. Stochastic examples in the documentation seed both `random` and the
igraph RNG so that results are reproducible.

# Quality control

The package ships with a `pytest` suite of more than 150 unit tests across 14
modules, exercised in continuous integration on Python 3.9–3.12, together with
`ruff` linting and `mypy` type checking. An end-to-end integration example passes a
single, deliberately non-trivial synthetic temporal network — with named vertices
whose order is shuffled between snapshots, a genuine temporal gap, and injected
structural change — through every public function and asserts hand-checkable
invariants, providing an integration smoke test that complements the isolated unit
tests.

# Motivating application domains

The design is driven by recurring patterns in applied temporal-network analysis,
including epidemiological contact networks observed across containment phases,
transportation networks with seasonal or crisis-driven service gaps, infrastructure
systems with maintenance windows, and biological or social networks observed across
discrete stages. The synthetic examples distributed with the package illustrate
these patterns; they are designed to be reproducible rather than to stand in for
validated empirical datasets.

# Limitations and future work

Current limitations include reduced effectiveness of interactive HTML animations on
heavily gapped data (static plots are recommended), the requirement that some
community-detection algorithms operate on undirected graphs, and performance
considerations on very large graphs. Planned work includes parallel processing for
long sequences, support for heterogeneous node and edge types, richer interactive
gap annotation, temporal-motif analysis, and optional gap imputation.

# Availability

`temporal_networks` is released under the GNU General Public License v3.0 (GPL-3.0)
and developed openly on GitHub, with documentation hosted on Read the Docs. It can
be installed from source with `pip install -e .`; once published, it will also be
available from PyPI as `pip install temporal_networks`.

# Acknowledgments

The authors thank colleagues at the University of Manitoba for feedback on the
package design and documentation.

# References
