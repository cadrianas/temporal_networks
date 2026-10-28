---
title: 'temporal_networks: Gap-aware analysis of evolving networks in Python'
tags:
  - Python
  - temporal networks
  - network analysis
  - temporal gaps
  - community detection
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
date: 3 September 2026
bibliography: paper.bib
---

# Summary

Networks observed over time [@holme2012temporal] arise in studies of social
contact, disease transmission, transportation, infrastructure, and biological
systems. Their analysis requires more than applying a static graph measure
repeatedly: node identity must remain aligned, paths must respect time order, and
missing observation periods must not be mistaken for continuous evolution.

`temporal_networks` is an open-source Python package for analysing an ordered
sequence of graph snapshots. It constructs snapshots from event tables, measures
structural and node-level change, tracks communities and edges, detects anomalies,
and computes burstiness and time-respecting path metrics. Its defining feature is
automatic treatment of temporal gaps. Labels such as dates, weeks, quarters, or
years are used to identify missing periods relative to an inferred or specified
cadence. Gap-aware comparisons are reported as missing at these boundaries, and
associated time-series plots are split into continuous segments. This helps avoid
presenting an unobserved interval as an observed transition.

# Statement of need

Researchers commonly receive network data as timestamped interactions or as a
sequence of snapshots. A complete workflow then requires event binning, consistent
node identifiers, repeated structural calculations, temporal comparisons, and
figures that accurately represent the observation schedule. General-purpose graph
libraries provide many of the underlying algorithms; users may still need
additional code to coordinate snapshot-based analyses. Such code can become
study-specific and may inadvertently connect observations across periods in which
no data were collected.

`temporal_networks` is intended for researchers and students who work with
discrete-time network data and need a reproducible analysis pipeline, particularly
when observation periods are irregular. Missing periods occur naturally in contact
studies interrupted by holidays, epidemiological surveillance conducted in phases,
seasonal transportation systems, and infrastructure data affected by maintenance.
Treating a gap as an ordinary pair of adjacent rows can bias persistence measures,
infer time-respecting paths unsupported by the observations, and draw false continuity in a figure.
The package makes the observation schedule part of the computation rather than a
plot annotation added after analysis. Gap detection relies on the supplied labels
and cadence; users must distinguish missing observations from observed periods
with no interactions.

# State of the field

NetworkX [@hagberg2008networkx] and `python-igraph` [@csardi2006igraph] provide
broad collections of algorithms for individual graphs. `temporal_networks` builds
on `python-igraph` rather than replacing it: igraph performs the graph algorithms,
while this package supplies ingestion, identity alignment, gap-aware sequencing,
tabular results, and temporal visualisation.

Dedicated temporal-network tools make different representational choices.
`DyNetx` models interactions in dynamic graphs [@rossetti2020dynetx], while
`teneto` provides temporal-network measures with particular strengths in
time-varying connectivity and neuroimaging [@thompson2017teneto]. These tools
support a range of temporal-network workflows. `temporal_networks` focuses on
snapshot-based analyses in which missing observation periods need to be handled
consistently across ingestion, pairwise comparisons, trajectories, change
detection, and plots. It provides a compact workflow in which gap semantics and
node identity are shared across these analyses.

# Software design

The package represents a temporal network as a sequence of igraph graphs plus a
corresponding sequence of time labels. This design retains compatibility with
igraph algorithms and lets users bring existing graphs without converting to a
new container type. The trade-off is that sequence validation and temporal
metadata must be applied consistently. Central validation utilities therefore
check graph-label alignment, infer the cadence of supported label formats, and
identify continuous index segments for gap-aware analyses.

Cross-snapshot comparisons use vertex names or labels when available. When
neither attribute is present, they fall back to vertex indices, which requires
consistent vertex ordering across snapshots. Gap-aware pairwise metrics return
`NaN` at a gap boundary instead of computing a value across it, and
time-respecting path algorithms can exclude transitions across such boundaries.
This conservative choice distinguishes "not observed" from a numerical zero.

Most analysis functions return pandas data frames so results can be inspected,
joined, and plotted in ordinary scientific Python workflows. File output is opt-in for
analysis functions. The public API is divided into interoperable modules:

- event-table and edge-list ingestion;
- per-snapshot structure, centrality, and vertex trajectories;
- edge dynamics, snapshot stability, and inter-event burstiness [@goh2008burstiness];
- seven community-detection methods, including Leiden [@traag2019leiden] and
  Louvain [@blondel2008louvain], plus community lineage tracking;
- temporal reachability, distance, closeness, efficiency, and betweenness; and
- change-point and anomaly detection, with optional `ruptures` methods [@truong2020ruptures].

Optional dependencies are kept separate where possible, while the core relies on
widely used NumPy, pandas, igraph, Matplotlib, and Plotly components.

## Reproducibility and documentation

The repository provides reproducible materials demonstrating two research
workflows using synthetic data: a school contact-network walkthrough ingesting
simulated proximity events and
tracking communities around an unobserved holiday, and an epidemic
agent-based-modelling walkthrough analysing contact-tracing interruptions and
temporal reachability. Both examples are seeded and contain assertions about their results.
Reproducibility tests run each example twice and compare its console output;
accompanying tutorials are included in the documentation. 

An additional end-to-end integration test runs a sequence through every public
function to verify invariants across brokers, anomalies, and gaps. The repository includes 225 automated tests and a continuous integration
configuration for Python 3.9--3.12, linting, static type checking, and documentation
builds.

# AI usage disclosure

Generative-AI tools were not used to autonomously generate the package or to make
scientific and architectural decisions. They were used as auditing and review
assistants. Google Jules proposed candidate code modifications during an audit;
defects introduced were subsequently inspected, corrected, and retested by a
human author. Anthropic Claude assisted with code review and suggested a synthetic
integration example for exercising interactions among package functions. All
modifications retained in the repository were verified and revised by human
authors, who retain full responsibility for the accuracy, licensing, and final
submitted work. OpenAI Codex also assisted with manuscript editing, reference
checking, and document preparation.

# Acknowledgements

The authors thank colleagues at the University of Manitoba for feedback on the
package design and documentation.

# References


