# temporal_networks

[![PyPI version](https://badge.fury.io/py/temporal-networks.svg)](https://badge.fury.io/py/temporal-networks)
[![CI](https://github.com/cadrianas/temporal_networks/actions/workflows/tests.yml/badge.svg)](https://github.com/cadrianas/temporal_networks/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/temporal-networks/badge/?version=latest)](https://temporal-networks.readthedocs.io)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python package for analyzing temporal network evolution with automatic
gap detection.

`temporal_networks` is designed for researchers who need to understand how
network structure and properties change over time. Unlike general-purpose
graph libraries such as [`networkx`](https://networkx.org/) or
[`igraph`](https://igraph.org/), it treats the temporal dimension as a
first-class feature: it can ingest raw event streams directly into snapshots,
measure how much structure persists between time steps, and characterise
whether activations are bursty or regular — all with native support for
irregular time series and missing data periods.

Compared to other temporal network packages such as
[`teneto`](https://teneto.readthedocs.io/) and
[`dynetx`](https://dynetx.readthedocs.io/), `temporal_networks` focuses
specifically on gap-aware analysis and visualization, automatically
detecting and correctly representing seasonal closures, maintenance windows,
or crisis-driven interruptions rather than drawing misleading continuous
lines across missing periods.

See the full documentation at
[temporal-networks.readthedocs.io](https://temporal-networks.readthedocs.io).

## Installation

```bash
pip install temporal_networks
```

Or install from source:

```bash
git clone https://github.com/cadrianas/temporal_networks
cd temporal_networks
pip install -e .
```

## Quick Start

The example below goes from a raw event table to three kinds of temporal
analysis in under 20 lines. The gap in May is detected automatically — no
special configuration needed.

```python
import random
import pandas as pd
import igraph as ig
from temporal_networks import (
    snapshots_from_events,
    snapshot_similarity,
    burstiness_coefficient,
)

# --- 1. Build snapshots from a raw event stream ---
events = pd.DataFrame({
    "time":   ["2024-01","2024-01","2024-02","2024-03",
               "2024-03","2024-04","2024-06","2024-06"],
    "source": ["A","B","A","B","A","A","B","A"],
    "target": ["B","C","C","C","C","B","C","C"],
})
# 2024-05 is absent from the data — gap is inferred automatically
graphs, labels = snapshots_from_events(
    events, time_col="time", source_col="source", target_col="target"
)
print(f"Snapshots: {labels}")
# ['2024-01', '2024-02', '2024-03', '2024-04', '2024-06']

# --- 2. How similar are consecutive snapshots? ---
sim = snapshot_similarity(graphs, graph_labels=labels, report_gaps=False)
print(sim[["Graph", "jaccard", "edge_persistence"]])
#      Graph  jaccard  edge_persistence
# 0  2024-02      0.0               0.0
# 1  2024-03      0.5               1.0
# 2  2024-04      0.0               0.0
# 3  2024-06      NaN               NaN   <- gap, not compared

# --- 3. Are edges bursty or regular? ---
bdf = burstiness_coefficient(
    graphs, graph_labels=labels, report_gaps=False
)
print(bdf[["entity", "n_events", "burstiness"]])
# Burstiness B: -1 = perfectly regular, 0 = Poisson-like, +1 = bursty
```

No files are written unless you pass `save_path`.

## What's in the package

### Data ingestion

| Function | Description |
|---|---|
| `snapshots_from_events()` | Build igraph snapshots from a long-form event DataFrame |
| `snapshots_from_edgelist()` | Load a timestamped CSV edge list into snapshots |

### Snapshot-level analysis

| Function | Description |
|---|---|
| `network_properties()` | Structural metrics (density, diameter, clustering, …) per snapshot |
| `calculate_centralities()` | 13 centrality measures tracked over time |
| `communities_measures()` | 7 community detection algorithms with evolution tracking |
| `vertex_properties()` | Single-node property trajectory |
| `compute_edge_dynamics()` | Formation and dissolution counts between consecutive snapshots |

### Temporal structure

| Function | Description |
|---|---|
| `snapshot_similarity()` | Jaccard, edge/node persistence, topological overlap between pairs |
| `temporal_correlation_coefficient()` | Sequence-level average topological correlation |
| `inter_event_times()` | Inter-event intervals (in inferred time units) per edge or node |
| `burstiness_coefficient()` | Goh–Barabási B ∈ [−1, 1] per edge or node |

### Utilities

| Function | Description |
|---|---|
| `detect_temporal_gaps()` | Standalone gap analysis and reporting |

See the [API reference](https://temporal-networks.readthedocs.io/api) for
full parameter documentation.

## Gap handling

All analysis functions accept a `graph_labels` list of date strings
(`"YYYY-MM"`, `"YYYY-MM-DD"`, `"YYYY-W##"`, `"YYYY-Q#"`, or `"YYYY"`).
The package infers the natural cadence from the label format and flags any
skipped period as a gap. Gap-straddling results are reported as `NaN` rather
than computed across missing data, and plots break the line at each gap so
no false continuity is implied.

## Tutorials

- [New features walkthrough](examples/example_specs_01_02_03.py) —
  ingestion, snapshot stability, and burstiness on a named-node sequence
  with a deliberate gap
- [Synthetic data walkthrough](examples/example_1_synthetic.py) —
  gap-aware analysis comparing continuous and seasonal datasets

## Reproducibility

To reproduce the synthetic results and gap-aware visualizations:

```bash
python examples/example_specs_01_02_03.py
python examples/example_1_synthetic.py
```

`example_specs_01_02_03.py` exercises the ingestion, stability, and
burstiness modules on a small reproducible dataset and prints all results
to stdout. `example_1_synthetic.py` runs the full suite of analyses on
hub-and-spoke networks and saves comparative visualizations to `plots/`.

## Citation

If you use `temporal_networks` in your research, please cite:

```bibtex
@software{ciupeanu_arino_temporal_networks,
  author    = {Ciupeanu, Adriana-Stefania and Arino, Julien},
  title     = {temporal\_networks: A Python package for analyzing
               temporal network evolution with automatic gap detection},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/cadrianas/temporal_networks},
  note      = {Version 0.1.0}
}
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
instructions on setting up a development environment and running the test
suite locally.

## License

GPL-3.0. See [LICENSE](LICENSE) for details.
