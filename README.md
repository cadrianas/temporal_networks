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
feature, with native support for irregular time series and 
missing data periods. Compared to other temporal network packages such as 
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

```python
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
```

This prints a pandas DataFrame with one row per time step and columns for 
density, diameter, clustering coefficient, and connectivity. No files are 
written unless you pass `save_path`.

## API Overview

| Function | Description |
|---|---|
| `network_properties()` | Structural metrics for each network snapshot |
| `calculate_centralities()` | 13 centrality measures tracked over time |
| `communities_measures()` | 7 community detection algorithms with evolution tracking |
| `vertex_properties()` | Single-node property trajectory |
| `compute_edge_dynamics()` | Formation and dissolution patterns between snapshots |
| `detect_temporal_gaps()` | Standalone gap analysis utility |

See the [API reference](https://temporal-networks.readthedocs.io/api) for 
full parameter documentation.

## Tutorials

- [Synthetic data walkthrough](examples/example_1_synthetic.py) -- 
  gap-aware analysis on generated data
- Further tutorials are forthcoming.

## Reproducibility

To reproduce the synthetic results and gap-aware visualizations:

```bash
python examples/example_1_synthetic.py
```

This generates both continuous and gapped datasets, runs a full suite of 
analyses, and saves comparative visualizations to the `plots/` directory.

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
