# temporal_networks

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/cadrianas/temporal_networks/actions/workflows/ci.yml/badge.svg)](https://github.com/cadrianas/temporal_networks/actions)
[![Documentation Status](https://readthedocs.org/projects/temporal-networks/badge/?version=latest)](https://temporal-networks.readthedocs.io/en/latest/?badge=latest)

A Python package for analyzing temporal network evolution with **automatic gap detection**.

`temporal_networks` is designed for understanding how network structure and properties change over time. Unlike general-purpose libraries, it handles the temporal dimension natively, providing automatic detection and visualization of temporal gaps (missing data periods)—a critical feature for real-world datasets with seasonal closures, maintenance windows, or crisis-driven interruptions.

## Key Features

- **Automatic Gap Detection**: Seamlessly handles irregular time series and seasonal data.
- **Network Properties**: Track evolution of density, diameter, clustering, and connectivity.
- **Centrality Trajectories**: Compute 13 different centrality measures across time steps.
- **Community Evolution**: Track structural changes using 7 different algorithms (Leiden, Louvain, Walktrap, etc.).
- **Edge Dynamics**: Analyze formation and dissolution patterns between consecutive time points.
- **Vertex Tracking**: Follow individual node importance and local structure over time.
- **Gap-Aware Visualization**: Automatic generation of plots that correctly represent temporal discontinuities.

## Installation

```bash
git clone https://github.com/cadrianas/temporal_networks
cd temporal_networks
pip install -e .
```

## Quick Start

```python
import igraph as ig
from temporal_networks import network_properties

# Load or create your temporal graphs (12 months of data)
graphs = [ig.Graph.Barabasi(n=50, m=2) for _ in range(12)]
# Create labels for each month
labels = [f"2024-{i+1:02d}" for i in range(12)]

# Analyze properties with automatic gap detection
# Results are returned as a pandas DataFrame and plots are saved to 'plots/'
props = network_properties(graphs, graph_labels=labels, save_path="plots/")
```

## API Overview

The package provides several high-level functions for temporal analysis:

- `network_properties()`: Compute structural metrics for each network snapshot.
- `calculate_centralities()`: Track 13 different centrality measures over time.
- `communities_measures()`: Apply 7 community detection algorithms and track evolution.
- `vertex_properties()`: Track a specific node's properties through the temporal sequence.
- `edge_formation()` / `edge_dissolution()`: Analyze route changes between consecutive steps.
- `detect_temporal_gaps()`: Standalone utility for analyzing temporal discontinuities.

## Reproducibility

To reproduce the synthetic results and gap-aware visualizations demonstrated in our research:

```bash
python examples/example_1_synthetic.py
```
This script generates both continuous and gapped datasets, performs a full suite of analyses, and saves comparative visualizations to the `plots/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Testing locally

Run the full test suite to ensure no regressions:
```bash
python tests/run_tests.py
```

## Citation

If you use this package in your research, please cite:

Ciupeanu, A.-S., & Arino, J. (2026). temporal_networks: A Python package for analyzing temporal network evolution with automatic gap detection.

## Credits

Developed by **Adriana-Stefania Ciupeanu** and **Julien Arino** at the University of Manitoba.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
