# temporal_networks

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

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
git clone https://github.com/YOUR_USERNAME/temporal_networks.git
cd temporal_networks
pip install -e .
```

## Quick Start

```python
import igraph as ig
from temporal_networks import network_properties

# Load or create your temporal graphs
graphs = [ig.Graph.Barabasi(n=50, m=2) for _ in range(12)]
labels = ["2024-01", "2024-02", "2024-03", ..., "2024-12"]

# Analyze properties with automatic gap detection
props = network_properties(graphs, graph_labels=labels)
```

## Use Cases

- **Epidemiology**: Contact networks with lockdown/reopening cycles.
- **Transportation**: Flight or bike-sharing networks with seasonal operation.
- **Infrastructure**: Systems with scheduled maintenance windows.
- **Social Science**: Dynamic interaction networks around major events.

## Credits

Developed by **Adriana-Stefania Ciupeanu** and **Julien Arino** at the University of Manitoba.

## License

This project is licensed under the GPL-3.0 License.
