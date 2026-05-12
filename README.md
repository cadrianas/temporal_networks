# Temporal Network Analysis

Python package for analyzing dynamic networks that change over time.

## Features

- **network_properties**: Compute structural metrics (density, diameter, clustering)
- **calculate_centralities**: Calculate node importance (betweenness, PageRank, etc.)
- **communities_measures**: Detect communities using multiple algorithms
- **plot_community_evolution**: Animate how communities change over time
- **vertex_properties**: Track individual node metrics

## Installation

Clone and install:
```bash
git clone https://github.com/YOUR_USERNAME/temporal-network-analysis.git
cd temporal-network-analysis
pip install -e .
```

## Quick Start
```python
from temporal_networks import network_properties
import igraph as ig

# Load your graphs
graphs = [...]  # list of igraph Graph objects

# Compute properties
props = network_properties(graphs, visualisation=True)
print(props)
```

## Case Studies

- **Helsinki City Bikes**: Dynamic bike-sharing network (2016-2020)
- **Air Transportation**: Global flight networks during COVID-19 (2019-2022)

See `examples/` for reproducible scripts.

## Dependencies

- Python 3.8+
- igraph, networkx, pandas, numpy
- matplotlib, seaborn, plotly

## License

GPL-3.0 license 