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
  - epidemiology
  - transportation networks
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
date: 
 - 09 March 2026
bibliography: 
 - paper.bib
---

# Summary

`temporal_networks` is a Python package designed for **temporal network analysis**—understanding how network structure and properties evolve over time. Unlike general-purpose libraries (igraph, NetworkX) that analyze individual graphs and require users to build temporal infrastructure themselves, this package automatically handles the temporal dimension, temporal metrics, temporal visualization, and crucially, **temporal gaps** (missing time periods in the data). The package computes comprehensive temporal network metrics—centrality trajectories, community evolution across 7 algorithms, edge formation/dissolution dynamics, and individual node property tracking—while automatically detecting and correctly visualizing temporal gaps as visual breaks rather than false continuity lines. This capability is critical for real-world applications where networks have seasonal closures (tourism networks, education networks), maintenance windows (infrastructure systems), crisis-driven interruptions (disease networks during lockdowns), or multi-phase observation periods.

# Statement of Need

Temporal network analysis has become essential for understanding dynamic systems across diverse domains: epidemiological models of disease spread during phases with different containment measures, transportation networks that operate seasonally or have crisis-driven interruptions, social network dynamics around major events, infrastructure system monitoring with maintenance windows, and biological networks across developmental stages. However, temporal network analysis is fundamentally different from static graph analysis, and current tools do not adequately address this difference.

## The Temporal Paradigm Problem

Existing libraries like `python-igraph` (Csárdi & Nepusz, 2006) and `NetworkX` (Hagberg et al., 2008) are designed to analyze **individual graphs**. When researchers have temporal network data (a sequence of networks over time), they must:

1. **Build temporal infrastructure themselves** - There is no standard way to track time across analyses, compare networks at different time points, or aggregate temporal results
2. **Manually handle temporal gaps** - When data has missing periods (seasonal closures, maintenance windows, observation interruptions), users must remember to:
   - Identify where gaps occur
   - Handle gap-specific visualizations (avoiding false continuity lines)
   - Document gaps in analysis
   - Interpret results accounting for gaps
3. **Create custom temporal visualization** - Plotting how network properties change over time is left to the user
4. **Implement temporal metrics** - Metrics like "how many edges formed this month?" or "how did this node's importance change?" must be custom-coded

## The Real Problem: False Continuity Lines

A subtle but serious issue arises from naive visualization: **index-based plotting creates false continuity lines across temporal gaps**. For example, a bike-sharing system operating Mar-Aug and Nov-Feb (summer + winter, 3-month operational gap) would appear to show network transitions from August to November if plotted naively with index-based connection. This creates an illusion of network dynamics that did not actually occur, misleading both analysts and readers. Standard approaches (igraph, NetworkX) do not prevent this; users must manually handle it.

## The Solution: Temporal Networks

`temporal_networks` approaches this problem fundamentally differently. Rather than viewing temporal network analysis as "static graph analysis repeated N times," it makes **temporal analysis**:

- **Temporal awareness is built-in** - The package knows about time; labels like "2024-03", "2024-08", "2024-11" are not just metadata but central to analysis
- **Automatic gap detection** - No manual specification; flexible datetime format support (YYYY-MM, YYYY-MM-DD, YYYY-W##, YYYY-Q#, YYYY)
- **Automatic gap handling** - All visualizations automatically use gap-aware plotting with visual breaks
- **Temporal metrics** - Edge dynamics (formation/dissolution), community evolution, vertex trajectories
- **Transparent reporting** - Users explicitly see where gaps are and their implications

This creates a systematic, scientific approach to temporal network analysis where gaps are handled correctly by default, and users receive clear information about their data structure.

# Implementation

## Core Architecture

The package consists of six main analysis modules plus a gap-detection framework:

### Gap Detection Framework (`_gap_utilities.py`)

The foundation is a flexible gap detection system supporting multiple datetime formats:

- Monthly: `YYYY-MM` (e.g., "2024-03")
- Daily: `YYYY-MM-DD` (e.g., "2024-03-15")  
- Weekly: `YYYY-W##` (e.g., "2024-W12")
- Quarterly: `YYYY-Q#` (e.g., "2024-Q2")
- Annual: `YYYY` (e.g., "2024")

The `detect_temporal_gaps(graph_labels, gap_threshold=1, unit="months")` function:
1. Parses temporal labels in any format
2. Calculates time differences between consecutive labels
3. Identifies gaps exceeding the threshold (default: 1 unit)
4. Returns gap details and continuous segments for downstream functions

```python
gap_info = detect_temporal_gaps(labels, gap_threshold=1, unit="months")
# Returns: {
#   "has_gaps": bool,
#   "num_gaps": int,
#   "gaps": [{"start_idx": ..., "end_idx": ..., "start_label": ..., 
#             "end_label": ..., "gap_size": ...}],
#   "segments": [(0, 6), (9, 10)]  # Continuous periods
# }
```

All downstream functions use these segments for correct plotting via `plot_with_gap_handling()`, which draws separate line segments for each continuous period rather than connecting across gaps.

### Network Properties (`network_properties()`)

Computes structural metrics for each temporal snapshot:
- Node and edge counts
- Network density
- Diameter and average path length
- Clustering coefficient
- Reciprocity (for directed networks)
- Connectivity components

Returns a DataFrame with one row per time step and automatically reports temporal gaps.

### Centrality Measures (`calculate_centralities()`)

Computes 13 node-level centrality scores for each time step:
- Degree centrality
- Closeness centrality  
- Betweenness centrality
- Eigenvector centrality
- PageRank
- Harmonic centrality
- Eccentricity
- Constraint (Burt's constraint)
- Hub and authority scores

Returns a DataFrame with one row per node per time step, tracking how individual node importance evolves. Optional parameter enables temporal visualization of centrality trajectories with gap-aware plotting.

### Community Detection (`communities_measures()`)

Applies seven community detection algorithms to each snapshot:
- Leiden (most modern)
- Louvain (widely-used)
- Walktrap (short random walks)
- Fast Greedy (optimization-based)
- Label Propagation (iterative)
- Spinglass (simulated annealing)
- Infomap (information-theoretic)

For each algorithm, returns:
1. Node-to-community assignments
2. Summary statistics (number of communities, size distribution, modularity)
3. Gap-aware plots showing community evolution

This enables users to assess algorithm stability and choose appropriate methods for their networks.

### Vertex Properties (`vertex_properties()`)

Tracks how properties of specific nodes change over time. Computes the complete suite of centrality measures plus:
- Local clustering coefficient
- Coreness (k-core decomposition)
- Network constraint

Generates individual plots for each metric plus combined visualization with normalized values for comparing multiple metrics.

### Edge Dynamics (`edge_formation_dissolution()`)

Analyzes which edges appear (formation) and disappear (dissolution) between consecutive time steps:
- Absolute counts of new and deleted edges
- Percentage changes relative to network size
- Identifies periods of high network churn vs. stability

Useful for understanding network turnover and identifying critical transition points.

### Community Evolution (`plot_community_evolution()`)

Generates interactive HTML visualization of community dynamics. Includes pragmatic gap handling: when temporal gaps are detected, prints a warning and recommends using static plots from `communities_measures()` (which visualize gaps correctly) rather than potentially misleading interactive animations.

## Gap-Aware Visualization

The key innovation is gap-aware plotting:

```python
def plot_with_gap_handling(ax, graph_labels, y_values, gap_segments, ...):
    """Plot separate line segments for continuous periods only."""
    for segment_start, segment_end in gap_segments:
        x_indices = np.arange(segment_start, segment_end)
        y_segment = [y_values[i] for i in x_indices]
        ax.plot(x_indices, y_segment, ...)  # Separate segment
```

This draws separate line segments for each continuous time period. When data has gaps, visual breaks clearly indicate missing data rather than suggesting (false) network continuity.

**Comparison:**
- **Without gap handling:** Index-based plotting `ax.plot(range(n), y_values)` connects all points, creating false lines across temporal gaps
- **With gap handling:** Only connects points within continuous periods, showing visual breaks where data is missing

## Transparent Gap Reporting

All functions include automatic gap reporting printed to console:

```
================================================================================
TEMPORAL DATA STRUCTURE ANALYSIS
================================================================================

Dataset Overview:
  Number of observations: 10
  Time unit: months
  Date range: 2024-03 to 2025-02

⚠ Data has GAPS: 1 gap(s) detected

  Gap #1:
    From: 2024-08 (index 5)
    To:   2024-11 (index 6)
    Size: 3.0 months

Impact on Temporal Visualization:
  ✓ Plots show SEPARATE LINE SEGMENTS for each continuous period
  ✓ No lines are drawn across gaps
  ✓ Visual breaks indicate where data is missing
================================================================================
```

This transparency ensures users understand their data structure and can make informed interpretations.

# Features

| Feature | Details |
|---------|---------|
| **Automatic gap detection** | No manual specification; supports 5 datetime formats |
| **Flexible thresholds** | Configurable gap_threshold and time units (days/weeks/months/years) |
| **Clear reporting** | Console output showing gap locations, sizes, and implications |
| **Correct visualization** | Gap-aware plotting built into all functions |
| **Comprehensive metrics** | 13 centrality measures, 7 community algorithms, edge dynamics, vertex tracking |
| **Production testing** | 40+ unit tests for core functionality |
| **Scientific transparency** | All gap information available for analysis and reporting |
| **Flexible input** | Works with any temporal network data with temporal labels |

# Example Usage

## Continuous Network Data (No Gaps)

```python
import igraph as ig
from temporal_networks import network_properties, communities_measures

# Create 12 months of continuous network snapshots
graphs = [ig.Graph.Barabasi(n=50, m=2) for _ in range(12)]
labels = ["2024-01", "2024-02", ..., "2024-12"]

# Analyze network properties
props = network_properties(graphs, graph_labels=labels)
# Output: "✓ Data is CONTINUOUS (no gaps detected)"
# Creates: plots with smooth lines connecting all months
# Saves: results to CSV

# Analyze communities
communities = communities_measures(graphs, graph_labels=labels)
# Output: Multiple plots showing community evolution
```

## Seasonal Network with Gaps

```python
# Bike-sharing network: operates Mar-Aug (summer) and Nov-Feb (winter)
graphs = [ig.Graph.Barabasi(n=50, m=2) for _ in range(10)]
labels = ["2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08",
          "2024-11", "2024-12", "2025-01", "2025-02"]

# Analyze properties
props = network_properties(graphs, graph_labels=labels)
# Output: 
#   ⚠ Data has GAPS: 1 gap(s) detected
#   Gap #1: From 2024-08 to 2024-11, Size: 3.0 months
# Creates: plots with visual break between summer and winter
# Interpretation: Avoids false dynamics in Aug-Nov transition
```

## Track Individual Node Importance

```python
from temporal_networks import vertex_properties

# Track one node's importance over time
props = vertex_properties(graphs, node_name="Node_5", graph_labels=labels)
# Returns: DataFrame with node's centrality across all time steps
# Creates: Individual plots for each centrality measure
# Gap-aware: Shows where node disappears/reappears seasonally
```

# Applications

The package has been validated on real-world datasets from multiple domains:

1. **Epidemiological Networks** - COVID-19 contact networks with lockdown/reopening cycles
2. **Transportation Networks** - Flight networks with seasonal patterns and crisis disruptions
3. **Infrastructure Systems** - Bike-sharing networks with seasonal operation windows
4. **Biological Networks** - Temporal protein interaction networks across development
5. **Social Networks** - Event-driven dynamics with inter-event gaps


# Documentation and Accessibility

Complete documentation includes:

- **API documentation** - Docstrings with examples for all functions
- **Quick reference guide** - One-page implementation cheat sheet
- **Testing guide** - Instructions for running local tests and CI/CD
- **Example scripts** - Working code with continuous and gapped data
- **JOSS paper** - This manuscript describing motivation and implementation

All code is available on GitHub with MIT license. Installation via `pip install temporal-networks` or `pip install -e .` from source.

# Limitations and Future Work

**Current Limitations:**
- Interactive HTML animations are less effective with gapped data (static plots recommended)
- Some community detection algorithms require undirected graphs
- Very large graphs (100,000+ nodes) may have performance considerations

**Future Enhancements:**
- Parallel processing for large temporal sequences
- Support for heterogeneous temporal networks (different node/edge types)
- Advanced visualization with interactive gap annotation
- Temporal motif and pattern analysis
- Integration with machine learning for gap imputation

# Acknowledgments



# References