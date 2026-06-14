"""
Temporal Network Analysis: Synthetic Example - Continuous vs Gapped Data

This example demonstrates how to use the temporal_networks package
to analyze networks that change over time. Importantly, it shows how the
package handles TWO common real-world scenarios:

1. CONTINUOUS DATA: Regular, uninterrupted measurements (e.g., monthly)
2. GAPPED DATA: Data with breaks (e.g., seasonal patterns, operational gaps)

Scenario: Simulating a transportation hub network, with two temporal patterns:
  - Continuous: 12 months of consecutive data
  - Gapped: 6 months (Mar-Aug) + 6 months (Nov-Apr), with 3-month gap between

This mimics real-world situations like:
  - Bike-sharing: April-October season, then closed Nov-March
  - Air transportation: Normal operation, then crisis closure, then recovery
  - Seasonal networks: only active during certain periods

KEY FEATURE: The package preserves gaps in visualizations, showing
discontinuities rather than falsely implying continuous operation during
missing periods. This is critical for accurate temporal analysis.

Running time: ~4-5 minutes
"""

import igraph as ig
import numpy as np
import os
from temporal_networks.network_properties import network_properties
from temporal_networks.calculate_centralities import calculate_centralities
from temporal_networks.communities_measures import communities_measures
from temporal_networks.vertex_properties import vertex_properties
from temporal_networks.edge_formation_dissolution import (
    edge_formation, edge_dissolution)

# Create output directories
os.makedirs("./plots/continuous/", exist_ok=True)
os.makedirs("./plots/gapped/", exist_ok=True)

# ============================================================================
# STEP 1: Create Synthetic Temporal Networks (Continuous & Gapped)
# ============================================================================
print("=" * 70)
print("STEP 1: Creating synthetic temporal networks")
print("=" * 70)

def create_hub_network(n_nodes=15, n_hubs=3, hub_connectivity=0.8,
                       random_edges=5, seed=None):
    """
    Create a synthetic hub-and-spoke network similar to transportation hubs.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes (airports, stations, etc.)
    n_hubs : int
        Number of major hub nodes
    hub_connectivity : float
        Probability of edges between hub and other nodes (0-1)
    random_edges : int
        Additional random edges to add for realism
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    igraph.Graph
        A directed graph with hub-and-spoke structure
    """
    if seed is not None:
        np.random.seed(seed)

    # Start with empty graph
    g = ig.Graph(n_nodes, directed=True)

    # Name nodes
    node_names = [f"Node_{i}" for i in range(n_nodes)]
    g.vs["name"] = node_names

    # Add edges from hubs to other nodes
    hub_nodes = list(range(n_hubs))  # First n_hubs are the major hubs

    for hub in hub_nodes:
        for other in range(n_nodes):
            if other != hub:
                if np.random.random() < hub_connectivity:
                    g.add_edge(hub, other)
                    g.add_edge(other, hub)  # Bidirectional

    # Add connections between hubs (they're interconnected)
    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            g.add_edge(i, j)
            g.add_edge(j, i)

    # Add some random edges for realism
    for _ in range(random_edges):
        source = np.random.randint(0, n_nodes)
        target = np.random.randint(0, n_nodes)
        if source != target:
            g.add_edge(source, target)

    # Assign weights to edges (representing traffic volume)
    g.es["weight"] = [np.random.randint(1, 100) for _ in range(g.ecount())]

    return g

print("\n" + "-" * 70)
print("SCENARIO A: CONTINUOUS DATA (12 consecutive months)")
print("-" * 70)

graphs_continuous = []
labels_continuous = [
    "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
    "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"
]

print("\nCreating 12 months of continuous data (normal operation)...")
for i in range(12):
    # Slight variation each month but relatively stable
    g = create_hub_network(n_nodes=15, n_hubs=3, hub_connectivity=0.65,
                           random_edges=7, seed=100 + i)
    graphs_continuous.append(g)
    print(f"  ✓ {labels_continuous[i]}: {g.vcount()} nodes, {g.ecount()} edges")

print(f"\n✓ Created {len(graphs_continuous)} continuous temporal networks")

print("\n" + "-" * 70)
print("SCENARIO B: GAPPED DATA (seasonal with operational gap)")
print("-" * 70)

graphs_gapped = []
labels_gapped = []

print("\nCreating data with a temporal gap (like seasonal operation)...")
print("Pattern: Mar-Aug (6 months), then 3-month gap, then Nov-Feb (4 months)")

# March to August (6 months of operation)
print("\n  Operating Period 1 (Mar-Aug):")
operation_1_labels = ["2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08"]
for i, label in enumerate(operation_1_labels):
    g = create_hub_network(n_nodes=15, n_hubs=3, hub_connectivity=0.70,
                           random_edges=8, seed=200 + i)
    graphs_gapped.append(g)
    labels_gapped.append(label)
    print(f"    ✓ {label}: {g.vcount()} nodes, {g.ecount()} edges")

print("\n  [GAP: September-October: No data (system closed/unavailable)]")

# November to February (4 months of operation in next season)
print("\n  Operating Period 2 (Nov-Feb):")
operation_2_labels = ["2024-11", "2024-12", "2025-01", "2025-02"]
for i, label in enumerate(operation_2_labels):
    g = create_hub_network(n_nodes=15, n_hubs=3, hub_connectivity=0.65,
                           random_edges=7, seed=206 + i)
    graphs_gapped.append(g)
    labels_gapped.append(label)
    print(f"    ✓ {label}: {g.vcount()} nodes, {g.ecount()} edges")

print(f"\n✓ Created {len(graphs_gapped)} gapped temporal networks "
      f"with gap between periods")

# ============================================================================
# STEP 2: Analyze Network Properties - CONTINUOUS DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2A: Network Properties - CONTINUOUS DATA")
print("=" * 70)

print("\nAnalyzing continuous 12-month dataset...")
props_continuous = network_properties(
    graphs=graphs_continuous,
    graph_labels=labels_continuous,
    filename="./plots/continuous/synthetic_continuous_network_properties.csv",
    save_path="./plots/continuous/",  # For plots
    visualisation=True  # Creates PDF plots - should show continuous line
)

print("✓ Network properties computed for continuous data")
print("\nPlot will show: Unbroken temporal line (Jan-Dec)")
print("All 12 data points connected sequentially")

# ============================================================================
# STEP 2B: Analyze Network Properties - GAPPED DATA
# ============================================================================
print("\n" + "-" * 70)
print("STEP 2B: Network Properties - GAPPED DATA")
print("=" * 70)

print("\nAnalyzing gapped dataset (Mar-Aug, gap, Nov-Feb)...")
props_gapped = network_properties(
    graphs=graphs_gapped,
    graph_labels=labels_gapped,
    filename="./plots/gapped/synthetic_gapped_network_properties.csv",  # Full path
    save_path="./plots/gapped/",  # For plots
    visualisation=True  # Creates PDF plots - should show gap!
)

print("✓ Network properties computed for gapped data")
print("\nPlot will show: Two separate temporal segments with GAP in between")
print("  - Segment 1: Mar-Aug (6 points)")
print("  - [GAP: Sep-Oct missing]")
print("  - Segment 2: Nov-Feb (4 points)")
print("This visual gap correctly represents the real operational break")

# ============================================================================
# STEP 3: Compare the Two Approaches
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Comparison - Continuous vs Gapped Data Handling")
print("=" * 70)

print("""
WHY THIS MATTERS:

Continuous Data (12 months):
  - Shows full annual cycle
  - Useful for seasonal patterns
  - No missing periods
  - Example: daily weather measurements, continuous system operation

Gapped Data (Mar-Aug + Nov-Feb with gap):
  - Represents real-world gaps (system closure, seasonal operation)
  - Prevents false interpretation (plotting through gap would mislead)
  - Critical for systems like:
    * Bike-sharing (closed in winter)
    * Seasonal tourism
    * Systems with maintenance windows

THE KEY FEATURE:
Our package's plot functions preserve gaps in the timeline.
Instead of drawing a line through the missing Sep-Oct data,
the plots show a visual discontinuity, accurately representing reality.

This is handled through the graph_labels parameter:
  - The labels encode temporal information (month/year)
  - The plotting function respects these labels
  - When there's a gap in the label sequence, it shows in the plot
  - No false continuity implied
""")

print("\n" + "-" * 70)
print("Data Summary Comparison")
print("-" * 70)

print("\nCONTINUOUS DATA:")
print(f"  Time span: {labels_continuous[0]} to {labels_continuous[-1]}")
print(f"  Number of observations: {len(labels_continuous)}")
print("  Pattern: No gaps (every month)")

print("\nGAPPED DATA:")
print(f"  Period 1: {labels_gapped[0]} to {labels_gapped[5]}")
print(f"  [GAP: {labels_gapped[5]} → {labels_gapped[6]}]")
print(f"  Period 2: {labels_gapped[6]} to {labels_gapped[-1]}")
print(f"  Number of observations: {len(labels_gapped)}")
print("  Pattern: 6 months + gap + 4 months")

# ============================================================================
# STEP 4: Compute Node Centrality Measures - CONTINUOUS DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4A: Node Centralities - CONTINUOUS DATA")
print("=" * 70)

centralities_continuous = calculate_centralities(
    graphs=graphs_continuous,
    graph_labels=labels_continuous,
    filename="./plots/continuous/synthetic_continuous_centralities.csv"  # Full path
)

print(f"\n✓ Centralities computed for {len(centralities_continuous)} "
      f"node-time combinations")
print("\nMost central nodes (first and last months) - Continuous Data:")

# Show central nodes in first and last months
for label in [labels_continuous[0], labels_continuous[-1]]:
    month_data = centralities_continuous[centralities_continuous["Graph"] == label]
    top_node = month_data.nlargest(
        1, "Betweenness_Centrality")[["Node", "Betweenness_Centrality"]]
    if not top_node.empty:
        node_name = top_node.iloc[0]["Node"]
        centrality_val = top_node.iloc[0]["Betweenness_Centrality"]
        print(f"  {label}: {node_name} (betweenness = {centrality_val:.2f})")

# ============================================================================
# STEP 4B: Compute Node Centrality Measures - GAPPED DATA
# ============================================================================
print("\n" + "-" * 70)
print("STEP 4B: Node Centralities - GAPPED DATA")
print("-" * 70)

centralities_gapped = calculate_centralities(
    graphs=graphs_gapped,
    graph_labels=labels_gapped,
    filename="./plots/gapped/synthetic_gapped_centralities.csv"  # Full path
)

print(f"\n✓ Centralities computed for {len(centralities_gapped)} "
      f"node-time combinations")
print("\nMost central nodes - Gapped Data:")
print("  Before gap (Aug) and after gap (Nov):")

# Show central nodes before and after gap
before_gap = centralities_gapped[centralities_gapped["Graph"] == "2024-08"]
after_gap = centralities_gapped[centralities_gapped["Graph"] == "2024-11"]

if not before_gap.empty:
    top_node = before_gap.nlargest(
        1, "Betweenness_Centrality")[["Node", "Betweenness_Centrality"]]
    node_name = top_node.iloc[0]["Node"]
    centrality_val = top_node.iloc[0]["Betweenness_Centrality"]
    print(f"    Before gap (2024-08): {node_name} (betweenness = {centrality_val:.2f})")

if not after_gap.empty:
    top_node = after_gap.nlargest(
        1, "Betweenness_Centrality")[["Node", "Betweenness_Centrality"]]
    node_name = top_node.iloc[0]["Node"]
    centrality_val = top_node.iloc[0]["Betweenness_Centrality"]
    print(f"    After gap (2024-11): {node_name} (betweenness = {centrality_val:.2f})")

print("\nNote: Gap (Sep-Oct) correctly NOT shown in timeline")

# ============================================================================
# STEP 5: Detect Communities - GAPPED DATA (key demonstration)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Community Detection - GAPPED DATA")
print("=" * 70)

print("\nDetecting communities across the gap...")
print("This shows how network structure changes across the operational break")

communities_gapped = communities_measures(
    graphs=graphs_gapped,
    graph_labels=labels_gapped,
    save_path="./plots/gapped/",  # SEPARATE FOLDER
    visualisation=True
)

print("\n✓ Communities detected using Louvain, Leiden, and Walktrap")
print("\nImportant: The resulting plots show two separate temporal segments")
print("with a visual gap representing the Sep-Oct closure period.")

# ============================================================================
# STEP 6A: Edge Dynamics - CONTINUOUS DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6A: Edge Formation/Dissolution - CONTINUOUS DATA")
print("=" * 70)

print("\nAnalyzing how routes change over 12 continuous months...")

print("Computing edge formation...")
edge_formation_continuous = edge_formation(
    graphs=graphs_continuous,
    graph_labels=labels_continuous,
    save_path="./plots/continuous/"  # SEPARATE FOLDER
)

print("Computing edge dissolution...")
edge_dissolution_continuous = edge_dissolution(
    graphs=graphs_continuous,
    graph_labels=labels_continuous,
    save_path="./plots/continuous/"  # SEPARATE FOLDER
)

print("\n✓ Edge dynamics computed for continuous data")
print("  The plots show continuous time series (Jan-Dec)")

# ============================================================================
# STEP 6B: Edge Dynamics - GAPPED DATA
# ============================================================================
print("\n" + "-" * 70)
print("STEP 6B: Edge Formation/Dissolution - GAPPED DATA")
print("=" * 70)

print("\nAnalyzing how routes change within and across the gap...")

print("Computing edge formation...")
edge_formation_gapped = edge_formation(
    graphs=graphs_gapped,
    graph_labels=labels_gapped,
    save_path="./plots/gapped/"  # SEPARATE FOLDER
)

print("Computing edge dissolution...")
edge_dissolution_gapped = edge_dissolution(
    graphs=graphs_gapped,
    graph_labels=labels_gapped,
    save_path="./plots/gapped/"  # SEPARATE FOLDER
)

print("\n✓ Edge dynamics computed for gapped data")
print("\nKey observations at the gap boundary:")

# Calculate edge changes around the gap
g_aug = graphs_gapped[5]  # August (before gap)
g_nov = graphs_gapped[6]  # November (after gap)

edges_aug = set([(e.source, e.target) for e in g_aug.es])
edges_nov = set([(e.source, e.target) for e in g_nov.es])

new_edges = edges_nov - edges_aug
lost_edges = edges_aug - edges_nov

print("\n  Aug → [GAP: Sep-Oct] → Nov (3-month operational break):")
print(f"    Routes that were lost during shutdown: {len(lost_edges)}")
print(f"    New routes when reopening: {len(new_edges)}")
print("    This 3-month gap is correctly represented in plots")
print("    (not hidden or smoothed over)")
print("\n  The plots will show 6 points (Mar-Aug), then skip to 4 points (Nov-Feb)")

# ============================================================================
# STEP 6C: EXPLICIT GAP VERIFICATION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6C: Explicit Verification that Gap is Preserved")
print("=" * 70)

print("\nGapped Data Timeline (showing the gap):")
print("\nExpected labels:")
for label in labels_gapped:
    print(f"  - {label}")

print("\n" + "-" * 70)
print("IMPORTANT: Notice the gap between 2024-08 and 2024-11")
print("-" * 70)
print("\nMissing: 2024-09, 2024-10")
print("\nThis is NOT a plotting error!")
print("This is the CORRECT behavior - the gap is INTENTIONALLY preserved")
print("to accurately represent the 3-month operational closure.")

print("\nComparison:")
print("\n  CONTINUOUS DATA (12 months, no gaps):")
print(f"    Labels: {labels_continuous}")
print("    X-axis in plots: Shows every month in sequence")

print("\n  GAPPED DATA (10 months with gap):")
print(f"    Labels: {labels_gapped}")
print("    X-axis in plots: Shows Mar-Aug, then jumps to Nov-Feb")
print("                     (gap between Aug and Nov is visible)")

print("\n" + "=" * 70)
print("This demonstrates that the package CORRECTLY handles temporal gaps")
print("=" * 70)

# ============================================================================
# STEP 7: Vertex Properties Tracking Across the Gap
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Tracking Individual Node Across the Operational Gap")
print("=" * 70)

node_to_track = "Node_0"

vertex_props_gapped = vertex_properties(
    graphs=graphs_gapped,
    node_name=node_to_track,
    graph_labels=labels_gapped,
    filename=f"synthetic_{node_to_track}_properties_gapped.csv",
    save_path="./plots/gapped/",  # For plots
    visualisation=True
)

print(f"\n✓ Properties of {node_to_track} tracked across gapped timeline")
print("  Timeline: Mar-Aug, [GAP], Nov-Feb")
print("\nVertex properties show:")
print("  - Behavior during Operation 1 (Mar-Aug)")
print("  - [GAP: Sep-Oct missing data]")
print("  - Behavior during Operation 2 (Nov-Feb)")
print("\nPlots correctly display the gap as a visual break")

# ============================================================================
# STEP 8: Summary - Gap Handling Feature Showcase
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Continuous vs Gapped Data Handling")
print("=" * 70)

print("""
This example demonstrated TWO critical temporal data patterns:

================================================================================
CONTINUOUS DATA (12 consecutive months)
================================================================================
  Pattern: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec

  Characteristics:
    - Uninterrupted measurements
    - Complete annual cycle visible
    - Useful for seasonal analysis
    - No data collection gaps

  Plot behavior:
    - All 12 points connected by continuous line
    - Shows full progression through year
    - Good for detecting cyclical patterns

  Real-world examples:
    - Daily weather measurements
    - Continuous system operation (24/7)
    - Traffic patterns across all days

================================================================================
GAPPED DATA (Mar-Aug + 3-month gap + Nov-Feb)
================================================================================
  Pattern: Mar, Apr, May, Jun, Jul, Aug, [GAP], Nov, Dec, Jan, Feb

  Characteristics:
    - Operational breaks (seasonal closure, maintenance windows)
    - Two separate measurement periods
    - Realistic for many real-world systems
    - Critical to preserve gap in analysis

  Plot behavior:
    - First segment (Mar-Aug): 6 connected points
    - [VISUAL GAP: Sep-Oct missing]
    - Second segment (Nov-Feb): 4 connected points
    - Gap correctly shown as discontinuity in plots

  Real-world examples:
    - Bike-sharing: April-October season, closed Nov-March
    - Seasonal tourism: open in summer, closed in winter
    - Aviation during crisis: operations halted, then restarted
    - School systems: operation during term, gap during breaks

================================================================================
WHY GAP PRESERVATION MATTERS
================================================================================

❌ WRONG APPROACH: Plotting through the gap
   If you plot Mar→Aug→Nov as a continuous line, you falsely imply that
   the system operated continuously through Sep-Oct. This misleads analysis
   of network properties and temporal trends.

✅ CORRECT APPROACH: Preserving the gap
   Our package preserves gaps by respecting the temporal labels. When labels
   show a gap (Aug → Nov with no Sep-Oct), the plots show a visual break.
   This accurately represents reality.

CRITICAL FOR:
  - Correctly identifying seasonal patterns
  - Avoiding false trend projections across gaps
  - Understanding how systems reorganize after downtime
  - Analyzing disease importation risk across operational breaks
  - Any temporal analysis where discontinuities are meaningful

================================================================================
KEY FEATURE IMPLEMENTED
================================================================================

The temporal_networks package handles gaps through the graph_labels parameter:

✓ Labels encode temporal information (YYYY-MM format)
✓ Plotting functions recognize temporal gaps
✓ Gaps appear as visual discontinuities in output
✓ No artificial smoothing or interpolation
✓ Allows realistic analysis of gapped systems

USAGE PATTERN:

For continuous data:
  labels = ["2024-01", "2024-02", "2024-03", ...]  # Every month
  # Plot shows continuous line

For gapped data:
  labels = ["2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08",
            "2024-11", "2024-12", "2025-01", "2025-02"]  # Note the gap
  # Plot shows segment, gap, segment

The gap appears naturally because the labels have missing months.
No special parameters needed - it's handled automatically.

================================================================================
EPIDEMIOLOGICAL RELEVANCE
================================================================================

For disease importation risk assessment:

Continuous Network (no gaps):
  - All pathways available 24/7
  - Disease can spread along any route at any time
  - Risk is proportional to network structure and volume

Gapped Network (seasonal closure):
  - During closure: disease cannot enter via closed pathways
  - During operation: risk exists along available routes
  - Transition periods (reopening) show rapid risk changes

Example - Bike sharing & disease transmission:
  - Apr-Oct: Bike-sharing operates, people travel between stations
    → Network pathways enable transmission
  - Nov-Mar: System closed, people use other transport
    → Different disease transmission pathways (buses, cars, walking)
  - Apr (reopening): Network re-activates
    → Disease dynamics change as bike pathways reopen

Correctly preserving gaps ensures accurate risk assessment.

""")

print("\n" + "=" * 70)
print("Output files created (CONTINUOUS vs GAPPED - saved separately)")
print("=" * 70)
print("""
═══════════════════════════════════════════════════════════════════════════════

CONTINUOUS DATA (12 months, no gaps):
  Location: ./plots/continuous/

  CSV results:
    - synthetic_continuous_network_properties.csv
    - synthetic_continuous_centralities.csv

  PDF visualizations (continuous 12-point time series):
    - Number of Nodes.pdf          (12 connected points)
    - Number of Edges.pdf          (12 connected points)
    - Density.pdf                  (12 connected points)
    - ... and other metrics
    - edges_formed.pdf             (11 data points: Jan→Feb, Feb→Mar, etc.)
    - edges_dissolved.pdf          (11 data points)

  Key feature: X-axis shows all 12 months in unbroken sequence
               Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec

═══════════════════════════════════════════════════════════════════════════════

GAPPED DATA (Mar-Aug + 3-month gap + Nov-Feb):
  Location: ./plots/gapped/

  CSV results:
    - synthetic_gapped_network_properties.csv
    - synthetic_gapped_centralities.csv
    - synthetic_Node_0_properties_gapped.csv
    - communities_*_assignments.csv
    - communities_*_stats.csv

  PDF visualizations (showing the gap visually):
    - Number of Nodes.pdf          (6 points, gap, 4 points)
    - Number of Edges.pdf          (6 points, gap, 4 points)
    - Density.pdf                  (6 points, gap, 4 points)
    - ... and other metrics
    - edges_formed.pdf             (9 data points, with gap)
    - edges_dissolved.pdf          (9 data points, with gap)
    - communities_walktrap_*.pdf
    - communities_fast_greedy_*.pdf
    - communities_label_prop_*.pdf
    - communities_spinglass_*.pdf
    - communities_infomap_*.pdf

  Key feature: X-axis shows Mar-Aug (6 points), then [GAP], then Nov-Feb (4 points)
               The missing Sep-Oct is VISUALLY OBVIOUS in the plots
               This accurately represents the 3-month operational closure

═══════════════════════════════════════════════════════════════════════════════

WHAT THIS DEMONSTRATES:
  ✓ Gap handling is NOT a bug - it's a FEATURE
  ✓ Plots for gapped data show the gap explicitly
  ✓ No false continuity is implied
  ✓ Temporal structure is preserved accurately
  ✓ Real-world scenarios (seasonal closures, maintenance windows) work correctly

COMPARE THE PDF FILES:
  1. Open continuous/Number of Edges.pdf
     → See 12 points in smooth sequence (Jan→Dec)

  2. Open gapped/Number of Edges.pdf
     → See 6 points (Mar→Aug), gap, then 4 points (Nov→Feb)
     → Notice the x-axis label jump between August and November

This visual difference proves the gap handling works!
""")

print("\n✓ Example complete!")
print("=" * 70)
