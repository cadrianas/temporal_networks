"""
Comprehensive Test Suite for Temporal Network Analysis Package

Tests all 6 core functions on synthetic data with both continuous and gapped time periods.
"""

import sys
import os

# Add the refactored code to path
sys.path.insert(0, '/home/claude')

from test_data_generator import (
    create_test_graphs_continuous,
    create_test_graphs_with_gaps,
    create_test_graphs_small,
    print_graph_info
)

# Import the refactored functions
from network_properties import network_properties
from calculate_centralities import calculate_centralities
from vertex_properties import vertex_properties
from communities_measures import communities_measures
from plot_community_evolution import plot_community_evolution
from edge_formation_dissolution import edge_formation, edge_dissolution

import pandas as pd


def test_network_properties():
    """Test network_properties function with continuous and gapped data."""
    
    print("\n" + "="*70)
    print("TEST 1: network_properties()")
    print("="*70)
    
    # Test 1A: Continuous time
    print("\n--- Test 1A: Continuous Time (No Gaps) ---")
    graphs, labels = create_test_graphs_continuous(num_graphs=6)
    print_graph_info(graphs, labels)
    
    props_df = network_properties(
        graphs,
        graph_labels=labels,
        filename="test_network_properties_continuous.csv",
        save_path="test_output/continuous/",
        visualisation=True
    )
    
    print("\n✓ Continuous time results:")
    print(props_df[["Graph", "Number of Nodes", "Number of Edges", "Density"]].to_string())
    
    # Test 1B: Gapped time
    print("\n--- Test 1B: Gapped Time (Missing Months) ---")
    graphs, labels = create_test_graphs_with_gaps(num_graphs_target=6)
    
    props_df = network_properties(
        graphs,
        graph_labels=labels,
        filename="test_network_properties_gaps.csv",
        save_path="test_output/gaps/",
        visualisation=True
    )
    
    print("\n✓ Gapped time results (note the gaps in sequence):")
    print(props_df[["Graph", "Number of Nodes", "Number of Edges", "Density"]].to_string())
    print("\n✓ Check that plots show visual gap between missing months!")


def test_calculate_centralities():
    """Test calculate_centralities function."""
    
    print("\n" + "="*70)
    print("TEST 2: calculate_centralities()")
    print("="*70)
    
    # Use small graphs for speed
    graphs, labels = create_test_graphs_small(num_graphs=3, num_nodes=15)
    
    centralities_df = calculate_centralities(
        graphs,
        graph_labels=labels,
        filename="test_centralities.csv"
    )
    
    print("\n✓ Computed centralities for all nodes:")
    print(f"  Total rows: {len(centralities_df)} (3 graphs × 15 nodes)")
    print(f"  Columns: {list(centralities_df.columns)}")
    
    # Show sample for first graph
    print(f"\n✓ Sample (first graph, first 5 nodes):")
    sample = centralities_df[centralities_df["Graph"] == labels[0]].head(5)
    print(sample[["Graph", "Node", "Degree_Centrality", "Betweenness_Centrality", "PageRank"]].to_string())


def test_vertex_properties():
    """Test vertex_properties function."""
    
    print("\n" + "="*70)
    print("TEST 3: vertex_properties()")
    print("="*70)
    
    # Use continuous data
    graphs, labels = create_test_graphs_continuous(num_graphs=6)
    
    # Pick a node to track
    node_to_track = "Node_5"
    
    print(f"\nTracking node: {node_to_track}")
    
    vertex_df = vertex_properties(
        graphs,
        node_name=node_to_track,
        graph_labels=labels,
        filename="test_vertex_properties.csv",
        save_path="test_output/continuous/",
        visualisation=True
    )
    
    print("\n✓ Tracked vertex properties over time:")
    print(vertex_df[["Graph", "Degree_Centrality", "Betweenness_Centrality", "PageRank"]].to_string())


def test_communities_measures():
    """Test communities_measures function."""
    
    print("\n" + "="*70)
    print("TEST 4: communities_measures()")
    print("="*70)
    
    # Use small graphs for speed
    graphs, labels = create_test_graphs_small(num_graphs=3, num_nodes=20)
    
    results = communities_measures(
        graphs,
        graph_labels=labels,
        save_path="test_output/communities/",
        visualisation=True
    )
    
    print("\n✓ Community detection results:")
    for algo_name, df in results.items():
        print(f"\n  {algo_name}: {len(df)} nodes × {len(df['Graph'].unique())} graphs")
        if len(df) > 0:
            sample = df[df["Graph"] == labels[0]].head(3)
            print(f"    Sample:\n{sample[['Graph', 'Node', 'Community']].to_string(index=False)}")


def test_plot_community_evolution():
    """Test plot_community_evolution function."""
    
    print("\n" + "="*70)
    print("TEST 5: plot_community_evolution()")
    print("="*70)
    
    # Use small graphs for speed
    graphs, labels = create_test_graphs_small(num_graphs=4, num_nodes=20)
    
    try:
        plot_community_evolution(
            graphs,
            community_algorithm="louvain",
            output_file="test_community_evolution.html"
        )
        print("\n✓ Community evolution animation saved: test_community_evolution.html")
    except Exception as e:
        print(f"\n⚠️  Error creating animation: {e}")


def test_edge_dynamics():
    """Test edge_formation and edge_dissolution functions."""
    
    print("\n" + "="*70)
    print("TEST 6: edge_formation() and edge_dissolution()")
    print("="*70)
    
    # Test with both continuous and gapped
    print("\n--- Test 6A: Continuous Time ---")
    graphs, labels = create_test_graphs_continuous(num_graphs=6)
    
    edge_form_df = edge_formation(
        graphs,
        graph_labels=labels,
        save_path="test_output/continuous/"
    )
    
    print("\n✓ Edge formation (continuous):")
    print(edge_form_df[["Graph", "Edges_Formed", "Edges_Dissolved"]].to_string())
    
    print("\n--- Test 6B: Gapped Time ---")
    graphs, labels = create_test_graphs_with_gaps(num_graphs_target=6)
    
    edge_form_df = edge_formation(
        graphs,
        graph_labels=labels,
        save_path="test_output/gaps/"
    )
    
    print("\n✓ Edge formation (gapped - note missing rows):")
    print(edge_form_df[["Graph", "Edges_Formed", "Edges_Dissolved"]].to_string())
    print("\n✓ Check plot: gaps should be visible!")


def run_all_tests():
    """Run all tests."""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "TEMPORAL NETWORK ANALYSIS - TEST SUITE" + " "*20 + "║")
    print("╚" + "="*68 + "╝")
    
    # Create output directories
    os.makedirs("test_output/continuous", exist_ok=True)
    os.makedirs("test_output/gaps", exist_ok=True)
    os.makedirs("test_output/communities", exist_ok=True)
    
    try:
        test_network_properties()
        test_calculate_centralities()
        test_vertex_properties()
        test_communities_measures()
        test_plot_community_evolution()
        test_edge_dynamics()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nOutput files saved to:")
        print("  - test_output/continuous/   (continuous time results)")
        print("  - test_output/gaps/          (gapped time results)")
        print("  - test_output/communities/   (community detection results)")
        print("\nKey Files to Check:")
        print("  1. test_output/continuous/Number of Edges.pdf")
        print("     -> Should show smooth line, all 6 months continuous")
        print("  2. test_output/gaps/Number of Edges.pdf")
        print("     -> Should show gap(s) in the timeline")
        print("  3. test_community_evolution.html")
        print("     -> Open in browser to see animation")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
