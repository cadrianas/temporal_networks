"""
Test Data Generator for Temporal Network Analysis

Creates synthetic temporal networks for testing the package.
Generates both continuous time sequences (no gaps) and gapped sequences
(with missing months) to validate the date/gap handling.
"""

import igraph as ig
import random
from typing import List, Tuple


def create_test_graphs_continuous(num_graphs: int = 12,
                                  base_nodes: int = 30,
                                  base_edges_per_node: int = 3) -> Tuple[List, List]:
    """
    Create synthetic temporal network with continuous time (no gaps).
    
    Parameters
    ----------
    num_graphs : int
        Number of graphs to generate (default: 12 = one year of months)
    base_nodes : int
        Base number of nodes (default: 30)
    base_edges_per_node : int
        Average edges per node (default: 3)
        
    Returns
    -------
    tuple
        (graphs, labels) where:
        - graphs: list of igraph.Graph objects
        - labels: list of date strings like ["2019-01", "2019-02", ...]
    """
    
    graphs = []
    labels = []
    
    # Generate 12 months of data
    year = 2019
    month = 1
    
    # Keep node names consistent across time
    node_names = [f"Node_{i}" for i in range(base_nodes)]
    
    for i in range(num_graphs):
        # Create label
        label = f"{year}-{month:02d}"
        labels.append(label)
        
        # Create graph with slight variations
        # Add/remove ~10% of edges each month for realism
        edges = []
        num_edges = base_nodes * base_edges_per_node + random.randint(-5, 5)
        
        for _ in range(num_edges):
            source = random.randint(0, base_nodes - 1)
            target = random.randint(0, base_nodes - 1)
            if source != target:  # No self-loops
                edges.append((source, target))
        
        # Create directed graph
        g = ig.Graph(base_nodes, edges, directed=True)
        g.vs["name"] = node_names
        
        # Add label as attribute
        g["filename"] = label
        
        graphs.append(g)
        
        # Increment month
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    print(f"✓ Created {len(graphs)} continuous graphs: {labels[0]} to {labels[-1]}")
    return graphs, labels


def create_test_graphs_with_gaps(num_graphs_target: int = 12,
                                base_nodes: int = 30,
                                base_edges_per_node: int = 3,
                                missing_months: List[int] = None) -> Tuple[List, List]:
    """
    Create synthetic temporal network with gaps (missing time periods).
    
    Parameters
    ----------
    num_graphs_target : int
        Target number of graphs to generate (will be fewer due to gaps)
    base_nodes : int
        Base number of nodes (default: 30)
    base_edges_per_node : int
        Average edges per node (default: 3)
    missing_months : list of int, optional
        Months to skip (1-12). E.g., [2, 5, 8] skips Feb, May, Aug
        If None, defaults to [2, 5, 8] (every ~3 months)
        
    Returns
    -------
    tuple
        (graphs, labels) with missing months
    """
    
    if missing_months is None:
        missing_months = [2, 5, 8, 11]  # Skip every ~3 months
    
    graphs = []
    labels = []
    
    year = 2019
    month = 1
    graphs_created = 0
    
    node_names = [f"Node_{i}" for i in range(base_nodes)]
    
    # Create graphs for 24 months (2 years), skipping certain months
    for _ in range(24):
        if month not in missing_months and graphs_created < num_graphs_target:
            # Create label
            label = f"{year}-{month:02d}"
            labels.append(label)
            
            # Create graph
            edges = []
            num_edges = base_nodes * base_edges_per_node + random.randint(-5, 5)
            
            for _ in range(num_edges):
                source = random.randint(0, base_nodes - 1)
                target = random.randint(0, base_nodes - 1)
                if source != target:
                    edges.append((source, target))
            
            g = ig.Graph(base_nodes, edges, directed=True)
            g.vs["name"] = node_names
            g["filename"] = label
            
            graphs.append(g)
            graphs_created += 1
        
        # Increment month
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    print(f"✓ Created {len(graphs)} graphs with gaps: {labels[0]} to {labels[-1]}")
    print(f"  Missing months: {missing_months}")
    print(f"  Actual sequence: {labels}")
    return graphs, labels


def create_test_graphs_small(num_graphs: int = 3,
                            num_nodes: int = 15) -> Tuple[List, List]:
    """
    Create very small test graphs for quick testing.
    
    Parameters
    ----------
    num_graphs : int
        Number of graphs (default: 3)
    num_nodes : int
        Number of nodes per graph (default: 15)
        
    Returns
    -------
    tuple
        (graphs, labels)
    """
    
    graphs = []
    labels = []
    node_names = [f"N{i}" for i in range(num_nodes)]
    
    for i in range(num_graphs):
        label = f"2019-Q{i+1}"
        labels.append(label)
        
        # Create small Barabasi-Albert graph
        g = ig.Graph.Barabasi(num_nodes, m=2)
        g.vs["name"] = node_names
        g["filename"] = label
        
        graphs.append(g)
    
    print(f"✓ Created {len(graphs)} small graphs: {labels}")
    return graphs, labels


def print_graph_info(graphs: List, labels: List) -> None:
    """Print summary statistics about the graphs."""
    
    print("\n" + "="*60)
    print("GRAPH INFORMATION")
    print("="*60)
    
    for label, graph in zip(labels, graphs):
        print(f"\n{label}:")
        print(f"  Nodes: {graph.vcount()}")
        print(f"  Edges: {graph.ecount()}")
        print(f"  Density: {graph.density():.4f}")
        print(f"  Directed: {graph.is_directed()}")
        print(f"  Connected: {graph.is_connected()}")
