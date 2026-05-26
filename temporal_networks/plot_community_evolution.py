"""
Temporal Network Analysis: Community Evolution Visualization Module

This module provides functions for creating interactive animated visualizations
of how community structures change over time in networks.
"""

import igraph as ig
import plotly.graph_objs as go
from plotly.offline import plot
import random
from typing import List


def plot_community_evolution(graphs: List,
                            community_algorithm: str,
                            output_file: str = "community_evolution.html") -> None:
    """
    Create interactive animation of community evolution across temporal network.
    
    Generates an interactive Plotly animation showing how community structures
    detected by a specified algorithm evolve across a sequence of networks.
    Nodes are colored by community membership and animation allows frame-by-frame
    viewing of the temporal evolution.
    
    Parameters
    ----------
    graphs : list of igraph.Graph
        List of igraph.Graph objects to analyze. All graphs should have
        consistent node labels.
    community_algorithm : str
        Community detection algorithm to use. Options:
        - "edge_betweenness"
        - "walktrap"
        - "fast_greedy"
        - "label_prop"
        - "spinglass"
        - "leiden"
        - "louvain"
    output_file : str, optional
        Filename for saving the HTML animation (default: "community_evolution.html")
        
    Returns
    -------
    None
        Saves output_file with interactive animation
        
    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import plot_community_evolution
    >>> graphs = [ig.Graph.Barabasi(n=50, m=2) for _ in range(12)]
    >>> plot_community_evolution(graphs, community_algorithm="louvain")
    
    Notes
    -----
    The animation creates one frame per graph, with playback controls:
    - Play: Animate through all frames
    - Pause: Stop animation
    - Restart: Return to first frame
    
    Node positions are randomly assigned if not present in graph attributes.
    For better visualization, pre-compute positions (e.g., using Fruchterman-Reingold)
    and store as "x" and "y" vertex attributes.
    """
    
    # Validate inputs
    if not graphs:
        raise ValueError("graphs list cannot be empty")
    
    # Map algorithm names to igraph functions
    algorithm_map = {
        "edge_betweenness": ig.Graph.community_edge_betweenness,
        "walktrap": ig.Graph.community_walktrap,
        "fast_greedy": ig.Graph.community_fastgreedy,
        "label_prop": ig.Graph.community_label_propagation,
        "spinglass": ig.Graph.community_spinglass,
        "leiden": ig.Graph.community_leiden,
        "louvain": ig.Graph.community_multilevel,
    }
    
    # Get algorithm function
    if community_algorithm.lower() not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {community_algorithm}. "
                        f"Must be one of: {list(algorithm_map.keys())}")
    
    algo_func = algorithm_map[community_algorithm.lower()]
    algo_name = community_algorithm.capitalize()
    
    print(f"Computing community evolution with {algo_name} algorithm...")
    
    # Detect communities for each graph
    communities_list = []
    
    for graph_idx, graph in enumerate(graphs):
        try:
            # Convert to undirected if needed for specific algorithms
            g = graph.copy()
            if g.is_directed() and community_algorithm.lower() in ["walktrap", 
                                                                    "fast_greedy",
                                                                    "label_prop",
                                                                    "spinglass"]:
                g = g.as_undirected()
            
            # Detect communities
            try:
                if community_algorithm.lower() in ["walktrap", "fast_greedy"]:
                    partition = algo_func(g).as_clustering()
                else:
                    partition = algo_func(g)
            except Exception as e:
                print(f"  Warning: Community detection failed for graph {graph_idx}: {e}")
                communities_list.append(None)
                continue
            
            communities_list.append(partition)
            
        except Exception as e:
            print(f"  Warning: Error processing graph {graph_idx}: {e}")
            communities_list.append(None)

    # Create animation frames
    frames = []
    
    for frame_idx, (graph, partition) in enumerate(zip(graphs, communities_list)):
        
        if partition is None:
            print(f"  Skipping frame {frame_idx} due to detection failure")
            continue
        
        try:
            # Get node positions
            if "x" in graph.vs.attributes() and "y" in graph.vs.attributes():
                pos = [(node["x"], node["y"]) for node in graph.vs]
            else:
                # Assign random positions
                pos = [(random.uniform(0, 1), random.uniform(0, 1)) 
                       for _ in graph.vs]
                print(f"  Note: Frame {frame_idx} using random positions (no x/y attributes)")
            
            # Get community membership
            try:
                community_membership = partition.membership
            except AttributeError:
                # partition is already a Clustering object
                community_membership = list(partition.membership)
            
            # Create node trace with community colors
            node_trace = go.Scatter(
                x=[p[0] for p in pos],
                y=[p[1] for p in pos],
                mode="markers",
                marker=dict(
                    size=10,
                    color=community_membership,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Community")
                ),
                text=[f"Node {i} (Community {community_membership[i]})" 
                      for i in range(len(pos))],
                hoverinfo="text",
                name=f"Frame {frame_idx + 1}"
            )
            
            # Create edge trace
            edge_x = []
            edge_y = []
            for edge in graph.es:
                source = edge.source
                target = edge.target
                edge_x.extend([pos[source][0], pos[target][0], None])
                edge_y.extend([pos[source][1], pos[target][1], None])
            
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines",
                name=f"Edges {frame_idx + 1}"
            )
            
            # Create frame
            frame = go.Frame(
                data=[edge_trace, node_trace],
                name=f"Frame {frame_idx + 1}"
            )
            frames.append(frame)
            
        except Exception as e:
            print(f"  Warning: Could not create visualization for frame {frame_idx}: {e}")
            continue

    if not frames:
        raise RuntimeError("No frames were successfully created. "
                          "Check that community detection worked correctly.")

    # Create layout with animation controls
    layout = go.Layout(
        title=f"Community Evolution ({algo_name})",
        xaxis=dict(title="X coordinate", showgrid=False, zeroline=False),
        yaxis=dict(title="Y coordinate", showgrid=False, zeroline=False),
        showlegend=False,
        hovermode="closest",
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True,
                        }],
                        "label": "▶ Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                        }],
                        "label": "⏸ Pause",
                        "method": "animate"
                    },
                    {
                        "args": [["Frame 1"], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                        }],
                        "label": "⏮ Restart",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": True,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 1.15,
                "yanchor": "top"
            }
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "y": 0,
            "xanchor": "left",
            "x": 0.1,
            "len": 0.9,
            "transition": {"duration": 300},
            "pad": {"b": 10, "t": 50},
            "currentvalue": {
                "prefix": "Time: ",
                "visible": True,
                "xanchor": "right"
            },
            "steps": [
                {
                    "args": [[f.name], {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}
                    }],
                    "method": "animate",
                    "label": f.name
                }
                for f in frames
            ]
        }]
    )

    # Create initial figure with first frame data
    initial_data = frames[0].data if frames else []
    
    fig = go.Figure(
        data=initial_data,
        layout=layout,
        frames=frames
    )

    # Save and display
    plot(fig, filename=output_file, auto_open=False)
    print(f"✓ Community evolution animation saved to {output_file}")
