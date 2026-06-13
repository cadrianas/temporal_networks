"""
Temporal Network Analysis: Community Evolution Visualization Module

This module provides functions for creating interactive animated visualizations
of how community structures change over time in networks.
"""

import igraph as ig
import plotly.graph_objs as go
from plotly.offline import plot
import random
from typing import List, Tuple, Any
from ._gap_utilities import validate_and_setup_graphs


def _detect_communities(graphs: List,
                        community_algorithm: str) -> Tuple[List[Any], str]:
    """
    Detect communities for each graph using the chosen algorithm.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Graphs on which to run community detection.
    community_algorithm : str
        Name of the community detection algorithm to apply.

    Returns
    -------
    tuple of (list, str)
        The per-graph partitions (None where detection failed) and the
        capitalized algorithm name.
    """
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

    if community_algorithm.lower() not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {community_algorithm}. "
                        f"Must be one of: {list(algorithm_map.keys())}")

    algo_func = algorithm_map[community_algorithm.lower()]
    algo_name = community_algorithm.capitalize()

    print(f"Computing community evolution with {algo_name} algorithm...")

    communities_list = []
    for graph_idx, graph in enumerate(graphs):
        try:
            g = graph.copy()
            if g.is_directed() and community_algorithm.lower() in ["walktrap",
                                                                    "fast_greedy",
                                                                    "label_prop",
                                                                    "spinglass"]:
                g = g.as_undirected()

            try:
                if community_algorithm.lower() in ["walktrap", "fast_greedy"]:
                    partition = algo_func(g).as_clustering()
                else:
                    partition = algo_func(g)
            except Exception as e:
                print(f"  Warning: Community detection failed for "
                      f"graph {graph_idx}: {e}")
                communities_list.append(None)
                continue

            communities_list.append(partition)

        except Exception as e:
            print(f"  Warning: Error processing graph {graph_idx}: {e}")
            communities_list.append(None)

    return communities_list, algo_name


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
    >>> import random
    >>> import igraph as ig
    >>> from temporal_networks import plot_community_evolution
    >>> ig.set_random_number_generator(random.Random(42))
    >>> graphs = [ig.Graph.Barabasi(n=50, m=2) for _ in range(12)]
    >>> plot_community_evolution(
    ...     graphs, community_algorithm="louvain")  # doctest: +SKIP

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

    if not graphs:
        raise ValueError("graphs list cannot be empty")

    # Validate inputs
    validate_and_setup_graphs(graphs)

    communities_list, algo_name = _detect_communities(graphs, community_algorithm)

    frames = _create_animation_frames(graphs, communities_list)

    if not frames:
        raise RuntimeError("No frames were successfully created. "
                          "Check that community detection worked correctly.")

    layout = _create_animation_layout(algo_name, frames)

    initial_data = frames[0].data if frames else []

    fig = go.Figure(
        data=initial_data,
        layout=layout,
        frames=frames
    )

    plot(fig, filename=output_file, auto_open=False)
    print(f"✓ Community evolution animation saved to {output_file}")


def _create_animation_frames(graphs: List,
                             communities_list: List[Any]) -> List[go.Frame]:
    """
    Create Plotly animation frames from graphs and their partitions.

    Parameters
    ----------
    graphs : list of igraph.Graph
        Graphs to render, one per animation frame.
    communities_list : list
        Community partition for each graph (None entries are skipped).

    Returns
    -------
    list of plotly.graph_objs.Frame
        One frame per successfully rendered graph.
    """
    frames = []

    for frame_idx, (graph, partition) in enumerate(zip(graphs, communities_list)):
        if partition is None:
            print(f"  Skipping frame {frame_idx} due to detection failure")
            continue

        try:
            if "x" in graph.vs.attributes() and "y" in graph.vs.attributes():
                pos = [(node["x"], node["y"]) for node in graph.vs]
            else:
                pos = [(random.uniform(0, 1), random.uniform(0, 1))
                       for _ in graph.vs]
                print(f"  Note: Frame {frame_idx} using random "
                      f"positions (no x/y attributes)")

            try:
                community_membership = partition.membership
            except AttributeError:
                community_membership = list(partition.membership)

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

            edges = graph.get_edgelist()
            pos_x = [p[0] for p in pos]
            pos_y = [p[1] for p in pos]

            edge_x = []
            edge_y = []
            app_x = edge_x.append
            app_y = edge_y.append

            for u, v in edges:
                app_x(pos_x[u])
                app_x(pos_x[v])
                app_x(None)
                app_y(pos_y[u])
                app_y(pos_y[v])
                app_y(None)

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines",
                name=f"Edges {frame_idx + 1}"
            )

            frame = go.Frame(
                data=[edge_trace, node_trace],
                name=f"Frame {frame_idx + 1}"
            )
            frames.append(frame)

        except Exception as e:
            print(f"  Warning: Could not create visualization for "
                  f"frame {frame_idx}: {e}")
            continue

    return frames


def _create_animation_layout(algo_name: str, frames: List[go.Frame]) -> go.Layout:
    """
    Build the Plotly layout with animation playback controls.

    Parameters
    ----------
    algo_name : str
        Algorithm name shown in the plot title.
    frames : list of plotly.graph_objs.Frame
        Animation frames, used to build the slider steps.

    Returns
    -------
    plotly.graph_objs.Layout
        Layout configured with play/pause/restart buttons and a slider.
    """
    return go.Layout(
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
