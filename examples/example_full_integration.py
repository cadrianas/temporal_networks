"""
End-to-end integration toy example (audit area 8).

Builds ONE deliberately non-trivial temporal network and threads it through
*every* public function in ``temporal_networks.__all__``. This is an
integration smoke test: it catches wiring bugs the isolated unit tests miss
(broken imports, signature drift, output schemas that don't compose).

The dataset on purpose includes:
- 10 named nodes in two communities L = n0..n4 and R = n5..n9,
- 8 snapshots with a genuine temporal gap (2024-06 is missing),
- per-snapshot vertex-order shuffling, so node identity is tested by name
  rather than by index alignment,
- a single inter-community bridge (n4-n5) present in only one snapshot, which
  makes those two nodes the unique temporal brokers,
- a sparse anomalous snapshot (2024-05) whose structure collapses,
- a regularly active edge (n0, n1) for a clean burstiness signal.

It seeds every RNG and asserts hand-checkable invariants rather than only
printing shapes. Run from an installed package:

    python examples/example_full_integration.py
"""

import os
import random
import tempfile

import igraph as ig
import pandas as pd

import temporal_networks as tn
from temporal_networks import (
    snapshots_from_events,
    snapshots_from_edgelist,
    network_properties,
    calculate_centralities,
    communities_measures,
    vertex_properties,
    compute_edge_dynamics,
    edge_formation,
    edge_dissolution,
    plot_edge_dynamics,
    plot_community_evolution,
    detect_temporal_gaps,
    snapshot_similarity,
    temporal_correlation_coefficient,
    inter_event_times,
    burstiness_coefficient,
    temporal_reachability,
    temporal_distances,
    temporal_closeness,
    temporal_efficiency,
    temporal_betweenness,
    detect_change_points,
    flag_anomalous_snapshots,
    track_communities,
    plot_community_lineage,
)

SEED = 42
L = [f"n{i}" for i in range(5)]       # community L
R = [f"n{i}" for i in range(5, 10)]   # community R

# Track which public functions we exercise, for a coverage report.
_COVERED: set = set()


def _cover(name: str) -> None:
    _COVERED.add(name)


def _clique_edges(nodes):
    """All undirected pairs within ``nodes``."""
    return [(nodes[a], nodes[b])
            for a in range(len(nodes))
            for b in range(a + 1, len(nodes))]


def _build_events() -> pd.DataFrame:
    """One long-form event table driving the whole example."""
    # date -> list of (source, target) active that month
    schedule = {
        "2024-01-15": _clique_edges(L) + _clique_edges(R),
        "2024-02-15": _clique_edges(L) + _clique_edges(R),
        # bridge n4-n5 appears only here -> the sole inter-community link
        "2024-03-15": _clique_edges(L) + _clique_edges(R) + [("n4", "n5")],
        "2024-04-15": _clique_edges(L) + _clique_edges(R),
        # anomalous snapshot: structure collapses to a single edge
        "2024-05-15": [("n0", "n1")],
        # 2024-06 deliberately absent -> temporal gap
        "2024-07-15": _clique_edges(L) + _clique_edges(R),
        "2024-08-15": _clique_edges(L) + _clique_edges(R),
        "2024-09-15": _clique_edges(L) + _clique_edges(R),
    }
    rows = []
    for date, edges in schedule.items():
        for src, tgt in edges:
            rows.append({"time": date, "source": src, "target": tgt})
    return pd.DataFrame(rows)


def _shuffle_vertices(graphs):
    """Permute each snapshot's vertices independently (names preserved).

    Ingestion gives every snapshot the same union vertex order, so we shuffle
    to force node identity to be resolved by name, not by index.
    """
    rng = random.Random(SEED)
    shuffled = []
    for g in graphs:
        perm = list(range(g.vcount()))
        rng.shuffle(perm)
        shuffled.append(g.permute_vertices(perm))
    return shuffled


def _check(label: str, condition: bool) -> None:
    status = "OK  " if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(f"Invariant failed: {label}")


def main() -> None:
    ig.set_random_number_generator(random.Random(SEED))
    print("=" * 70)
    print("Full integration toy example — exercising every public function")
    print("=" * 70)

    tmp = tempfile.mkdtemp(prefix="tn_toy_")
    events = _build_events()

    # ---- Spec 01: ingestion (the shared dataset originates here) --------
    print("\n[ingestion]")
    graphs0, labels = snapshots_from_events(
        events, time_col="time", source_col="source", target_col="target")
    _cover("snapshots_from_events")

    csv_path = os.path.join(tmp, "events.csv")
    events.to_csv(csv_path, index=False)
    graphs_el, labels_el = snapshots_from_edgelist(
        csv_path, time_col="time", source_col="source", target_col="target")
    _cover("snapshots_from_edgelist")

    _check("8 snapshots ingested", len(graphs0) == 8)
    _check("gap month 2024-06 absent from labels",
           "2024-06" not in labels and labels == labels_el)

    # Shuffle vertices per snapshot -> identity must be resolved by name.
    graphs = _shuffle_vertices(graphs0)

    def _name_index(g):
        return {name: idx for idx, name in enumerate(g.vs["name"])}

    idx_first = _name_index(graphs[0])
    idx_third = _name_index(graphs[2])
    _check("vertex order differs across snapshots (identity by name)",
           idx_first != idx_third)

    # ---- Gaps -----------------------------------------------------------
    print("\n[gaps]")
    gap_info = detect_temporal_gaps(labels)
    _cover("detect_temporal_gaps")
    _check("exactly one temporal gap detected", gap_info["num_gaps"] == 1)

    # ---- Snapshot-level structural metrics ------------------------------
    print("\n[snapshot metrics]")
    props = network_properties(graphs, graph_labels=labels,
                               visualisation=False, report_gaps=False)
    _cover("network_properties")
    cents = calculate_centralities(graphs, graph_labels=labels,
                                   report_gaps=False)
    _cover("calculate_centralities")
    comm = communities_measures(graphs, graph_labels=labels,
                                visualisation=False, report_gaps=False)
    _cover("communities_measures")
    vprops = vertex_properties(graphs, node_name="n0", graph_labels=labels,
                               visualisation=False, report_gaps=False)
    _cover("vertex_properties")
    _check("network_properties has one row per snapshot",
           len(props) == len(graphs))
    _check("centralities/communities/vertex outputs non-empty",
           len(cents) > 0 and len(comm) > 0 and len(vprops) > 0)

    # ---- Edge dynamics --------------------------------------------------
    print("\n[edge dynamics]")
    dyn = compute_edge_dynamics(graphs, graph_labels=labels)
    _cover("compute_edge_dynamics")
    formed = edge_formation(graphs, graph_labels=labels, report_gaps=False)
    _cover("edge_formation")
    dissolved = edge_dissolution(graphs, graph_labels=labels,
                                 report_gaps=False)
    _cover("edge_dissolution")
    plot_edge_dynamics(dyn, labels, gap_info, metric="Edges_Formed",
                       save_path=tmp)
    _cover("plot_edge_dynamics")
    _check("edge dynamics computed for transitions",
           len(formed) > 0 and len(dissolved) > 0)

    # ---- Snapshot stability ---------------------------------------------
    print("\n[stability]")
    sim = snapshot_similarity(graphs, graph_labels=labels, report_gaps=False)
    _cover("snapshot_similarity")
    tcc = temporal_correlation_coefficient(graphs, graph_labels=labels)
    _cover("temporal_correlation_coefficient")
    gap_row = sim[sim["Graph"] == "2024-07"]
    _check("similarity across the gap is NaN",
           bool(gap_row["jaccard"].isna().iloc[0]))
    _check("temporal correlation coefficient in [0, 1]",
           0.0 <= tcc <= 1.0)

    # ---- Burstiness -----------------------------------------------------
    print("\n[burstiness]")
    iet = inter_event_times(graphs, graph_labels=labels, by="edge")
    _cover("inter_event_times")
    burst = burstiness_coefficient(graphs, graph_labels=labels, by="edge",
                                   report_gaps=False)
    _cover("burstiness_coefficient")
    n0n1 = burst[burst["entity"] == "('n0', 'n1')"]
    _check("regularly active edge (n0,n1) has burstiness -1",
           not n0n1.empty and float(n0n1["burstiness"].iloc[0]) == -1.0)
    _check("inter-event intervals were produced", len(iet) > 0)

    # ---- Time-respecting paths ------------------------------------------
    print("\n[temporal paths]")
    reach = temporal_reachability(graphs, graph_labels=labels)
    _cover("temporal_reachability")
    dist = temporal_distances(graphs, graph_labels=labels)
    _cover("temporal_distances")
    _check("temporal distances produced for all node pairs",
           len(dist) == 100)
    close = temporal_closeness(graphs, graph_labels=labels, report_gaps=False)
    _cover("temporal_closeness")
    eff = temporal_efficiency(graphs, graph_labels=labels)
    _cover("temporal_efficiency")
    betw = temporal_betweenness(graphs, graph_labels=labels,
                                report_gaps=False)
    _cover("temporal_betweenness")

    def _reach(src, tgt):
        row = reach[(reach["source"] == src) & (reach["target"] == tgt)]
        return bool(row["reachable"].iloc[0])

    _check("cross-community pair n0->n9 reachable via the n4-n5 bridge",
           _reach("n0", "n9"))

    def _betw(node):
        return float(betw[betw["node"] == node]["betweenness"].iloc[0])

    top_two = set(betw.head(2)["node"])
    _check("the two bridge endpoints n4,n5 are the top temporal brokers",
           top_two == {"n4", "n5"})
    _check("bridges broker strictly more than a clique-internal node",
           _betw("n4") > _betw("n0") and _betw("n5") > _betw("n0"))
    _check("global temporal efficiency is positive", eff > 0.0)
    _check("closeness produced one row per node", len(close) == 10)

    # ---- Change points / anomalies --------------------------------------
    print("\n[change points]")
    cp = detect_change_points(props, method="diff", threshold=3.0,
                              gap_info=gap_info)
    _cover("detect_change_points")
    flags = flag_anomalous_snapshots(graphs, graph_labels=labels,
                                     method="diff", threshold=3.0)
    _cover("flag_anomalous_snapshots")
    _check("anomalous snapshot 2024-05 is flagged",
           "2024-05" in set(flags["label"]) and
           "2024-05" in set(cp["label"]))

    # ---- Community tracking ---------------------------------------------
    print("\n[community tracking]")
    track = track_communities(graphs, graph_labels=labels,
                              algorithm="louvain", match_threshold=0.3,
                              bridge_gaps=False, report_gaps=False)
    _cover("track_communities")
    plot_community_lineage(track, labels, gap_info, save_path=tmp)
    _cover("plot_community_lineage")
    post_gap_events = set(track[track["Graph"] == "2024-07"]["event"])
    _check("communities after the gap start fresh lineages (birth)",
           "birth" in post_gap_events)

    # ---- Community evolution animation ----------------------------------
    print("\n[community evolution plot]")
    html_path = os.path.join(tmp, "evolution.html")
    plot_community_evolution(graphs, community_algorithm="louvain",
                             output_file=html_path)
    _cover("plot_community_evolution")
    _check("community-evolution HTML was written",
           os.path.exists(html_path))

    # ---- Coverage report ------------------------------------------------
    print("\n" + "=" * 70)
    print("Public-function coverage")
    print("=" * 70)
    public = [n for n in tn.__all__]
    missing = [n for n in public if n not in _COVERED]
    for name in public:
        mark = "covered    " if name in _COVERED else "NOT COVERED"
        print(f"  [{mark}] {name}")
    _check(f"all {len(public)} public functions exercised", not missing)

    print(f"\nArtifacts written under: {tmp}")
    print("=" * 70)
    print("Integration example passed: all functions ran, all invariants "
          "held. ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
