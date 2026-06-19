"""
Multi-gap end-to-end example that SAVES every result for inspection.

A second, richer integration run than ``example_full_integration.py``. It
builds one temporal network with **two** temporal gaps and threads it through
every public function, writing every output (DataFrames -> CSV, plots -> PDF/
HTML, scalars -> a text summary) into an output directory so the results can
be opened and checked by hand.

Dataset (12 named nodes, communities L = n0..n5, R = n6..n11):
- 19 snapshots spanning 2024-01 .. 2026-07 (31 months) split into FIVE
  segments by FOUR gaps of varying length (1, 3, 6 and 2 months),
- a long gap-free opening segment (2024-01..06) carrying intermittent chord
  edges whose irregular timing makes burstiness non-degenerate (a spread of
  B values strictly between -1 and 0, alongside the always-on clique edges
  at B = -1),
- broker snapshots (a single n5-n6 bridge in 2024-02 and again in 2025-04),
- an in-segment community MERGE (2024-09) and SPLIT (2024-11), plus a
  second MERGE (2026-02),
- a sparse ANOMALY snapshot (2025-06),
- per-snapshot vertex shuffling so identity is tested by name.

Run (writes to ./integration_results by default):

    python examples/example_multigap_report.py [output_dir]
"""

import os
import sys
import random

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

SEED = 7
L = [f"n{i}" for i in range(6)]        # community L: n0..n5
R = [f"n{i}" for i in range(6, 12)]    # community R: n6..n11

_COVERED: set = set()
_SCALARS: list = []


def _cover(name: str) -> None:
    _COVERED.add(name)


def _clique(nodes):
    return [(nodes[a], nodes[b])
            for a in range(len(nodes))
            for b in range(a + 1, len(nodes))]


# Intermittent intra-community edges, each mapped to the snapshots on which it
# is active. Their irregular within-segment timing makes the burstiness
# distribution non-degenerate: a clique edge that fires every snapshot is
# perfectly regular (B = -1), while these land strictly between -1 and 0.
# All of these patterns live inside the long opening segment (2024-01..06),
# which is gap-free so the intervals are not discarded as gap-spanning.
SPORADIC = {
    # L-community chords
    ("n1", "n3"): {"2024-01", "2024-02", "2024-03", "2024-06"},  # 1,1,3
    ("n2", "n4"): {"2024-01", "2024-04", "2024-06"},             # 3,2
    ("n0", "n4"): {"2024-01", "2024-05", "2024-06"},             # 4,1
    # R-community chords
    ("n7", "n10"): {"2024-01", "2024-02", "2024-06"},            # 1,4
    ("n8", "n11"): {"2024-02", "2024-03", "2024-06"},            # 1,3
    ("n6", "n9"): {"2024-01", "2024-03", "2024-04", "2024-06"},  # 2,1,2
}


def _month(date: str) -> str:
    """'2024-01-15' -> '2024-01' (the label snapshots_from_events emits)."""
    return date[:7]


def _dense(nodes, date):
    """Clique over ``nodes``, dropping sporadic chords inactive on ``date``."""
    month = _month(date)
    out = []
    for edge in _clique(nodes):
        if edge in SPORADIC and month not in SPORADIC[edge]:
            continue
        out.append(edge)
    return out


def _build_events() -> pd.DataFrame:
    allnodes = L + R

    # (date, kind). kind drives the snapshot's structure:
    #   two     -> two communities (sporadic chords per their schedule)
    #   broker  -> two communities + a single n5-n6 bridge
    #   merge   -> one dense community over all nodes
    #   anomaly -> structure collapses to a single edge
    plan = [
        # --- segment 1 (6 snapshots, gap-free): hosts the bursty chords ---
        ("2024-01-15", "two"),
        ("2024-02-15", "broker"),
        ("2024-03-15", "two"),
        ("2024-04-15", "two"),
        ("2024-05-15", "two"),
        ("2024-06-15", "two"),
        # GAP 1 (1 month): 2024-07 missing
        # --- segment 2: in-segment MERGE then SPLIT ---
        ("2024-08-15", "two"),
        ("2024-09-15", "merge"),
        ("2024-10-15", "merge"),
        ("2024-11-15", "two"),     # split
        # GAP 2 (3 months): 2024-12, 2025-01, 2025-02 missing
        # --- segment 3: two communities, a broker, then an anomaly ---
        ("2025-03-15", "two"),
        ("2025-04-15", "broker"),
        ("2025-05-15", "two"),
        ("2025-06-15", "anomaly"),
        # GAP 3 (6 months): 2025-07 .. 2025-12 missing
        # --- segment 4: a second MERGE ---
        ("2026-01-15", "two"),
        ("2026-02-15", "merge"),
        ("2026-03-15", "merge"),
        # GAP 4 (2 months): 2026-04, 2026-05 missing
        # --- segment 5: back to two communities ---
        ("2026-06-15", "two"),
        ("2026-07-15", "two"),
    ]

    rows = []
    for date, kind in plan:
        if kind == "anomaly":
            edges = [("n0", "n1")]
        elif kind == "merge":
            # Fully complete so the two communities unambiguously collapse
            # into one (a near-complete graph can still split under louvain).
            edges = _clique(allnodes)
        else:  # two or broker
            edges = _dense(L, date) + _dense(R, date)
            if kind == "broker":
                edges = edges + [("n5", "n6")]
        for src, tgt in edges:
            rows.append({"time": date, "source": src, "target": tgt})
    return pd.DataFrame(rows)


def _shuffle_vertices(graphs):
    rng = random.Random(SEED)
    out = []
    for g in graphs:
        perm = list(range(g.vcount()))
        rng.shuffle(perm)
        out.append(g.permute_vertices(perm))
    return out


def _save(df: pd.DataFrame, name: str, outdir: str, preview: int = 12) -> None:
    """Write a DataFrame to CSV and echo a preview to stdout."""
    path = os.path.join(outdir, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"\n### {name}  ->  {name}.csv  ({len(df)} rows)")
    with pd.option_context("display.width", 120,
                           "display.max_columns", 20):
        print(df.head(preview).to_string(index=False))


def main() -> None:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "integration_results"
    os.makedirs(outdir, exist_ok=True)
    ig.set_random_number_generator(random.Random(SEED))

    print("=" * 72)
    print("Multi-gap integration report")
    print(f"Saving all results under: {os.path.abspath(outdir)}")
    print("=" * 72)

    events = _build_events()
    events.to_csv(os.path.join(outdir, "events.csv"), index=False)

    # ---- ingestion ------------------------------------------------------
    graphs0, labels = snapshots_from_events(
        events, time_col="time", source_col="source", target_col="target")
    _cover("snapshots_from_events")
    graphs_el, _ = snapshots_from_edgelist(
        os.path.join(outdir, "events.csv"),
        time_col="time", source_col="source", target_col="target")
    _cover("snapshots_from_edgelist")
    graphs = _shuffle_vertices(graphs0)

    print(f"\nSnapshots ({len(graphs)}): {labels}")
    _SCALARS.append(f"snapshots = {labels}")

    # ---- gaps -----------------------------------------------------------
    gap_info = detect_temporal_gaps(labels)
    _cover("detect_temporal_gaps")
    print(f"\nGaps detected: {gap_info['num_gaps']}  "
          f"segments={gap_info['segments']}")
    _SCALARS.append(f"num_gaps = {gap_info['num_gaps']}")
    _SCALARS.append(f"segments = {gap_info['segments']}")
    for g in gap_info["gaps"]:
        _SCALARS.append(
            f"gap: {g['start_label']} -> {g['end_label']} "
            f"({g['gap_size']} months)")

    # ---- snapshot-level metrics -----------------------------------------
    props = network_properties(graphs, graph_labels=labels,
                               visualisation=False, report_gaps=False)
    _cover("network_properties")
    _save(props[["Graph", "Number of Nodes", "Number of Edges", "Density"]],
          "network_properties", outdir)

    cents = calculate_centralities(graphs, graph_labels=labels,
                                   report_gaps=False)
    _cover("calculate_centralities")
    _save(cents, "centralities", outdir, preview=14)

    comm = communities_measures(graphs, graph_labels=labels,
                                visualisation=False, report_gaps=False)
    _cover("communities_measures")
    _SCALARS.append(f"communities_measures algorithms = {list(comm.keys())}")
    print(f"\n### communities_measures -> algorithms {list(comm.keys())}")

    vprops = vertex_properties(graphs, node_name="n0", graph_labels=labels,
                               visualisation=False, report_gaps=False)
    _cover("vertex_properties")
    _save(vprops, "vertex_n0_properties", outdir)

    # ---- edge dynamics --------------------------------------------------
    dyn = compute_edge_dynamics(graphs, graph_labels=labels)
    _cover("compute_edge_dynamics")
    _save(dyn, "edge_dynamics", outdir)
    formed = edge_formation(graphs, graph_labels=labels, report_gaps=False)
    _cover("edge_formation")
    _save(formed, "edge_formation", outdir)
    dissolved = edge_dissolution(graphs, graph_labels=labels,
                                 report_gaps=False)
    _cover("edge_dissolution")
    _save(dissolved, "edge_dissolution", outdir)
    plot_edge_dynamics(dyn, labels, gap_info, metric="Edges_Formed",
                       save_path=outdir)
    _cover("plot_edge_dynamics")

    # ---- stability ------------------------------------------------------
    sim = snapshot_similarity(graphs, graph_labels=labels, report_gaps=False)
    _cover("snapshot_similarity")
    _save(sim, "snapshot_similarity", outdir)
    tcc = temporal_correlation_coefficient(graphs, graph_labels=labels)
    _cover("temporal_correlation_coefficient")
    _SCALARS.append(f"temporal_correlation_coefficient = {tcc:.4f}")

    # ---- burstiness -----------------------------------------------------
    iet = inter_event_times(graphs, graph_labels=labels, by="edge")
    _cover("inter_event_times")
    _save(iet, "inter_event_times", outdir)
    burst = burstiness_coefficient(graphs, graph_labels=labels, by="edge",
                                   report_gaps=False, save_path=outdir)
    _cover("burstiness_coefficient")
    # Show the spread, not just the degenerate B = -1 backbone.
    burst = burst.sort_values("burstiness").reset_index(drop=True)
    _save(burst, "burstiness", outdir)
    b_dist = burst["burstiness"].round(3).value_counts().sort_index()
    print("\nBurstiness distribution (rounded B -> #edges):")
    print(b_dist.to_string())
    _SCALARS.append(
        f"burstiness distinct values = {burst['burstiness'].nunique()}, "
        f"range = [{burst['burstiness'].min():.3f}, "
        f"{burst['burstiness'].max():.3f}]")

    # ---- temporal paths -------------------------------------------------
    reach = temporal_reachability(graphs, graph_labels=labels)
    _cover("temporal_reachability")
    _save(reach.head(20), "temporal_reachability_head", outdir, preview=20)
    reach.to_csv(os.path.join(outdir, "temporal_reachability_full.csv"),
                 index=False)
    dist = temporal_distances(graphs, graph_labels=labels)
    _cover("temporal_distances")
    dist.to_csv(os.path.join(outdir, "temporal_distances_full.csv"),
                index=False)
    close = temporal_closeness(graphs, graph_labels=labels,
                               report_gaps=False, save_path=outdir)
    _cover("temporal_closeness")
    _save(close, "temporal_closeness", outdir)
    eff = temporal_efficiency(graphs, graph_labels=labels)
    _cover("temporal_efficiency")
    _SCALARS.append(f"temporal_efficiency = {eff:.4f}")
    betw = temporal_betweenness(graphs, graph_labels=labels,
                                report_gaps=False, save_path=outdir)
    _cover("temporal_betweenness")
    _save(betw, "temporal_betweenness", outdir)

    # ---- change points / anomalies --------------------------------------
    cp = detect_change_points(props, method="diff", threshold=3.0,
                              gap_info=gap_info)
    _cover("detect_change_points")
    _save(cp, "change_points", outdir)
    flags = flag_anomalous_snapshots(graphs, graph_labels=labels,
                                     method="diff", threshold=3.0)
    _cover("flag_anomalous_snapshots")
    _save(flags, "anomalous_snapshots", outdir)

    # ---- community tracking ---------------------------------------------
    track = track_communities(graphs, graph_labels=labels,
                              algorithm="louvain", match_threshold=0.3,
                              bridge_gaps=False, report_gaps=False)
    _cover("track_communities")
    track_to_save = track.copy()
    track_to_save["members"] = track_to_save["members"].apply(
        lambda m: "|".join(m))
    _save(track_to_save, "community_tracking", outdir, preview=30)
    plot_community_lineage(track, labels, gap_info, save_path=outdir)
    _cover("plot_community_lineage")

    # ---- community evolution animation ----------------------------------
    plot_community_evolution(
        graphs, community_algorithm="louvain",
        output_file=os.path.join(outdir, "community_evolution.html"))
    _cover("plot_community_evolution")

    # ---- scalar summary + coverage --------------------------------------
    with open(os.path.join(outdir, "scalars.txt"), "w") as fh:
        fh.write("\n".join(_SCALARS) + "\n")

    public = list(tn.__all__)
    missing = [n for n in public if n not in _COVERED]
    cov_lines = [f"{'covered' if n in _COVERED else 'NOT COVERED'}: {n}"
                 for n in public]
    with open(os.path.join(outdir, "coverage.txt"), "w") as fh:
        fh.write("\n".join(cov_lines) + "\n")

    print("\n" + "=" * 72)
    print("SCALAR RESULTS")
    print("=" * 72)
    for line in _SCALARS:
        print(f"  {line}")

    print("\n" + "=" * 72)
    print(f"COVERAGE: {len(public) - len(missing)}/{len(public)} "
          f"public functions exercised")
    if missing:
        print(f"  NOT COVERED: {missing}")
    print("=" * 72)

    # Hand-checkable invariants across the FOUR gaps.
    re_entry = ["2024-08", "2025-03", "2026-01", "2026-06"]
    nan_rows = sim[sim["jaccard"].isna()]["Graph"].tolist()
    print("\nChecks:")
    print(f"  gaps detected: {gap_info['num_gaps']}")
    print(f"  similarity NaN at gap re-entry snapshots: {nan_rows}")
    assert gap_info["num_gaps"] == 4, "expected exactly four gaps"
    assert all(lbl in nan_rows for lbl in re_entry), \
        "every gap re-entry should yield NaN similarity"
    assert "2025-06" in set(flags["label"]), "anomaly 2025-06 not flagged"
    assert "merge" in set(track[track["Graph"] == "2024-09"]["event"]), \
        "expected a merge at 2024-09"
    assert "split" in set(track[track["Graph"] == "2024-11"]["event"]), \
        "expected a split at 2024-11"
    assert "merge" in set(track[track["Graph"] == "2026-02"]["event"]), \
        "expected a second merge at 2026-02"
    n_regular = int((burst["burstiness"] == -1.0).sum())
    n_irregular = int((burst["burstiness"] > -1.0).sum())
    assert burst["burstiness"].nunique() >= 4 and n_irregular >= 5, \
        "burstiness should be non-degenerate (not all -1)"
    assert float(burst[burst["entity"] == "('n0', 'n1')"]
                 ["burstiness"].iloc[0]) == -1.0, \
        "the always-on edge (n0,n1) should be perfectly regular (B=-1)"
    print("  [OK] four gaps, each producing NaN similarity at re-entry")
    print("  [OK] anomaly 2025-06 flagged")
    print("  [OK] merge at 2024-09, split at 2024-11, merge at 2026-02 "
          "tracked")
    print(f"  [OK] burstiness non-degenerate: {n_irregular} irregular edges "
          f"(B > -1), {n_regular} perfectly regular (B = -1)")

    print(f"\nAll files saved under: {os.path.abspath(outdir)}")
    files = sorted(os.listdir(outdir))
    print(f"({len(files)} files): {files}")
    print("Done. ✅")


if __name__ == "__main__":
    main()
