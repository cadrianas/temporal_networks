"""
Smoke test / walkthrough for specs 01, 02, 03.

Builds a small named-node sequence with a deliberate gap (no 2024-05),
then exercises every public function added by the three specs:

  Spec 01 — snapshots_from_events, snapshots_from_edgelist
  Spec 02 — snapshot_similarity, temporal_correlation_coefficient
  Spec 03 — inter_event_times, burstiness_coefficient

Run:  .venv-audit/bin/python examples/example_specs_01_02_03.py
"""

import random
import tempfile
import os

import igraph as ig
import pandas as pd

from temporal_networks import (
    snapshots_from_events,
    snapshots_from_edgelist,
    snapshot_similarity,
    temporal_correlation_coefficient,
    inter_event_times,
    burstiness_coefficient,
)

# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------

def _make_snapshots(seed: int = 42):
    """Return (graphs, labels): 6 named-node snapshots, gap at 2024-05."""
    rng = random.Random(seed)
    ig.set_random_number_generator(rng)

    nodes = [f"city_{i}" for i in range(8)]
    labels = ["2024-01", "2024-02", "2024-03", "2024-04",
              "2024-06", "2024-07"]  # 2024-05 missing on purpose

    graphs = []
    for _ in labels:
        g = ig.Graph()
        g.add_vertices(nodes)
        possible = [(a, b) for i, a in enumerate(nodes)
                    for b in nodes[i + 1:]]
        rng.shuffle(possible)
        g.add_edges(possible[: rng.randint(5, 10)])
        perm = list(range(len(nodes)))
        rng.shuffle(perm)
        graphs.append(g.permute_vertices(perm))
    return graphs, labels


def _make_events_df():
    """Long-form edge-list (spec 01 input): 3 cities, 6 months, gap at May."""
    rows = []
    active = {
        "2024-01": [("city_A", "city_B"), ("city_B", "city_C")],
        "2024-02": [("city_A", "city_C")],
        "2024-03": [("city_A", "city_B"), ("city_A", "city_C")],
        "2024-04": [("city_B", "city_C")],
        # 2024-05 missing
        "2024-06": [("city_A", "city_B"), ("city_B", "city_C"),
                    ("city_A", "city_C")],
    }
    for t, edges in active.items():
        for src, tgt in edges:
            rows.append({"time": t, "source": src, "target": tgt})
    return pd.DataFrame(rows)


def _make_edgelist_df():
    """Multi-month CSV-style edge list (spec 01 snapshots_from_edgelist)."""
    rows = []
    for month, pairs in [
        ("2024-01", [("A", "B"), ("B", "C")]),
        ("2024-02", [("A", "C")]),
        ("2024-03", [("A", "B")]),
    ]:
        for src, tgt in pairs:
            rows.append({"month": month, "u": src, "v": tgt})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Specs 01, 02, 03 — end-to-end smoke test")
    print("6 snapshots, named vertices, deliberate gap at 2024-05")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Spec 01 — event ingestion                                           #
    # ------------------------------------------------------------------ #
    print("\n--- Spec 01: event ingestion ---")

    events_df = _make_events_df()
    graphs_ev, labels_ev = snapshots_from_events(
        events_df,
        time_col="time",
        source_col="source",
        target_col="target",
        directed=False,
    )
    print(f"[snapshots_from_events]  {len(graphs_ev)} snapshots, "
          f"labels={labels_ev}")

    edgelist_df = _make_edgelist_df()
    with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", delete=False) as fh:
        edgelist_csv = fh.name
        edgelist_df.to_csv(fh, index=False)
    try:
        graphs_el, labels_el = snapshots_from_edgelist(
            edgelist_csv,
            time_col="month",
            source_col="u",
            target_col="v",
        )
    finally:
        os.unlink(edgelist_csv)
    print(f"[snapshots_from_edgelist] {len(graphs_el)} snapshots, "
          f"labels={labels_el}")

    # ------------------------------------------------------------------ #
    # Spec 02 — snapshot stability                                        #
    # ------------------------------------------------------------------ #
    print("\n--- Spec 02: snapshot stability ---")

    graphs, labels = _make_snapshots()

    sim = snapshot_similarity(graphs, graph_labels=labels, report_gaps=False)
    print(f"[snapshot_similarity]  shape={sim.shape}")
    print(sim[["Graph", "jaccard", "edge_persistence",
               "temporal_correlation"]].to_string(index=False))

    tcc = temporal_correlation_coefficient(graphs, graph_labels=labels)
    print(f"[temporal_correlation_coefficient] TCC = {tcc:.4f}")

    # Pair straddling the gap should be NaN
    gap_row = sim[sim["Graph"] == "2024-06"]
    jaccard_val = gap_row["jaccard"].iloc[0]
    assert str(jaccard_val) == "nan", (
        f"Expected NaN for gap pair, got {jaccard_val}")
    print("  ✓ Gap pair correctly reported as NaN")

    # ------------------------------------------------------------------ #
    # Spec 03 — burstiness                                                #
    # ------------------------------------------------------------------ #
    print("\n--- Spec 03: burstiness ---")

    iet = inter_event_times(graphs, graph_labels=labels, by="edge")
    print(f"[inter_event_times]  shape={iet.shape}  "
          f"(gap intervals excluded by default)")
    if not iet.empty:
        print(iet.head(5).to_string(index=False))

    bdf = burstiness_coefficient(
        graphs, graph_labels=labels, by="edge", report_gaps=False)
    print(f"\n[burstiness_coefficient]  shape={bdf.shape}")
    print(bdf.to_string(index=False))
    print()

    # ------------------------------------------------------------------ #
    # Spec 03 with node tracking                                          #
    # ------------------------------------------------------------------ #
    iet_node = inter_event_times(graphs, graph_labels=labels, by="node")
    print(f"[inter_event_times by=node]  shape={iet_node.shape}")

    bdf_node = burstiness_coefficient(
        graphs, graph_labels=labels, by="node", report_gaps=False)
    print(f"[burstiness_coefficient by=node]  shape={bdf_node.shape}")
    print(bdf_node.to_string(index=False))

    # ------------------------------------------------------------------ #
    # Spec 03 — save_path writes PDF                                      #
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as tmp:
        burstiness_coefficient(graphs, graph_labels=labels,
                               save_path=tmp, report_gaps=False)
        files = os.listdir(tmp)
    assert "burstiness_edge.pdf" in files, (
        f"Expected burstiness_edge.pdf, got {files}")
    print("\n  ✓ burstiness_edge.pdf written to save_path")

    # ------------------------------------------------------------------ #
    # Done                                                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("All specs 01, 02, 03 ran without error. ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
