"""
Social science walkthrough: face-to-face contacts in a school.

Scenario
--------
A SocioPatterns-style proximity study. Wearable sensors record face-to-face
contacts between 18 named actors (two classes of eight students plus two
teachers) over an 18-week academic term. Contacts are logged as individual
events and binned into weekly snapshots.

Two features of the study drive the analysis:

  1. **A two-week holiday break** (ISO weeks 11 and 12) during which no
     sensors were deployed. Those weeks are simply absent from the event
     log, so the package reports them as a temporal gap rather than as
     two weeks in which nobody spoke to anybody.
  2. **A class reshuffle after the break** — two students move from class
     A to class B. This is the substantive question the example answers:
     are the groups observed after the break the *same* groups, or new
     ones?

What this example demonstrates
------------------------------
  - ``snapshots_from_events``   ingest a raw contact log, weight edges by
                                contact frequency
  - ``detect_temporal_gaps``    identify the unobserved holiday weeks
  - ``network_properties``      cohesion of the contact network over time
  - ``snapshot_similarity``     week-to-week churn; NaN across the break
  - ``track_communities``       group lifecycles, with and without
                                bridging the unobserved period
  - ``burstiness_coefficient``  sustained friendships vs project-driven
                                bursts of contact
  - ``vertex_properties``       a single teacher's brokerage trajectory

Run:  .venv-audit/bin/python examples/example_social_contacts.py

No files are written unless you pass ``save_path`` to the analysis
functions.
"""

import ast
import random
from datetime import date, timedelta

import igraph as ig
import pandas as pd

from temporal_networks import (
    snapshots_from_events,
    detect_temporal_gaps,
    network_properties,
    snapshot_similarity,
    track_communities,
    burstiness_coefficient,
    vertex_properties,
)

# ---------------------------------------------------------------------------
# Study design
# ---------------------------------------------------------------------------

CLASS_A = [f"student_A{i}" for i in range(1, 9)]
CLASS_B = [f"student_B{i}" for i in range(1, 9)]
TEACHERS = ["teacher_Ines", "teacher_Omar"]

# Weeks of the term, as Mondays. Weeks 11-12 of the term are the holiday
# break: no sensors deployed, so no events are emitted for them at all.
TERM_START = date(2024, 1, 8)
N_WEEKS = 18
HOLIDAY_WEEKS = {11, 12}

# After the break, these two students join class B.
SWITCHERS = ["student_A7", "student_A8"]

# Burstiness is computed from the *intervals between the weeks in which a
# tie is active*, so the three pairs below are designed to sit at different
# points of that scale.

# Irregular: two contact episodes separated by a long silence, both inside
# the first continuous stretch of the term.
PROJECT_PAIR = ("student_A1", "student_B1")
PROJECT_WEEKS = {1, 2, 3, 9, 10}

# Perfectly regular: active every observed week, so every interval is one
# week and B collapses to -1.
STEADY_PAIR = ("student_A2", "student_A3")

# Active in a block before the holiday and a block after it. The long
# interval between the two blocks straddles unobserved time, which is what
# ``exclude_gaps`` decides whether to count.
STRADDLE_PAIR = ("student_A4", "student_B4")
STRADDLE_WEEKS = {8, 9, 10, 13, 14, 15}


def _class_of(actor: str, week: int) -> str:
    """Return the class an actor belongs to in a given term week."""
    if actor in TEACHERS:
        return "staff"
    if actor in SWITCHERS and week > max(HOLIDAY_WEEKS):
        return "B"
    return "A" if actor in CLASS_A else "B"


def build_contact_log(seed: int = 7) -> pd.DataFrame:
    """
    Generate an event-level face-to-face contact log.

    Returns a long-form DataFrame with one row per recorded contact
    (``date``, ``actor_i``, ``actor_j``, ``duration_min``). Contacts are
    dense within a class, sparse between classes, and teachers bridge the
    two. Holiday weeks emit no rows.
    """
    rng = random.Random(seed)
    actors = CLASS_A + CLASS_B + TEACHERS
    rows = []

    for week in range(1, N_WEEKS + 1):
        if week in HOLIDAY_WEEKS:
            continue  # sensors not deployed -> week absent from the log

        monday = TERM_START + timedelta(weeks=week - 1)

        for i, a in enumerate(actors):
            for b in actors[i + 1:]:
                cls_a, cls_b = _class_of(a, week), _class_of(b, week)

                # Contact probability depends on shared context.
                if "staff" in (cls_a, cls_b):
                    p = 0.55 if cls_a != cls_b else 0.9   # teacher-student
                elif cls_a == cls_b:
                    p = 0.7                                # same class
                else:
                    p = 0.06                               # across classes

                pair = tuple(sorted((a, b)))
                if pair == tuple(sorted(STEADY_PAIR)):
                    p = 1.0
                elif pair == tuple(sorted(PROJECT_PAIR)):
                    p = 1.0 if week in PROJECT_WEEKS else 0.0
                elif pair == tuple(sorted(STRADDLE_PAIR)):
                    p = 1.0 if week in STRADDLE_WEEKS else 0.0

                if rng.random() >= p:
                    continue

                # One to four separate encounters during the school week.
                for _ in range(rng.randint(1, 4)):
                    day = monday + timedelta(days=rng.randint(0, 4))
                    rows.append({
                        "date": day.isoformat(),
                        "actor_i": a,
                        "actor_j": b,
                        "duration_min": rng.choice([5, 10, 15, 20, 30]),
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Community detection is randomised, and seeding `random` alone does
    # NOT affect igraph's generators. Without this line the community
    # lineages below differ from run to run.
    ig.set_random_number_generator(random.Random(42))

    print("=" * 72)
    print("Face-to-face contacts in a school: 18 actors, 18 term weeks,")
    print("two-week holiday break (unobserved), class reshuffle after it")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # 1. Ingest the raw contact log                                       #
    # ------------------------------------------------------------------ #
    print("\n--- 1. Ingestion: contact events -> weekly snapshots ---")

    contacts = build_contact_log()
    print(f"Raw contact log: {len(contacts):,} recorded encounters "
          f"between {contacts['actor_i'].nunique() + 1} actors")
    print(contacts.head(4).to_string(index=False))

    # Weekly bins. Edge weight = total contact minutes that week, which is
    # the standard proximity-study intensity measure.
    graphs, labels = snapshots_from_events(
        contacts,
        time_col="date",
        source_col="actor_i",
        target_col="actor_j",
        freq="W",
        weight_col="duration_min",
    )
    print(f"\n{len(graphs)} weekly snapshots: {labels[0]} ... {labels[-1]}")
    print(f"Snapshot 1: {graphs[0].vcount()} actors, "
          f"{graphs[0].ecount()} contact pairs")

    # ------------------------------------------------------------------ #
    # 2. The holiday break is detected, not imputed                       #
    # ------------------------------------------------------------------ #
    print("\n--- 2. Gap detection: the unobserved holiday ---")

    gap_info = detect_temporal_gaps(labels)
    print(f"Gaps found: {gap_info['num_gaps']}")
    for gap in gap_info["gaps"]:
        print(f"  between {gap['start_label']} and {gap['end_label']}: "
              f"{gap['gap_size']:.0f} weeks with no observation")
    print(f"Continuous segments (index ranges): {gap_info['segments']}")

    assert gap_info["num_gaps"] == 1, "expected exactly one holiday gap"

    # The first snapshot after the break: start of the second segment.
    post_break = labels[gap_info["segments"][1][0]]
    print(f"First week back: {post_break}")

    # ------------------------------------------------------------------ #
    # 3. Network cohesion over the term                                   #
    # ------------------------------------------------------------------ #
    print("\n--- 3. Structural properties per week ---")

    props = network_properties(graphs, graph_labels=labels,
                               visualisation=False)
    print(props[["Graph", "Density", "Mean Degree",
                 "Average Path Length", "Transitivity"]].to_string(
        index=False))
    print("\n  Transitivity stays high throughout: contacts close into "
          "triangles,\n  as classroom proximity data normally does.")

    # ------------------------------------------------------------------ #
    # 4. Week-to-week churn, with the break left uncompared               #
    # ------------------------------------------------------------------ #
    print("\n--- 4. Snapshot similarity (the break is NOT compared) ---")

    sim = snapshot_similarity(graphs, graph_labels=labels)
    print(sim[["Graph", "jaccard", "edge_persistence",
               "node_persistence"]].to_string(index=False))

    # The first post-holiday row straddles the gap and must be NaN.
    straddling = sim.loc[sim["Graph"] == post_break, "jaccard"]
    assert straddling.isna().all(), (
        "the pair straddling the holiday should be NaN, not a computed "
        "similarity across ten unobserved days")
    print(f"\n  {post_break} is NaN: it is the first week back, and the "
          f"package\n  refuses to call it 'similar' to a week three weeks "
          f"earlier.")

    # ------------------------------------------------------------------ #
    # 5. Do the same groups come back after the break?                    #
    # ------------------------------------------------------------------ #
    print("\n--- 5. Community tracking across an unobserved period ---")

    strict = track_communities(graphs, graph_labels=labels,
                               bridge_gaps=False)
    bridged = track_communities(graphs, graph_labels=labels,
                               bridge_gaps=True)

    print(f"Distinct lineages, bridge_gaps=False: "
          f"{strict['lineage_id'].nunique()}")
    print(f"Distinct lineages, bridge_gaps=True:  "
          f"{bridged['lineage_id'].nunique()}")

    births_after = strict.loc[
        (strict["Graph"] == post_break) & (strict["event"] == "birth")]
    print(f"\nWith bridge_gaps=False, the first week back shows "
          f"{len(births_after)} 'birth' event(s):\n"
          f"unobserved time is treated as a break in continuity, so the "
          f"groups are\nnew lineages rather than silent continuations.")

    print("\nGroups in the final week, and who is in them:")
    final = strict.loc[strict["Graph"] == labels[-1]]
    for _, row in final.iterrows():
        members = row["members"]
        if isinstance(members, str):        # stored as a repr in CSV round-trips
            members = ast.literal_eval(members)
        print(f"  lineage {row['lineage_id']} (size {row['size']}, "
              f"{row['event']}):")
        print(f"    {', '.join(sorted(members))}")

    # The substantive check: the two students who changed class are now
    # grouped with class B, recovered from contact data alone.
    b_group = max(
        (ast.literal_eval(m) if isinstance(m, str) else m
         for m in final["members"]),
        key=lambda m: sum(1 for x in m if x in CLASS_B))
    for switcher in SWITCHERS:
        assert switcher in b_group, (
            f"{switcher} moved to class B and should be detected in the "
            f"class B group")
    print(f"\n  {' and '.join(SWITCHERS)} changed class over the holiday, "
          f"and the\n  tracker places them in the class B group "
          f"afterwards — recovered from\n  contact patterns alone, with no "
          f"class roster supplied.")

    # ------------------------------------------------------------------ #
    # 6. Sustained friendship vs project-driven burst                     #
    # ------------------------------------------------------------------ #
    print("\n--- 6. Burstiness of contact patterns ---")

    burst = burstiness_coefficient(graphs, graph_labels=labels, by="edge")
    burst = burst.dropna(subset=["burstiness"])

    def _b(pair):
        """
        Burstiness of one specific unordered pair of actors.

        ``entity`` holds the string repr of the endpoint tuple, e.g.
        ``"('student_A1', 'student_B1')"``, so it is parsed back before
        matching.
        """
        want = frozenset(pair)
        mask = burst["entity"].apply(
            lambda e: frozenset(ast.literal_eval(e)) == want)
        matched = burst.loc[mask, "burstiness"]
        assert len(matched) == 1, f"expected one row for {pair}"
        return float(matched.iloc[0])

    print("B ranges from -1 (perfectly regular) through 0 (Poisson-like) "
          "to +1 (bursty).\nIt is computed from the intervals between the "
          "weeks in which a tie is active.")

    b_project, b_steady = _b(PROJECT_PAIR), _b(STEADY_PAIR)
    print(f"\n  intermittent pair {PROJECT_PAIR}")
    print(f"    active in term weeks {sorted(PROJECT_WEEKS)}"
          f"  -> B = {b_project:+.3f}")
    print(f"  steady pair       {STEADY_PAIR}")
    print(f"    active every observed week"
          f"{'':17}-> B = {b_steady:+.3f}")

    assert b_project > b_steady, (
        "the intermittent tie should be less regular than the tie that "
        "is active every single week")

    print("\n  Two ties with a similar number of contacts can mean very "
          "different\n  things. The steady pair is metronomic (B = -1); "
          "the intermittent pair,\n  with one long silence between two "
          "episodes, sits near the Poisson\n  point. Neither judgement "
          "needs to know what the tie means.")

    # ------------------------------------------------------------------ #
    # 6b. What the holiday does to an inter-event interval               #
    # ------------------------------------------------------------------ #
    print("\n--- 6b. Why exclude_gaps matters for burstiness ---")

    lax = burstiness_coefficient(graphs, graph_labels=labels, by="edge",
                                 exclude_gaps=False).dropna(
        subset=["burstiness"])

    def _b_lax(pair):
        want = frozenset(pair)
        mask = lax["entity"].apply(
            lambda e: frozenset(ast.literal_eval(e)) == want)
        return float(lax.loc[mask, "burstiness"].iloc[0])

    strict_b, lax_b = _b(STRADDLE_PAIR), _b_lax(STRADDLE_PAIR)
    print(f"  {STRADDLE_PAIR} is active in term weeks "
          f"{sorted(STRADDLE_WEEKS)},\n  i.e. a block before the holiday "
          f"and a block after it.\n")
    print(f"    exclude_gaps=True  (default) -> B = {strict_b:+.3f}")
    print(f"    exclude_gaps=False           -> B = {lax_b:+.3f}")
    print("\n  With the default, the interval that spans the unobserved "
          "weeks is\n  dropped: we never saw whether the pair met during "
          "the holiday, so that\n  silence is not evidence of anything. "
          "With exclude_gaps=False the same\n  tie looks more irregular — "
          "an artefact of when the sensors were on,\n  not of how these "
          "two students behaved.")

    # ------------------------------------------------------------------ #
    # 7. One actor's trajectory                                           #
    # ------------------------------------------------------------------ #
    print("\n--- 7. A single teacher's brokerage over the term ---")

    vp = vertex_properties(graphs, node_name="teacher_Ines",
                           graph_labels=labels, visualisation=False)
    print(vp[["Graph", "Degree_Centrality", "Betweenness_Centrality",
              "Constraint"]].to_string(index=False))

    # Burt's Constraint is low for an actor whose contacts are themselves
    # unconnected: the structural-hole broker. Compare the bridging
    # teacher with a student embedded in a single class.
    print("\nTerm-average comparison:")
    scores = {}
    for who in ["teacher_Ines", "student_A2"]:
        row = vertex_properties(graphs, node_name=who,
                                graph_labels=labels, visualisation=False)
        scores[who] = (row["Constraint"].mean(),
                       row["Betweenness_Centrality"].mean())
        print(f"  {who:14s} constraint={scores[who][0]:.3f}  "
              f"betweenness={scores[who][1]:6.2f}")

    assert scores["teacher_Ines"][0] < scores["student_A2"][0], (
        "the teacher bridges two classes and should be less constrained "
        "than a student inside one")
    print("\n  The teacher is the less constrained of the two and carries "
          "roughly three\n  times the betweenness: the expected signature "
          "of an actor who spans two\n  otherwise separate groups.")

    print("\n" + "=" * 72)
    print("Done. Every result above is gap-aware: nothing is interpolated")
    print("across the holiday, and no group is assumed to have survived it.")
    print("=" * 72)


if __name__ == "__main__":
    main()
