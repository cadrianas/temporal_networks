"""
Agent-based epidemic walkthrough: contact tracing in two observation waves.

Scenario
--------
An agent-based SIR model runs on a spatially structured population: 90
agents in 30 small households arranged along a chain, where people mix
freely at home and visit only their immediate neighbours. Transmission is
therefore local, and the outbreak has to travel along the chain rather
than jumping across the population.

Three things happen over the 45 simulated days:

  1. **An outbreak**, seeded by a single index case on day 1.
  2. **A mobility restriction on day 9** that cuts visits between
     households. Household mixing is untouched.
  3. **Two tracing waves**: contacts are recorded on days 1-15 and days
     31-45, and the programme is suspended for the fifteen days in
     between. Note the asymmetry that makes this example worth running:
     *the epidemic does not pause*. Agents keep meeting and keep
     infecting one another; only the observation stops.

Because the model knows the ground truth, this example can do something a
real dataset cannot: compare what the *observed* contact network lets you
prove against what actually happened.

What this example demonstrates
------------------------------
  - ``snapshots_from_events``    daily contact networks from an event log
  - ``detect_temporal_gaps``     locate the unobserved stretch
  - ``network_properties``       the restriction's effect on mixing
  - ``detect_change_points``     find the restriction without mistaking
                                 the resumption of tracing for one
  - ``temporal_reachability``    who could have been infected, with and
                                 without assuming the unobserved period
                                 away — checked against the model's true
                                 infection set
  - ``temporal_efficiency``      how fast the contact network spreads
  - ``temporal_betweenness``     which agents broker transmission
  - ``flag_anomalous_snapshots`` unusual days

Run:  .venv-audit/bin/python examples/example_epidemic_abm.py

No files are written unless you pass ``save_path`` to the analysis
functions.
"""

import random
from datetime import date, timedelta

import pandas as pd

from temporal_networks import (
    snapshots_from_events,
    detect_temporal_gaps,
    network_properties,
    detect_change_points,
    flag_anomalous_snapshots,
    temporal_reachability,
    temporal_betweenness,
    temporal_efficiency,
)

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

N_HOUSEHOLDS = 30
HOUSEHOLD_SIZE = 3
N_AGENTS = N_HOUSEHOLDS * HOUSEHOLD_SIZE

N_DAYS = 45
DAY_ZERO = date(2024, 3, 1)
INDEX_CASE = "agent_00"

P_HOUSEHOLD = 0.60          # per pair per day, within a household
N_VISITS = 2                # visits per household per day, to the next one

RESTRICTION_DAY = 9
RESTRICTION_FACTOR = 0.30   # applied to between-household visits only

# Days on which contact tracing was running. Contacts outside these
# windows still occur and still transmit -- they are simply not observed.
WAVE_1 = range(1, 16)
WAVE_2 = range(31, N_DAYS + 1)
OBSERVED_DAYS = set(WAVE_1) | set(WAVE_2)

BETA = 0.45                 # per-contact transmission probability
INFECTIOUS_DAYS = 6

# Analysis settings established for this dataset
CHANGE_POINT_COLUMN = "Mean Degree"
CHANGE_POINT_THRESHOLD = 4.0


def _label(day: int) -> str:
    """Calendar label for a simulation day (1-indexed), e.g. '2024-03-01'."""
    return (DAY_ZERO + timedelta(days=day - 1)).isoformat()


def _day_of(label: str) -> int:
    """Inverse of :func:`_label`."""
    return (date.fromisoformat(label) - DAY_ZERO).days + 1


# ---------------------------------------------------------------------------
# The agent-based model
# ---------------------------------------------------------------------------

def run_abm(seed: int = 11):
    """
    Run an SIR agent-based model on a chain of households.

    Each household occupies one position on a ring; every day a household
    makes ``N_VISITS`` contacts with the next household along, damped
    after the mobility restriction. Households themselves keep mixing
    throughout.

    Returns
    -------
    contact_log : pandas.DataFrame
        One row per *observed* contact (``date``, ``agent_i``,
        ``agent_j``). Contacts outside the two tracing waves are omitted.
    epi_curve : pandas.DataFrame
        Daily S/I/R counts for every simulated day, observed or not --
        this is ground truth, not observation.
    ever_infected : set of str
        Every agent infected at any point, from the model itself.
    """
    rng = random.Random(seed)
    agents = [f"agent_{i:02d}" for i in range(N_AGENTS)]
    household_of = {a: i // HOUSEHOLD_SIZE for i, a in enumerate(agents)}
    members = {h: [a for a in agents if household_of[a] == h]
               for h in range(N_HOUSEHOLDS)}

    state = {a: "S" for a in agents}
    state[INDEX_CASE] = "I"
    infected_on = {INDEX_CASE: 1}
    ever_infected = {INDEX_CASE}

    contact_rows, curve_rows = [], []

    for day in range(1, N_DAYS + 1):
        damping = RESTRICTION_FACTOR if day >= RESTRICTION_DAY else 1.0

        # --- today's contacts -------------------------------------------
        todays_contacts = []
        for h in range(N_HOUSEHOLDS):
            group = members[h]
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    if rng.random() < P_HOUSEHOLD:
                        todays_contacts.append((a, b))

        for h in range(N_HOUSEHOLDS):
            for _ in range(N_VISITS):
                if rng.random() < damping:
                    a = rng.choice(members[h])
                    b = rng.choice(members[(h + 1) % N_HOUSEHOLDS])
                    todays_contacts.append(tuple(sorted((a, b))))

        # --- transmission over those contacts ---------------------------
        newly_infected = []
        for a, b in todays_contacts:
            sa, sb = state[a], state[b]
            if sa == "I" and sb == "S" and rng.random() < BETA:
                newly_infected.append(b)
            elif sb == "I" and sa == "S" and rng.random() < BETA:
                newly_infected.append(a)

        # --- record only what the tracing programme saw -----------------
        if day in OBSERVED_DAYS:
            for a, b in todays_contacts:
                contact_rows.append({"date": _label(day),
                                     "agent_i": a, "agent_j": b})

        # --- update states ----------------------------------------------
        for a in newly_infected:
            if state[a] == "S":
                state[a] = "I"
                infected_on[a] = day
                ever_infected.add(a)

        for a, first_day in infected_on.items():
            if state[a] == "I" and day - first_day >= INFECTIOUS_DAYS:
                state[a] = "R"

        counts = {"day": day, "date": _label(day),
                  "traced": day in OBSERVED_DAYS}
        for s in ("S", "I", "R"):
            counts[s] = sum(1 for v in state.values() if v == s)
        curve_rows.append(counts)

    return (pd.DataFrame(contact_rows),
            pd.DataFrame(curve_rows),
            ever_infected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Agent-based SIR on a chain of households")
    print(f"{N_AGENTS} agents in {N_HOUSEHOLDS} households, {N_DAYS} days")
    print(f"Mobility restriction on day {RESTRICTION_DAY}; tracing runs "
          f"days {WAVE_1.start}-{WAVE_1.stop - 1} and "
          f"{WAVE_2.start}-{WAVE_2.stop - 1}")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # 1. Run the model                                                    #
    # ------------------------------------------------------------------ #
    print("\n--- 1. The agent-based model ---")

    contacts, curve, ever_infected = run_abm()
    peak = curve.loc[curve["I"].idxmax()]
    print(f"Observed contacts logged: {len(contacts):,}")
    print(f"Epidemic peak: day {int(peak['day'])} ({peak['date']}) with "
          f"{int(peak['I'])} infectious")
    print(f"Ever infected (ground truth): {len(ever_infected)}/{N_AGENTS}")

    # Susceptibles lost while nobody was looking: S + I + R is constant,
    # so the drop in S is exactly the number newly infected.
    untraced = curve.loc[~curve["traced"]]
    infected_during_gap = int(untraced["S"].iloc[0] - untraced["S"].iloc[-1])
    print("\nGround-truth curve (every fifth day; 'traced' marks the "
          "observation waves):")
    print(curve[curve["day"] % 5 == 0][
        ["day", "date", "S", "I", "R", "traced"]].to_string(index=False))
    print(f"\n  {infected_during_gap} further agents were infected while "
          f"tracing was suspended.\n  None of those contacts appear in "
          f"the data below.")

    # ------------------------------------------------------------------ #
    # 2. Build daily contact networks                                     #
    # ------------------------------------------------------------------ #
    print("\n--- 2. Contact log -> daily snapshots ---")

    graphs, labels = snapshots_from_events(
        contacts, time_col="date", source_col="agent_i",
        target_col="agent_j", freq="D",
    )
    print(f"{len(graphs)} daily snapshots ({labels[0]} ... {labels[-1]}) "
          f"from {N_DAYS} simulated days")

    gap_info = detect_temporal_gaps(labels)
    print(f"\nGaps detected: {gap_info['num_gaps']}")
    for gap in gap_info["gaps"]:
        print(f"  between {gap['start_label']} and {gap['end_label']}: "
              f"{gap['gap_size']:.0f} days unobserved")
    print(f"Continuous segments (index ranges): {gap_info['segments']}")
    assert gap_info["num_gaps"] == 1, "expected exactly one unobserved run"

    resumption = labels[gap_info["segments"][1][0]]

    # ------------------------------------------------------------------ #
    # 3. Did the restriction change the contact network?                  #
    # ------------------------------------------------------------------ #
    print("\n--- 3. Mixing before and after the restriction ---")

    props = network_properties(graphs, graph_labels=labels,
                               visualisation=False)
    props["day"] = props["Graph"].apply(_day_of)

    pre = props.loc[props["day"] < RESTRICTION_DAY, "Mean Degree"].mean()
    post = props.loc[props["day"] >= RESTRICTION_DAY, "Mean Degree"].mean()
    print(f"Mean degree before day {RESTRICTION_DAY}: {pre:.2f}")
    print(f"Mean degree from day {RESTRICTION_DAY}:   {post:.2f}  "
          f"({100 * (1 - post / pre):.0f}% drop)")

    # ------------------------------------------------------------------ #
    # 4. Change points: find the restriction, not the resumption          #
    # ------------------------------------------------------------------ #
    print("\n--- 4. Change-point detection ---")

    series = props[["Graph", CHANGE_POINT_COLUMN]]
    found = detect_change_points(
        series, columns=[CHANGE_POINT_COLUMN], method="diff",
        threshold=CHANGE_POINT_THRESHOLD, gap_info=gap_info)

    print(f"Restriction imposed on : {_label(RESTRICTION_DAY)}")
    print(f"Tracing resumed on     : {resumption}")
    print(f"\nChange points in '{CHANGE_POINT_COLUMN}' "
          f"(diff method, threshold {CHANGE_POINT_THRESHOLD}):")
    print(found[["label", "score"]].to_string(index=False)
          if len(found) else "  (none)")

    flagged = set(found["label"])
    assert _label(RESTRICTION_DAY) in flagged, (
        "the mobility restriction is a real change in the contact network "
        "and should be detected")
    assert resumption not in flagged, (
        "the day tracing resumed is a jump in the data, not a change in "
        "the contact network, and must not be flagged")

    print(f"\n  The restriction on {_label(RESTRICTION_DAY)} is found; "
          f"{resumption}, the day the\n  programme came back online, is "
          f"not. First differences are taken within\n  each continuous "
          f"segment, so the boundary between the two waves is never\n"
          f"  differenced across.")

    # ------------------------------------------------------------------ #
    # 5. Who could have been infected?                                    #
    # ------------------------------------------------------------------ #
    print("\n--- 5. Temporal reachability from the index case ---")

    strict = temporal_reachability(graphs, graph_labels=labels,
                                   cross_gaps=False)
    loose = temporal_reachability(graphs, graph_labels=labels,
                                  cross_gaps=True)

    def _reached(df):
        hit = df[(df["source"] == INDEX_CASE) & df["reachable"] &
                 (df["target"] != INDEX_CASE)]
        return set(hit["target"])

    reach_strict, reach_loose = _reached(strict), _reached(loose)
    truth = ever_infected - {INDEX_CASE}

    print(f"Index case: {INDEX_CASE}   (population {N_AGENTS})")
    print(f"  actually infected in the model       : {len(truth):2d}")
    print(f"  reachable, cross_gaps=False (default): {len(reach_strict):2d}")
    print(f"  reachable, cross_gaps=True           : {len(reach_loose):2d}")

    assert len(reach_loose) > len(reach_strict), (
        "treating the unobserved period as contiguous should let paths "
        "through that the observed data cannot support")

    print("\n  Reachability is an upper bound on transmission: it counts "
          "everyone the\n  index case could have reached along "
          "time-respecting contacts, whether or\n  not the disease "
          "actually took that route.")
    print(f"\n  With cross_gaps=True the bound is {len(reach_loose)} — "
          f"most of the population —\n  because paths are allowed to run "
          f"straight through fifteen days nobody\n  observed. The default "
          f"bound of {len(reach_strict)} rests only on contacts that were "
          f"actually\n  recorded. In this run the true outbreak reached "
          f"{len(truth)}.")
    print("\n  Do not read the closeness of those two numbers as a "
          "guarantee: the\n  default is a bound built from observed "
          "contacts, not an estimator of\n  outbreak size. The point is "
          "that the naive setting inflates the bound\n  with paths no one "
          "ever saw.")

    eff_strict = temporal_efficiency(graphs, graph_labels=labels,
                                     cross_gaps=False)
    eff_loose = temporal_efficiency(graphs, graph_labels=labels,
                                    cross_gaps=True)
    print(f"\nTemporal efficiency, cross_gaps=False: {eff_strict:.4f}")
    print(f"Temporal efficiency, cross_gaps=True : {eff_loose:.4f}  "
          f"(+{100 * (eff_loose / eff_strict - 1):.0f}%)")

    # ------------------------------------------------------------------ #
    # 6. Who brokers transmission?                                        #
    # ------------------------------------------------------------------ #
    print("\n--- 6. Temporal betweenness: brokers of transmission ---")

    betw = temporal_betweenness(graphs, graph_labels=labels,
                                cross_gaps=False)
    print("Top 8 agents by temporal betweenness:")
    print(betw.head(8).to_string(index=False))
    print("\n  These agents most often lie on the quickest time-respecting "
          "route between\n  others: the people a tracing programme would "
          "most want to reach first,\n  ranked without crediting anyone "
          "for paths through the unobserved period.")

    # ------------------------------------------------------------------ #
    # 7. Unusual days                                                     #
    # ------------------------------------------------------------------ #
    print("\n--- 7. Anomalous snapshots ---")

    flags = flag_anomalous_snapshots(graphs, graph_labels=labels,
                                     method="zscore", threshold=2.5)
    if len(flags):
        print(flags[["column", "label", "score"]].to_string(index=False))
        assert resumption not in set(flags["label"]), (
            "the first day back should not read as an anomalous day")
    else:
        print("  No snapshot exceeds the z-score threshold within its "
              "segment.")

    print("\n" + "=" * 72)
    print("Done. The unobserved period never becomes a finding: not a")
    print("change point, not an anomalous day, and not a contact path.")
    print("=" * 72)


if __name__ == "__main__":
    main()
