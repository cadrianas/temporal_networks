# Code Review Checklist — `temporal_networks` (2026-07-06)

Findings from the pyOpenSci/JOSS-style review of the core modules, verified
empirically by the integration toy script (61/61 assertions passed).
Check items off as they are fixed.

## Critical (fix before release)

- [x] **1. `Average Path Length` is mathematically wrong** *(fixed 2026-07-06:
  diagonal excluded via `np.fill_diagonal(..., np.nan)`, edgeless graphs now
  NaN; regression tests added in `tests/unit/test_network_properties.py`)*
  `temporal_networks/network_properties.py:132-134` — `np.nanmean(dist_matrix)`
  includes the n diagonal zeros. Verified: path graph 0–1–2 has true APL 4/3
  (igraph's `average_path_length()` agrees) but the package reports 8/9 ≈ 0.889.
  An edgeless graph reports 0.0 instead of NaN.
  **Fix:** `np.fill_diagonal(dist_matrix, np.nan)` before the mean, or use
  `graph.average_path_length()` directly. Add a regression test.

- [x] **2. `cross_gaps=False` kills ALL reachability after the first gap, not
  just paths across it** *(fixed 2026-07-06: sources now re-seed at each
  segment start — paths confined to one continuous segment are valid, paths
  crossing a gap stay blocked; betweenness counts each pair once at its
  global foremost arrival; docstrings updated; regression tests added in
  `tests/unit/test_temporal_paths.py`)*
  `temporal_networks/temporal_paths.py:135-147` (`_bfs_from_source`) and
  `_foremost_paths_from_source` — at a gap boundary `reachable = set()` clears
  the frontier *including the source*, which is never re-seeded. Verified:
  E cannot reach H even though the E–H edge lies entirely in the post-gap
  segment; nodes appearing only post-gap get closeness/betweenness exactly 0.
  For multi-segment data, `temporal_closeness` / `temporal_efficiency` /
  `temporal_betweenness` are effectively first-segment-only.
  **Fix:** re-seed the source at the start of each segment (paths valid within
  any single segment), or clearly document the current "transmission dies at a
  closure" semantics. Two toy-script assertions pin the current behavior and
  will fail (usefully) when semantics change.

- [x] **3. Silent row-dropping + `print`-based error reporting** *(fixed
  2026-07-07: failing snapshots/pairs now emit NaN rows — shape always
  matches the input — in `snapshot_similarity`, `compute_edge_dynamics`,
  `network_properties`, and `vertex_properties` (missing node); all error
  paths in stability / network_properties / edge_formation_dissolution /
  temporal_paths / burstiness / vertex_properties report via
  `warnings.warn` instead of `print`; tests updated to `assertWarns`.
  Warning prints in communities_measures / calculate_centralities remain —
  covered by items 6 and 17)*
  `temporal_networks/stability.py:203-205`,
  `temporal_networks/edge_formation_dissolution.py:161-163`,
  `temporal_networks/network_properties.py:167-169`, plus analogous blocks in
  `temporal_paths.py` and `burstiness.py` — `except Exception: print(...);
  continue` makes failing rows vanish from output DataFrames, breaking 1:1
  alignment with `graph_labels`.
  **Fix:** emit a NaN row (preserve shape) and report via `warnings.warn` or
  `logging`, not `print`. Narrow the `except Exception` clauses.

- [x] **4. Gap detection silently disables itself on unparseable labels**
  *(fixed 2026-07-07: `detect_temporal_gaps` now emits a `UserWarning`
  naming the offending labels whenever parsing fails — except for the
  auto-generated "Graph N" placeholders, which mean no dates were supplied;
  docstring documents the Warns behavior; 4 unit tests added. Bonus catch:
  the new warning exposed invalid month labels ("2024-13".."2024-17") in
  `flag_anomalous_snapshots`' docstring example and its unit test — both
  fixed to roll over into 2025)*
  `temporal_networks/_gap_utilities.py:336-344` — any unparseable label (e.g.
  "Jan 2024") returns `has_gaps=False` with only a string in the dict; every
  "gap-aware" downstream function silently loses gap-awareness.
  **Fix:** raise a `UserWarning` when parsing fails.

- [x] **5. Edge-dynamics module is not gap-aware despite its docstring**
  *(fixed 2026-07-07: `compute_edge_dynamics` now NaNs gap-straddling pairs,
  matching `snapshot_similarity`; docstring documents the behavior and the
  resulting float dtype for gapped data; regression test added)*
  `temporal_networks/edge_formation_dissolution.py:125` — module claims
  "Properly handles temporal gaps" but `compute_edge_dynamics` compares the
  2024-05 → 2024-07 pair as if consecutive (verified: real numbers, no NaN),
  while `snapshot_similarity` NaNs the identical pair.
  **Fix:** NaN the gap-straddling row, mirroring `stability.py`.

- [x] **6. `communities_measures` always runs spinglass, which fails on any
  disconnected graph** *(fixed 2026-07-07: new `algorithms` parameter with
  validation; default runs all algorithms except spinglass, which is opt-in
  and documented as connected-graphs-only; every requested algorithm is
  guaranteed a key in the result (empty DataFrame + UserWarning on total
  failure, never a missing key); per-graph failures now `warnings.warn`;
  tests rewritten/extended. The dict return type itself is item 12)*
  `temporal_networks/communities_measures.py:112-120` — verified: fails on all
  8 toy snapshots, prints 8 warnings, and the `"spinglass"` key silently
  disappears from the returned dict (later `KeyError` for users).
  **Fix:** skip spinglass by default, run per-component, or make the algorithm
  list a parameter. Also consider the dict return type (see item 12).

- [x] **7. Temporal-path stack scales poorly** *(largely fixed 2026-07-10:
  latency column vectorized via `np.where`, closeness now one vectorized
  groupby pass — 8x at 300 nodes, identical output; the per-source
  re-keying in `_edges_keyed` remains a known, smaller cost)*
  **(original finding)** Temporal-path stack scales poorly (O(V²·T·E) pure Python + pandas
  overhead)**
  `temporal_networks/temporal_paths.py:150` — `_edges_keyed` re-keys every
  snapshot per source; `temporal_paths.py:446` — closeness filters the V²-row
  DataFrame per node (O(V³)); `temporal_paths.py:369` — `.apply(axis=1)`
  instead of `np.where(reach["reachable"], reach["first_arrival_idx"], np.inf)`.
  **Fix:** precompute keyed edge lists once per snapshot; use one
  `groupby("source")` for closeness; vectorize the latency column.

## Style & architecture (recommended)

- [x] **8. `calculate_centralities` defaults `save_path="plots/"`**
  *(fixed 2026-07-10: defaults to None; visualize without save_path warns)*
  `temporal_networks/calculate_centralities.py:37` — only function writing
  into the caller's cwd by default; everywhere else defaults to `None`.

- [x] **9. Typing is nominal** *(fixed 2026-07-11: List[ig.Graph]
  everywhere, GapInfo/GapDict TypedDicts, NodeKey alias; mypy-verified)* — signatures say `graphs: List` with no element
  type; `Dict`/`Set` bare. `py.typed` is shipped and mypy configured, so
  tighten to `list[ig.Graph]`, `dict[str, float]`, etc.

- [x] **10. Duplicated helper `_active_nodes`** *(fixed 2026-07-10:
  consolidated into `_gap_utilities`)*
  `temporal_networks/stability.py:43` and `temporal_networks/burstiness.py:47`
  are identical — move to `_gap_utilities` next to `_vertex_keys`.

- [x] **11. ISO-week round-trip bug (latent)** *(fixed 2026-07-10: parsed
  with `%G-W%V-%u`; year-boundary and 53-week-year regression tests added)*
  `temporal_networks/io.py` builds weekly labels with `isocalendar()` (ISO)
  but `_gap_utilities.py:143` parses with `%W` (non-ISO). Around year
  boundaries (e.g. 2025-W01 containing Dec 29) weekly gap sizes can be off by
  one. **Fix:** parse with `%G-W%V-%u`.

- [x] **12. Return-type inconsistency** *(resolved 2026-07-11: kept the
  dict return but declared it `Dict[str, pd.DataFrame]` and documented
  the every-requested-algorithm-has-a-key guarantee)* — `communities_measures` returns a
  dict of DataFrames while every other analysis function returns a DataFrame.

- [x] **13. Lexicographic label sorting in `inter_event_times`**
  *(fixed 2026-07-10: rows sorted by snapshot index)*
  `temporal_networks/burstiness.py:213` — sorting by `start_label` as string
  breaks chronology for non-zero-padded labels ("2024-W9" > "2024-W10").
  **Fix:** sort by snapshot index.

- [ ] **14. z-score footgun undocumented** — with population std, max |z| in a
  segment of n points is √(n−1), so the default `threshold=3.0` can never flag
  anything in segments shorter than 11 snapshots. Document in
  `detect_change_points`, and consider warning when a segment is too short.

- [x] **15. NaT timestamps silently dropped in ingestion** *(fixed
  2026-07-09: raises ValueError naming the offending rows — stricter than
  the suggested warn, matching the function's other validation)*
  `temporal_networks/io.py:193` — `pd.Grouper` drops NaT rows; events with
  missing timestamps vanish without any signal. Count and warn.

- [x] **16. Unseeded layout randomness in `plot_community_evolution`**
  *(fixed 2026-07-10: `seed` parameter, local RNG, global state untouched)*
  Node positions use the global `random` module when x/y attributes are
  absent — non-reproducible figures. Accept a `seed` parameter.

- [x] **17. Chatty library functions** *(fixed 2026-07-10: progress goes
  to the `temporal_networks` logger, errors to warnings.warn,
  `report_gaps` defaults to False)* — `edge_formation` prints "Computing
  edge formation..." unconditionally; `_detect_communities` prints the
  algorithm name even with `report_gaps=False`. Make progress output opt-in.

## Environment note (not a code defect)

- [ ] **18. `.venv-audit` editable install is broken on Python 3.14** —
  Python 3.14 skips the `__editable__*.pth` file as hidden, so the package
  only imports when cwd is the repo root; the stale finder also maps
  `docs`/`paper`/`specs` as packages (predates the `packages.find` include).
  A freshly built wheel is clean (verified), so PyPI users are unaffected.
  **Fix:** `pip install --force-reinstall .` (non-editable) or
  `pip install -e . --config-settings editable_mode=compat`.

---

### What was verified as correct (for balance)

- Gap logic in `snapshot_similarity`, `burstiness_coefficient`,
  `inter_event_times`, `detect_change_points`: exact hand-computed values
  (jaccard 7/9, burstiness −1/3, z = 2.0 anomaly, gap-pair NaN) all confirmed.
- Node-identity machinery: shuffled vertex order produces byte-identical
  results in `snapshot_similarity` and `compute_edge_dynamics`.
- `io.snapshots_from_events`: vectorized, well-validated, correct
  weightedness decision; the built wheel contains only `temporal_networks/`.
