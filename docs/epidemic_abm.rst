Tutorial: agent-based epidemic modelling
========================================

This tutorial analyses contact data produced by an agent-based SIR model,
the way a contact-tracing programme would analyse data from the field.

The population is spatially structured: 90 agents in 30 small households
arranged along a chain, mixing freely at home and visiting only their
immediate neighbours. Transmission is local, so the outbreak has to travel
along the chain rather than jumping across the population.

Three things happen over the 45 simulated days:

1. **An outbreak**, seeded by a single index case on day 1.
2. **A mobility restriction on day 9**, cutting visits between
   households. Household mixing is untouched.
3. **Two tracing waves** — contacts are recorded on days 1-15 and days
   31-45, and the programme is suspended in between.

The asymmetry in point 3 is what makes this example worth running: *the
epidemic does not pause*. Agents keep meeting and keep infecting one
another; only the observation stops. Because the model knows the ground
truth, the tutorial can compare what the observed contact network lets you
prove against what actually happened.

The complete runnable script is ``examples/example_epidemic_abm.py``.

.. contents:: On this page
   :local:
   :depth: 1


The model and its blind spot
----------------------------

The model emits one row per contact, and writes a row only when the
tracing programme is running::

   Observed contacts logged: 2,464
   Epidemic peak: day 6 (2024-03-06) with 14 infectious
   Ever infected (ground truth): 47/90

    day       date  S  I  R  traced
     15 2024-03-15 63  8 19    True
     20 2024-03-20 60  3 27   False
     25 2024-03-25 53  7 30   False
     30 2024-03-30 51  3 36   False
     35 2024-04-04 48  4 38    True

     11 further agents were infected while tracing was suspended.
     None of those contacts appear in the data below.

Everything that follows is computed from the observed contacts only —
exactly the position a real analyst is in.


Daily snapshots and the unobserved stretch
------------------------------------------

.. code-block:: python

   from temporal_networks import snapshots_from_events, detect_temporal_gaps

   graphs, labels = snapshots_from_events(
       contacts, time_col="date", source_col="agent_i",
       target_col="agent_j", freq="D",
   )
   gap_info = detect_temporal_gaps(labels)

``freq="D"`` produces ``YYYY-MM-DD`` labels, from which the gap detector
infers a one-day time step::

   30 daily snapshots (2024-03-01 ... 2024-04-14) from 45 simulated days

   Gaps detected: 1
     between 2024-03-15 and 2024-03-31: 16 days unobserved
   Continuous segments (index ranges): [(0, 15), (15, 30)]


Finding the intervention, not the resumption
--------------------------------------------

The restriction on day 9 cuts mean degree by 38 %. The question is whether
a change-point detector finds *that* rather than the moment the data comes
back.

.. code-block:: python

   from temporal_networks import network_properties, detect_change_points

   props = network_properties(graphs, graph_labels=labels,
                              visualisation=False)
   found = detect_change_points(
       props[["Graph", "Mean Degree"]],
       columns=["Mean Degree"], method="diff", threshold=4.0,
       gap_info=gap_info,          # <- makes the detector gap-aware
   )

::

   Restriction imposed on : 2024-03-09
   Tracing resumed on     : 2024-03-31

   Change points in 'Mean Degree' (diff method, threshold 4.0):
        label    score
   2024-03-09 5.106859

The restriction is found; ``2024-03-31``, the first day back, is not.
Passing ``gap_info`` makes first differences and rolling statistics run
independently inside each continuous segment, so the boundary between the
two waves is never differenced across. Without it, the jump from the last
day of wave 1 to the first day of wave 2 is treated as an ordinary
step between adjacent observations.

.. note::

   The ``"zscore"`` method uses the population standard deviation, so the
   largest ``|z|`` attainable in an *n*-point segment is ``sqrt(n - 1)``:
   ``threshold=3.0`` can only flag points in segments of 11 or more
   snapshots. For short segments, prefer ``method="diff"`` as used here.


What the data can and cannot prove
----------------------------------

This is the central question for contact data with an unobserved period.
:func:`~temporal_networks.temporal_reachability` counts who could have
been reached from a source along time-respecting contacts.

.. code-block:: python

   from temporal_networks import temporal_reachability

   strict = temporal_reachability(graphs, graph_labels=labels,
                                  cross_gaps=False)   # default
   loose  = temporal_reachability(graphs, graph_labels=labels,
                                  cross_gaps=True)

::

   Index case: agent_00   (population 90)
     actually infected in the model       : 46
     reachable, cross_gaps=False (default): 49
     reachable, cross_gaps=True           : 77

Reachability is an *upper bound* on transmission: it counts everyone the
index case could have reached, whether or not the disease actually took
that route.

With ``cross_gaps=True`` that bound is 77 — most of the population —
because paths are allowed to run straight through fifteen days nobody
observed. The default bound of 49 rests only on contacts that were
actually recorded.

.. warning::

   Do not read the closeness of 49 to the true 46 as a guarantee. The
   default is a bound built from observed contacts, not an estimator of
   outbreak size; a different contact structure will place it differently.
   The robust point is the other one: the naive setting inflates the bound
   with paths no one ever saw.

The same distinction shows up in aggregate::

   Temporal efficiency, cross_gaps=False: 0.1258
   Temporal efficiency, cross_gaps=True : 0.1424  (+13%)

Which setting is right depends on what the gap *is*. When the gap is a
genuine closure — a ward shut, a school closed, a border sealed — paths
really cannot cross it, and the default is literally correct. When the gap
is a surveillance blackout, as here, transmission did continue, and the
default is the conservative reading: it reports what the observed data
supports rather than what might have happened unobserved. What it never
does is silently assume the unobserved period was transparent.


Ranking agents for follow-up
----------------------------

.. code-block:: python

   from temporal_networks import temporal_betweenness

   betw = temporal_betweenness(graphs, graph_labels=labels,
                               cross_gaps=False)

::

       node  betweenness
   agent_48     0.059136
   agent_63     0.056106
   agent_10     0.054886
   agent_01     0.053756

These agents most often lie on the quickest time-respecting route between
others — the people a tracing programme would most want to reach first.
With ``cross_gaps=False`` nobody is credited for brokerage on a path that
runs through the unobserved period.


Running it
----------

.. code-block:: bash

   python examples/example_epidemic_abm.py

The script is seeded and self-checking: it asserts that exactly one
unobserved stretch is found, that the mobility restriction is flagged as a
change point, that the day tracing resumed is *not* flagged as either a
change point or an anomaly, and that allowing paths to cross the gap
strictly increases reachability. No files are written unless you pass
``save_path``.

See also the :doc:`social_contacts` tutorial, which covers ingestion,
community tracking, and burstiness on a face-to-face proximity study.
