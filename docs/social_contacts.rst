Tutorial: face-to-face contacts in a school
===========================================

This tutorial analyses a SocioPatterns-style proximity study with
``temporal_networks``. Wearable sensors record face-to-face contacts
between 18 named actors — two classes of eight students plus two teachers
— over an 18-week academic term.

Two features of the study shape the analysis:

1. **A two-week holiday break** during which no sensors were deployed.
   Those weeks are absent from the event log, and the package reports
   them as a temporal gap rather than as two weeks in which nobody spoke
   to anybody.
2. **A class reshuffle after the break** — two students move from class
   A to class B. The substantive question is whether the groups observed
   after the break are the *same* groups, or new ones.

The complete runnable script is
``examples/example_social_contacts.py``.

.. contents:: On this page
   :local:
   :depth: 1


From a contact log to weekly snapshots
--------------------------------------

Proximity studies produce event-level data: one row per encounter. The
ingestion helper bins those events into snapshots, keyed by actor name so
the same person can be followed across weeks.

.. code-block:: python

   from temporal_networks import snapshots_from_events

   # contacts: columns date, actor_i, actor_j, duration_min
   graphs, labels = snapshots_from_events(
       contacts,
       time_col="date",
       source_col="actor_i",
       target_col="actor_j",
       freq="W",
       weight_col="duration_min",
   )

Passing ``weight_col`` makes each edge weight the total contact minutes
for that week, which is the usual intensity measure in proximity studies.
``freq="W"`` produces ``YYYY-Wnn`` labels, which in turn tells the gap
detector that one week is the natural time step::

   Raw contact log: 2,500 recorded encounters between 18 actors
   16 weekly snapshots: 2024-W02 ... 2024-W19

Eighteen term weeks produce sixteen snapshots. The two holiday weeks
generated no events, so they are simply not there.


The holiday is detected, not imputed
------------------------------------

.. code-block:: python

   from temporal_networks import detect_temporal_gaps

   gap_info = detect_temporal_gaps(labels)

::

   Gaps found: 1
     between 2024-W11 and 2024-W14: 3 weeks with no observation
   Continuous segments (index ranges): [(0, 10), (10, 16)]

:func:`~temporal_networks.detect_temporal_gaps` returns a dictionary whose
``segments`` entry gives the index ranges of each continuous stretch —
here weeks 1-10 and weeks 11-16 of the snapshot sequence. Every
gap-aware function in the package works from this segmentation, and you
rarely need to call it yourself.


Week-to-week churn, with the break left uncompared
--------------------------------------------------

:func:`~temporal_networks.snapshot_similarity` compares each snapshot with
its predecessor. Pairs that straddle the gap are reported as ``NaN``:

.. code-block:: python

   from temporal_networks import snapshot_similarity

   sim = snapshot_similarity(graphs, graph_labels=labels)

::

      Graph  jaccard  edge_persistence  node_persistence
   2024-W10 0.413043          0.633333               1.0
   2024-W11 0.439560          0.571429               1.0
   2024-W14      NaN               NaN               NaN
   2024-W15 0.423529          0.610169               1.0

Around 60 % of ties survive from one week to the next during term.
The row for ``2024-W14`` — the first week back — is ``NaN`` because its
predecessor in the data is three weeks earlier. Reporting a number there
would describe the holiday, not the students.


Do the same groups come back?
-----------------------------

:func:`~temporal_networks.track_communities` detects communities per
snapshot and matches them across time by membership overlap, assigning
each a persistent ``lineage_id`` and a lifecycle ``event``.

.. code-block:: python

   from temporal_networks import track_communities

   strict  = track_communities(graphs, graph_labels=labels,
                               bridge_gaps=False)   # default
   bridged = track_communities(graphs, graph_labels=labels,
                               bridge_gaps=True)

::

   Distinct lineages, bridge_gaps=False: 8
   Distinct lineages, bridge_gaps=True:  6

With the default ``bridge_gaps=False``, the groups seen after the holiday
start fresh lineages and are labelled ``birth`` rather than ``continue``.
This is the conservative reading: three weeks passed unobserved, and
continuity of membership is an inference, not an observation. Passing
``bridge_gaps=True`` links lineages across the gap and yields six
lineages instead of eight.

The reshuffle itself is recovered from contact patterns alone::

     lineage 4 (size 6, continue):
       student_A1, ... , student_A6
     lineage 5 (size 8, merge):
       student_A7, student_A8, student_B4, ... , student_B8, teacher_Omar
     lineage 7 (size 4, birth):
       student_B1, student_B2, student_B3, teacher_Ines

``student_A7`` and ``student_A8`` changed class over the holiday, and
appear in a class B group afterwards. No class roster was supplied.

Note that the multilevel algorithm splits class B into two communities in
the final week rather than recovering the roster exactly. Community
detection answers "who is densely connected to whom", which need not
coincide with an administrative grouping — the script asserts only that
the switchers are grouped with class B members, not that the partition
reproduces the roster.


Sustained ties versus intermittent ones
---------------------------------------

The Goh–Barabási burstiness coefficient *B* summarises the intervals
between the weeks in which a tie is active: ``-1`` is perfectly regular,
``0`` is Poisson-like, ``+1`` is bursty.

.. code-block:: python

   from temporal_networks import burstiness_coefficient

   burst = burstiness_coefficient(graphs, graph_labels=labels, by="edge")

::

     intermittent pair ('student_A1', 'student_B1')
       active in term weeks [1, 2, 3, 9, 10]  -> B = -0.019
     steady pair       ('student_A2', 'student_A3')
       active every observed week             -> B = -1.000

Two ties with a similar number of contacts can mean quite different
things. The steady pair is metronomic; the intermittent pair, with one
long silence between two episodes, sits near the Poisson point.

.. note::

   ``entity`` holds the *string* repr of the endpoint tuple, e.g.
   ``"('student_A1', 'student_B1')"``. Parse it with
   :func:`ast.literal_eval` before matching against your own node names.


Why ``exclude_gaps`` matters here
---------------------------------

Consider a pair active in a block of weeks before the holiday and another
block after it. The interval between those blocks spans unobserved time.

.. code-block:: python

   strict = burstiness_coefficient(graphs, graph_labels=labels,
                                   by="edge")                     # default
   lax    = burstiness_coefficient(graphs, graph_labels=labels,
                                   by="edge", exclude_gaps=False)

::

     ('student_A4', 'student_B4'), active in term weeks [8, 9, 10, 13, 14, 15]

       exclude_gaps=True  (default) -> B = -1.000
       exclude_gaps=False           -> B = -0.273

Under the default, the interval spanning the holiday is dropped: nobody
observed whether these two met during the break, so that silence is not
evidence of anything, and what remains are two evenly spaced blocks.
With ``exclude_gaps=False`` the same tie looks markedly more irregular —
an artefact of when the sensors were switched on, not of how these two
students behaved.


Brokerage
---------

:func:`~temporal_networks.vertex_properties` follows one actor across the
term. Burt's *Constraint* is low for an actor whose contacts are
themselves unconnected — the structural-hole broker.

.. code-block:: python

   from temporal_networks import vertex_properties

   vp = vertex_properties(graphs, node_name="teacher_Ines",
                          graph_labels=labels, visualisation=False)

::

   teacher_Ines   constraint=0.236  betweenness= 15.19
   student_A2     constraint=0.324  betweenness=  5.07

The teacher who moves between the two classes is the less constrained of
the two and carries roughly three times the betweenness — the expected
signature of an actor spanning otherwise separate groups.


Running it
----------

.. code-block:: bash

   python examples/example_social_contacts.py

.. important::

   The script calls ``ig.set_random_number_generator(random.Random(42))``
   before any analysis. Community detection is randomised, and seeding
   :mod:`random` alone does **not** affect igraph's generators — without
   that line the lineage counts and ids above change on every run. Seed
   igraph explicitly in any analysis whose results you intend to quote.

The script is seeded and self-checking: it asserts that exactly one gap is
found, that the pair straddling it is ``NaN``, that the two switchers are
detected in their new class, and that the bridging teacher is less
constrained than a student inside one class. No files are written unless
you pass ``save_path``.

See also the :doc:`epidemic_abm` tutorial, which uses time-respecting
paths on contact data from an agent-based model.
