temporal_networks
=================

A Python package for analyzing temporal network evolution with automatic
gap detection.

``temporal_networks`` is designed for researchers who need to understand how
network structure and properties change over time. Unlike general-purpose
graph libraries such as `networkx <https://networkx.org/>`_ or
`igraph <https://igraph.org/>`_, it treats the temporal dimension as a
first-class feature, with native support for irregular time series and
missing data periods. Compared to other temporal network packages such as
`teneto <https://teneto.readthedocs.io/>`_ and
`dynetx <https://dynetx.readthedocs.io/>`_, ``temporal_networks`` focuses
specifically on gap-aware analysis and visualization — automatically
detecting and correctly representing seasonal closures, maintenance windows,
or crisis-driven interruptions rather than drawing misleading continuous
lines across missing periods.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api

Citation
--------

If you use ``temporal_networks`` in your research, please cite:

.. code-block:: bibtex

   @software{ciupeanu_arino_temporal_networks,
     author    = {Ciupeanu, Adriana-Stefania and Arino, Julien},
     title     = {temporal\_networks: A Python package for analyzing
                  temporal network evolution with automatic gap detection},
     year      = {2026},
     publisher = {GitHub},
     url       = {https://github.com/cadrianas/temporal_networks},
     note      = {Version 0.1.0}
   }

Indices
-------

* :ref:`genindex`
* :ref:`search`
