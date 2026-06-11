"""Sphinx configuration for the temporal_networks documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import temporal_networks  # noqa: E402

project = "temporal_networks"
author = "Adriana-Stefania Ciupeanu, Julien Arino"
copyright = "2026, Adriana-Stefania Ciupeanu, Julien Arino"
release = temporal_networks.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# NumPy-style docstrings only
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autosummary_generate = True
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "igraph": ("https://python.igraph.org/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"temporal_networks {release}"
