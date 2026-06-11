# Contributing to temporal_networks

Thank you for your interest in contributing. This document explains how to 
set up a development environment, run the test suite, and submit changes.

## Getting started

Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/cadrianas/temporal_networks
cd NetworkPackage
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest tests/ -v
```

All tests should pass before you submit a pull request. If you are adding 
a new feature, please include tests that cover the new functionality.

## Code style

This project targets PEP 8 with a maximum line length of 88 characters. 
Docstrings follow the NumPy convention. Before submitting, check your 
changes with:

```bash
ruff check temporal_networks/
```

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes, including tests and updated docstrings.
3. Run the full test suite and confirm all tests pass.
4. Open a pull request with a clear description of what changed and why.

If you are unsure whether a change is in scope, open an issue first to 
discuss it before writing code.

## Reporting issues

Please open a GitHub issue at 
https://github.com/cadrianas/NetworkPackage/issues and include:
- A minimal reproducible example
- Your Python version and igraph version (`pip show igraph`)
- The full error message or unexpected output

## Authors

- Adriana-Stefania Ciupeanu (primary contact)
- Julien Arino

This package was developed as part of research at the University of Manitoba, 
Department of Mathematics.
