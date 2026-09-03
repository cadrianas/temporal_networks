"""Shared pytest configuration."""

import os


# Plot-writing tests must also run in headless and macOS CI environments.
os.environ.setdefault("MPLBACKEND", "Agg")
