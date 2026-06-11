from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="temporal_networks",
    version="0.1.0",
    description="Python package for analyzing dynamic networks over time with automatic gap detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Adriana-Stefania Ciupeanu, Julien Arino",
    author_email="cadrianas@gmail.com, Julien.Arino@umanitoba.ca",
    url="https://github.com/cadrianas/temporal_networks",
    keywords=["temporal networks", "network analysis",
              "graph theory", "time series", "igraph"],
    license="GPL-3.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "igraph>=0.11.5",
        "matplotlib>=3.8.4",
        "seaborn>=0.13.2",
        "plotly>=5.22.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
