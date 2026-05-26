from setuptools import setup, find_packages

setup(
    name="temporal-network-analysis",
    version="0.1.0",
    description="Python package for analyzing dynamic networks over time with automatic gap detection",
    author="Adriana-Stefania Ciupeanu, Julien Arino",
    author_email="Julien.Arino@umanitoba.ca",
    license="GPL-3.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "igraph>=0.11.5",
        "networkx>=3.2.1",
        "matplotlib>=3.8.4",
        "seaborn>=0.13.2",
        "plotly>=5.22.0",
        "folium>=0.16.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
