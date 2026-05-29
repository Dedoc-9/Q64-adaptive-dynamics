#!/usr/bin/env python
"""
Setup script for Q64: Adaptive Representational Dynamics
Supports both pure Python and Rust-accelerated installations.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
requirements = [
    "numpy>=1.20",
    "scipy>=1.7",
    "scikit-learn>=1.0",
]

dev_requirements = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "flake8>=6.0",
    "mypy>=1.0",
    "isort>=5.12",
]

docs_requirements = [
    "sphinx>=5.0",
    "sphinx-rtd-theme>=1.0",
]

examples_requirements = [
    "jupyter>=1.0",
    "matplotlib>=3.5",
]

setup(
    name="q64-adaptive-dynamics",
    version="1.0.0",
    author="Q64 Collaborative Architecture",
    author_email="research@q64.org",
    description="Q64: Adaptive Representational Dynamics - Multi-scale structure discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dedoc-9/Q64-adaptive-dynamics",
    project_urls={
        "Bug Tracker": "https://github.com/Dedoc-9/Q64-adaptive-dynamics/issues",
        "Documentation": "https://github.com/Dedoc-9/Q64-adaptive-dynamics/blob/main/docs/README.md",
        "Source Code": "https://github.com/Dedoc-9/Q64-adaptive-dynamics",
    },
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": docs_requirements,
        "examples": examples_requirements,
        "all": dev_requirements + docs_requirements + examples_requirements,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords=[
        "adaptive-dynamics",
        "representation-learning",
        "spectral-analysis",
        "mutual-information",
        "hierarchical-clustering",
    ],
    include_package_data=True,
    zip_safe=False,
)
