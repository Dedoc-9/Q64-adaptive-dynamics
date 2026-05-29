"""
Q64: Adaptive Representational Dynamics

Multi-scale structure discovery via spectral projection and frozen referential anchoring.

License: AGPL-3.0
Version: 1.0.0
"""

from .core_dynamics import (
    MutualInformationEstimator,
    ProjectionOperator,
    SpectralConvergenceCriterion,
    Q64DynamicsEngine,
)

__version__ = "1.0.0"
__author__ = "Q64 Collaborative Architecture"
__license__ = "AGPL-3.0-only"

__all__ = [
    "MutualInformationEstimator",
    "ProjectionOperator",
    "SpectralConvergenceCriterion",
    "Q64DynamicsEngine",
]
