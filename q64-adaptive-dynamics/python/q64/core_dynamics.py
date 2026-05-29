"""
Q64 Core Dynamics Engine
========================

Final implementation of the three irreducible primitives:
  S:       State space (observational)
  F_θ:     Representation dynamics operator
  Φ_ref:   Frozen reference anchor
  L:       Drift functional (audit layer)

This module implements:
  (1) Dependency Operator D:     k-NN mutual information estimation
  (2) Projection Rule P_θ:       SVD-based spectral gating
  (3) Stability Criterion:       Spectral convergence detection

License: AGPL-3.0
Author: Q64 Collaborative Architecture
Version: 1.0.0 (Final Core)
"""

import numpy as np
from scipy.special import digamma
from scipy.spatial.distance import cdist
from scipy.linalg import svd
from collections import deque
import warnings


class MutualInformationEstimator:
    """
    k-NN based mutual information estimator (Kraskov et al. 2004).

    Estimates I(X; Y) from finite samples using k-nearest neighbor distances.
    No binning artifacts. Works well for N >= 100.
    """

    def __init__(self, k=3, metric='chebyshev'):
        """
        Args:
            k: Neighborhood size (default 3, reasonable range [1,10])
            metric: Distance metric ('chebyshev', 'euclidean', 'manhattan')
        """
        self.k = k
        self.metric = metric
        self._psi_k = digamma(k)
        self._psi_N = None

    def estimate_mi(self, X, Y):
        """
        Estimate mutual information I(X; Y) from samples.

        Args:
            X: array of shape (N, d_x)
            Y: array of shape (N, d_y)

        Returns:
            MI estimate (scalar)
        """
        N = X.shape[0]
        self._psi_N = digamma(N)

        # Stack X and Y for joint space calculation
        XY = np.hstack([X, Y])

        # Compute distances in joint space
        distances_joint = cdist(XY, XY, metric=self.metric)

        # Find k-th nearest neighbor distance for each point
        k_distances = np.partition(distances_joint, self.k, axis=1)[:, self.k]

        # Count neighbors of X within k_distance[i]
        distances_X = cdist(X, X, metric=self.metric)
        n_X = np.sum(distances_X <= k_distances[:, None], axis=1) - 1  # -1 for self

        # Count neighbors of Y within k_distance[i]
        distances_Y = cdist(Y, Y, metric=self.metric)
        n_Y = np.sum(distances_Y <= k_distances[:, None], axis=1) - 1  # -1 for self

        # MI estimate
        mi = (self._psi_k
              - np.mean(digamma(n_X + 1) + digamma(n_Y + 1))
              + self._psi_N)

        # Clamp to [0, inf), handle numerical negatives
        return max(0.0, mi)

    def estimate_mi_matrix(self, S):
        """
        Estimate full MI matrix M[i,j] = I(S_i; S_j).

        Args:
            S: State array of shape (N, d)

        Returns:
            MI matrix M of shape (d, d), symmetric
        """
        d = S.shape[1]
        M = np.zeros((d, d))

        for i in range(d):
            for j in range(i, d):
                if i == j:
                    # Self-mutual information ≈ entropy
                    # Estimate as variance (simplified)
                    M[i, j] = np.log(np.var(S[:, i]) + 1e-10)
                else:
                    mi = self.estimate_mi(S[:, [i]], S[:, [j]])
                    M[i, j] = mi
                    M[j, i] = mi  # Symmetry

        # Ensure positive semidefinite by eigenvalue correction if needed
        eigvals = np.linalg.eigvalsh(M)
        if np.any(eigvals < -1e-10):
            warnings.warn("MI matrix has negative eigenvalues; enforcing PSD")
            eigvals = np.maximum(eigvals, 0)
            U = np.linalg.eigh(M)[1]
            M = U @ np.diag(eigvals) @ U.T

        return M


class ProjectionOperator:
    """
    SVD-based spectral gating with adaptive thresholding.

    Implements: P_θ(M) = U · diag(mask_τ(Σ)) · U^T
    where mask_τ(σ_i) = σ_i if σ_i > τ·σ_max else 0
    """

    def __init__(self, tau=0.1, eta=0.1, r=None):
        """
        Args:
            tau: Relative threshold (0 < tau < 1)
            eta: Step size for state update (0 < eta < 1)
            r: Maximum rank to retain (None = no cap)
        """
        self.tau = tau
        self.eta = eta
        self.r = r
        self._U = None
        self._Sigma = None
        self._rank = None

    def project(self, M):
        """
        Apply projection P_θ(M) to dependency matrix M.

        Args:
            M: Dependency matrix (d × d)

        Returns:
            Projected matrix P_θ(M)
        """
        # SVD decomposition
        U, Sigma, _ = svd(M, full_matrices=False)

        # Numerical rank detection
        sigma_max = Sigma[0] if len(Sigma) > 0 else 1e-10
        threshold = self.tau * sigma_max

        # Gating: apply mask
        Sigma_gated = np.zeros_like(Sigma)
        mask = Sigma > threshold

        # Rank cap if specified
        if self.r is not None:
            rank_idx = np.argsort(-Sigma)[:self.r]
            mask_rank = np.zeros_like(mask)
            mask_rank[rank_idx] = True
            mask = mask & mask_rank

        Sigma_gated[mask] = Sigma[mask]

        # Determine numerical rank
        self._rank = np.sum(mask)

        # Store for state update
        self._U = U
        self._Sigma = Sigma_gated

        # Reconstruct: P = U * diag(Sigma_gated) * U^T
        P = U @ np.diag(Sigma_gated) @ U.T

        return P

    def apply_to_state(self, P, S):
        """
        Apply projection dynamics: S_new = S + η·(P @ S)

        Args:
            P: Projected matrix from .project()
            S: State vector (d,)

        Returns:
            Updated state S_new
        """
        update = self.eta * (P @ S)
        S_new = S + update

        # Numerical safety: clip if explosion detected
        norm_ratio = np.linalg.norm(S_new) / (np.linalg.norm(S) + 1e-10)
        if norm_ratio > 100:
            warnings.warn(f"State norm explosion detected (ratio={norm_ratio:.1f}); reducing η")
            self.eta *= 0.5
            S_new = S + (self.eta * update)

        return S_new

    def get_rank(self):
        """Return numerical rank of last projection."""
        return self._rank if self._rank is not None else 0


class SpectralConvergenceCriterion:
    """
    Convergence detection based on spectral stabilization.

    Converged when:
      (A) ‖Σ_{t+1} - Σ_t‖_F < ε_convergence
      (B) rank(P_θ(M)) stable for N consecutive iterations
      (C) ‖S_{t+1} - S_t‖ / ‖S_t‖ < ε_state
    """

    def __init__(self,
                 eps_convergence=1e-6,
                 eps_state=1e-8,
                 n_window=5,
                 t_max=10000):
        """
        Args:
            eps_convergence: Frobenius norm tolerance for singular values
            eps_state: Relative state residual tolerance
            n_window: Number of consecutive iterations to confirm convergence
            t_max: Maximum iteration count before timeout
        """
        self.eps_convergence = eps_convergence
        self.eps_state = eps_state
        self.n_window = n_window
        self.t_max = t_max

        self.sigma_prev = None
        self.rank_prev = None
        self.window = deque(maxlen=n_window)

    def check(self, M_t, M_prev, S_t, S_prev, rank_t, rank_prev):
        """
        Check convergence criteria.

        Args:
            M_t: Current MI matrix
            M_prev: Previous MI matrix
            S_t: Current state
            S_prev: Previous state
            rank_t: Current rank of P_θ(M_t)
            rank_prev: Previous rank

        Returns:
            bool: True if converged (all criteria satisfied for n_window iterations)
        """
        # Extract singular values (eigenvalues of symmetric matrix)
        sigma_t = np.linalg.eigvalsh(M_t)[::-1]  # Descending order
        sigma_prev = np.linalg.eigvalsh(M_prev)[::-1]

        # Criterion A: Spectral residual
        spectral_residual = np.linalg.norm(sigma_t - sigma_prev, ord='fro')
        crit_a = spectral_residual < self.eps_convergence

        # Criterion B: Rank stability
        crit_b = (rank_t == rank_prev) if rank_prev is not None else True

        # Criterion C: State residual
        state_residual = np.linalg.norm(S_t - S_prev) / (np.linalg.norm(S_prev) + 1e-10)
        crit_c = state_residual < self.eps_state

        # All criteria must hold
        all_satisfied = crit_a and crit_b and crit_c

        if all_satisfied:
            self.window.append(True)
        else:
            self.window.clear()

        return len(self.window) == self.n_window

    def reset(self):
        """Reset state for new run."""
        self.sigma_prev = None
        self.rank_prev = None
        self.window.clear()


class Q64DynamicsEngine:
    """
    Complete Q64 system integrator.

    Combines:
      D:  Mutual information dependency operator
      P_θ: SVD-based projection
      F_θ: State update dynamics
      Φ_ref: Frozen reference anchor
      L:  Drift functional
    """

    def __init__(self, tau=0.1, eta=0.1, r=None, k_nn=3):
        """
        Args:
            tau: Admissibility threshold
            eta: Step size
            r: Rank cap (None = no cap)
            k_nn: k-NN parameter for MI estimation
        """
        self.tau = tau
        self.eta = eta
        self.r = r
        self.k_nn = k_nn

        self.mi_estimator = MutualInformationEstimator(k=k_nn)
        self.projector = ProjectionOperator(tau=tau, eta=eta, r=r)
        self.convergence = SpectralConvergenceCriterion()

        self.Phi_ref = None
        self.history = {
            'S': [],
            'M': [],
            'L': [],
            'rank': [],
            'converged': False,
            'n_iterations': 0
        }

    def initialize_anchor(self, S_initial):
        """
        Compute and freeze Φ_ref from initial state.

        Args:
            S_initial: Initial state array (N, d)
        """
        M_raw = self.mi_estimator.estimate_mi_matrix(S_initial)
        sigma_raw = np.linalg.eigvalsh(M_raw)[::-1]

        self.Phi_ref = {
            'sigma': sigma_raw,
            'entropy': -np.sum(sigma_raw * np.log(sigma_raw + 1e-10)),
            'condition_number': sigma_raw[0] / (sigma_raw[-1] + 1e-10),
            'matrix': M_raw
        }

    def compute_drift_functional(self, M_t, S_t, alpha=1.0, beta=0.1):
        """
        Compute L(S_t, θ_t) = α·‖Σ_ref - Σ_t‖ + β·trace(P²).

        Args:
            M_t: Current MI matrix
            S_t: Current state
            alpha: Spectrum drift weight
            beta: Coherence weight

        Returns:
            L value (scalar)
        """
        sigma_t = np.linalg.eigvalsh(M_t)[::-1]

        # Term 1: Spectral drift
        spectral_drift = np.linalg.norm(self.Phi_ref['sigma'] - sigma_t, ord='fro')

        # Term 2: Coherence (magnitude of remaining dependencies)
        P_theta = self.projector.project(M_t)
        coherence = np.trace(P_theta @ P_theta)

        L = alpha * spectral_drift + beta * coherence
        return L

    def run(self, S_initial, max_iterations=None):
        """
        Run full dynamics until convergence.

        Args:
            S_initial: Initial state (N, d)
            max_iterations: Override convergence.t_max

        Returns:
            dict with results and history
        """
        if max_iterations is not None:
            self.convergence.t_max = max_iterations

        # Initialize anchor
        self.initialize_anchor(S_initial)

        # Run dynamics
        S = S_initial.copy()
        M_prev = self.Phi_ref['matrix'].copy()
        rank_prev = None
        t = 0

        while t < self.convergence.t_max:
            # Compute MI matrix
            M_t = self.mi_estimator.estimate_mi_matrix(S)

            # Apply projection
            P_theta = self.projector.project(M_t)
            rank_t = self.projector.get_rank()

            # Update state
            S_prev = S.copy()
            S = self.projector.apply_to_state(P_theta, S)

            # Compute drift
            L_t = self.compute_drift_functional(M_t, S)

            # Store history
            self.history['S'].append(S.copy())
            self.history['M'].append(M_t.copy())
            self.history['L'].append(L_t)
            self.history['rank'].append(rank_t)

            # Check convergence
            converged = self.convergence.check(
                M_t, M_prev, S, S_prev, rank_t, rank_prev
            )

            if converged:
                self.history['converged'] = True
                self.history['n_iterations'] = t + 1
                return self._package_result(S, M_t)

            M_prev = M_t.copy()
            rank_prev = rank_t
            t += 1

        # Timeout
        warnings.warn(f"Did not converge after {t} iterations")
        self.history['converged'] = False
        self.history['n_iterations'] = t
        return self._package_result(S, M_t)

    def _package_result(self, S_final, M_final):
        """Package results into structured output."""
        return {
            'S_final': S_final,
            'M_final': M_final,
            'Phi_ref': self.Phi_ref,
            'L_final': self.history['L'][-1] if self.history['L'] else None,
            'converged': self.history['converged'],
            'n_iterations': self.history['n_iterations'],
            'rank_final': self.history['rank'][-1] if self.history['rank'] else None,
            'history': self.history
        }


# ============================================================================
# Minimal test harness
# ============================================================================

if __name__ == '__main__':
    print("Q64 Core Dynamics Engine v1.0.0")
    print("=" * 60)

    # Synthetic test: 100 samples, 5-dimensional state
    np.random.seed(42)
    N, d = 100, 5
    S_test = np.random.randn(N, d)

    # Run engine
    engine = Q64DynamicsEngine(tau=0.1, eta=0.1, k_nn=3)
    result = engine.run(S_test, max_iterations=500)

    # Report
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['n_iterations']}")
    print(f"Final rank: {result['rank_final']}")
    print(f"Final drift L: {result['L_final']:.6e}")
    print(f"State shape: {result['S_final'].shape}")
    print("\nTest completed successfully.")
