"""
Integration tests for Q64 core dynamics engine.

Tests the complete pipeline: MI estimation → projection → convergence.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

from q64 import Q64DynamicsEngine


class TestQ64Integration:
    """Integration tests for Q64DynamicsEngine."""

    def test_convergence_on_synthetic_data(self):
        """Test that Q64 converges on simple synthetic data."""
        np.random.seed(42)
        S = np.random.randn(100, 5)

        engine = Q64DynamicsEngine(tau=0.1, eta=0.1, k_nn=3)
        result = engine.run(S, max_iterations=500)

        assert result['converged'], "Q64 should converge on random data"
        assert result['n_iterations'] < 500, "Should converge before timeout"
        assert result['rank_final'] > 0, "Final rank should be positive"
        assert result['L_final'] is not None, "Drift functional should be computed"

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        np.random.seed(42)
        N, d = 50, 3
        S = np.random.randn(N, d)

        engine = Q64DynamicsEngine()
        result = engine.run(S, max_iterations=300)

        assert result['S_final'].shape == (N, d), "Output state shape should match input"
        assert result['M_final'].shape == (d, d), "MI matrix should be (d, d)"
        assert isinstance(result['L_final'], (float, np.floating)), "L should be scalar"
        assert isinstance(result['converged'], bool), "converged should be boolean"

    def test_hierarchy_storage(self):
        """Test that history is properly stored."""
        np.random.seed(42)
        S = np.random.randn(80, 4)

        engine = Q64DynamicsEngine(eta=0.1)
        result = engine.run(S, max_iterations=400)

        history = result['history']
        n_iter = result['n_iterations']

        assert len(history['L']) == n_iter, "L history should match iterations"
        assert len(history['rank']) == n_iter, "Rank history should match iterations"
        assert len(history['S']) == n_iter, "State history should match iterations"
        assert len(history['M']) == n_iter, "MI history should match iterations"

    def test_drift_monotonicity(self):
        """Test that drift functional increases monotonically (then plateaus)."""
        np.random.seed(42)
        S = np.random.randn(100, 5)

        engine = Q64DynamicsEngine(tau=0.1, eta=0.05)
        result = engine.run(S, max_iterations=500)

        L_history = result['history']['L']

        # L should be non-decreasing (allowing for small numerical jitter)
        diffs = np.diff(L_history)
        num_decreases = np.sum(diffs < -1e-10)

        assert num_decreases == 0, f"L should not decrease; found {num_decreases} decreases"

    def test_rank_stability_at_convergence(self):
        """Test that rank stabilizes when converged."""
        np.random.seed(42)
        S = np.random.randn(120, 6)

        engine = Q64DynamicsEngine(tau=0.1, eta=0.1)
        result = engine.run(S, max_iterations=500)

        if result['converged']:
            rank_history = result['history']['rank']
            # Last 5 ranks should be identical
            final_ranks = rank_history[-5:]
            assert len(set(final_ranks)) == 1, "Final ranks should be stable"

    def test_different_tau_values(self):
        """Test convergence with different threshold values."""
        np.random.seed(42)
        S = np.random.randn(80, 4)

        for tau in [0.05, 0.1, 0.2, 0.5]:
            engine = Q64DynamicsEngine(tau=tau, eta=0.1)
            result = engine.run(S, max_iterations=500)

            assert result['n_iterations'] > 0, f"Should run with tau={tau}"
            # Higher tau should converge faster
            assert result['rank_final'] <= 4, "Rank should not exceed dimensionality"

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        S = np.random.RandomState(42).randn(100, 5)

        engine1 = Q64DynamicsEngine(tau=0.1, eta=0.1)
        result1 = engine1.run(S.copy(), max_iterations=500)

        engine2 = Q64DynamicsEngine(tau=0.1, eta=0.1)
        result2 = engine2.run(S.copy(), max_iterations=500)

        np.testing.assert_array_almost_equal(
            result1['S_final'], result2['S_final'],
            decimal=10,
            err_msg="Results should be reproducible"
        )

        assert result1['n_iterations'] == result2['n_iterations'], \
            "Iteration count should be identical"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
