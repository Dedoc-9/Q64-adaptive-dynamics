#!/usr/bin/env python
"""
Q64 Basic Usage Example

Minimal working example demonstrating the core Q64 system on synthetic data.

Run: python examples/basic_usage.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

from q64 import Q64DynamicsEngine


def main():
    """Run basic Q64 analysis on synthetic data."""

    print("=" * 70)
    print("Q64: Adaptive Representational Dynamics - Basic Usage Example")
    print("=" * 70)
    print()

    # ========================================================================
    # Step 1: Generate Synthetic Data
    # ========================================================================
    print("Step 1: Generating synthetic hierarchical data...")
    np.random.seed(42)

    N = 100      # Number of samples
    d = 5        # Dimensionality

    # Create data with some structure (3 clusters)
    cluster_size = N // 3
    S = np.vstack([
        np.random.randn(cluster_size, d) + np.array([2, 0, 0, 0, 0]),
        np.random.randn(cluster_size, d) + np.array([-2, 0, 0, 0, 0]),
        np.random.randn(cluster_size, d) + np.array([0, 2, 2, 0, 0]),
    ])

    print(f"  Generated S: shape {S.shape}, {len(np.unique(np.argmax(S, axis=1)))} clusters")
    print()

    # ========================================================================
    # Step 2: Initialize Q64 Engine
    # ========================================================================
    print("Step 2: Initializing Q64 engine...")
    engine = Q64DynamicsEngine(
        tau=0.1,           # SVD threshold
        eta=0.1,           # State update rate
        k_nn=3,            # k-NN parameter for MI estimation
        r=None             # No rank cap
    )
    print(f"  tau={engine.tau}, eta={engine.eta}, k_nn={engine.k_nn}")
    print()

    # ========================================================================
    # Step 3: Run Analysis
    # ========================================================================
    print("Step 3: Running Q64 analysis...")
    result = engine.run(S, max_iterations=500)

    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['n_iterations']}")
    print(f"  Final rank: {result['rank_final']}")
    print(f"  Final drift L: {result['L_final']:.6e}")
    print()

    # ========================================================================
    # Step 4: Inspect Results
    # ========================================================================
    print("Step 4: Analyzing results...")

    S_final = result['S_final']
    M_final = result['M_final']
    history = result['history']

    print(f"  Output state shape: {S_final.shape}")
    print(f"  Output state norm: {np.linalg.norm(S_final):.4f}")
    print(f"  MI matrix rank: {np.linalg.matrix_rank(M_final)}")
    print(f"  MI matrix spectral norm: {np.linalg.norm(M_final, ord=2):.6e}")
    print()

    # ========================================================================
    # Step 5: Analyze Convergence History
    # ========================================================================
    print("Step 5: Convergence history...")

    L_history = history['L']
    rank_history = history['rank']

    print(f"  Initial L: {L_history[0]:.6e}")
    print(f"  Final L:   {L_history[-1]:.6e}")
    print(f"  L change:  {L_history[-1] - L_history[0]:.6e}")
    print()

    print(f"  Initial rank: {rank_history[0]}")
    print(f"  Final rank:   {rank_history[-1]}")
    print(f"  Rank stable: {len(set(rank_history[-5:])) == 1}")
    print()

    # ========================================================================
    # Step 6: Summary
    # ========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if result['converged']:
        print("✓ Q64 converged successfully")
        print(f"  - Found stable structure at iteration {result['n_iterations']}")
        print(f"  - Final structural rank: {result['rank_final']}")
        print(f"  - Drift functional L plateaued")
    else:
        print("⚠ Q64 did not converge (timeout)")
        print(f"  - Ran {result['n_iterations']} iterations")
        print("  - May indicate insufficient data structure or need for parameter tuning")

    print()
    print("Next steps:")
    print("  1. Try varying τ (threshold) for different sensitivity")
    print("  2. Experiment with η (step size) for convergence speed")
    print("  3. Run on your own data")
    print("  4. See docs/IMPLEMENTATION_NOTES.md for parameter tuning")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
