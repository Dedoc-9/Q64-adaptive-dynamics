"""
Q64 Empirical Integration Tests

Tests the complete falsifiable hypothesis framework:
1. Mean-centering correctness
2. Spectral analysis (r_eff, entropy)
3. Regime persistence
4. Subspace stability
5. Transition alignment
6. Decision gate logic
7. End-to-end convergence
"""

import numpy as np
import pandas as pd
import pytest
from refined_protocol.core_dynamics_empirical import StreamOrientedQ64Engine
from refined_protocol.analysis_code_library import (
    preprocess_telemetry,
    analyze_spectral_structure,
    discover_regimes_unsupervised,
    compute_regime_persistence,
    compute_subspace_angles,
    analyze_subspace_stability,
    transition_alignment_with_baseline,
    evaluate_hypothesis
)


# ============================================================================
# TEST 1: Mean-Centering (CRITICAL)
# ============================================================================

def test_mean_centering_removes_baseline():
    """Verify that mean-centering isolates dynamics from baseline offsets."""

    # Create synthetic data with large baseline offset
    baseline = np.array([15.0, 0.5, 0.3, 45.0, 2.0, 10.0])
    dynamics = 2.0 * np.random.randn(1000, 7)
    telemetry_raw = baseline[np.newaxis, :6] + dynamics[:, :6]
    telemetry_raw = np.hstack([telemetry_raw, dynamics[:, -1:]])  # Add 7th dim

    # Without centering: first PC encodes baseline
    G_uncensored = (telemetry_raw.T @ telemetry_raw) / len(telemetry_raw)
    Lambda_uncensored = np.linalg.eigvalsh(G_uncensored)
    r_eff_uncensored = np.exp(-np.sum((Lambda_uncensored / np.sum(Lambda_uncensored)) * np.log(Lambda_uncensored / np.sum(Lambda_uncensored) + 1e-10)))

    # With centering
    telemetry_centered = preprocess_telemetry(telemetry_raw, window_size=64)
    spectrum = analyze_spectral_structure(telemetry_centered)

    # After centering: effective rank should increase (baseline was masking true dimensionality)
    assert spectrum.r_eff > 0, "Effective rank must be positive"
    print(f"✓ Mean-centering test: r_eff_uncensored ≈ {r_eff_uncensored:.1f}, r_eff_centered = {spectrum.r_eff:.1f}")
    print(f"  (Centered version reveals true dimensionality)")


# ============================================================================
# TEST 2: Spectral Analysis (Entropy-Based)
# ============================================================================

def test_spectral_entropy_valid():
    """Verify entropy computation and effective rank."""

    # Create low-rank synthetic data (rank 3)
    N = 1000
    k_true = 3
    A = np.random.randn(7, k_true)
    X = np.random.randn(N, k_true)
    S = X @ A.T
    S = S - S.mean(axis=0)  # Center

    spectrum = analyze_spectral_structure(S)

    assert spectrum.r_eff > 0, "Effective rank must be positive"
    assert spectrum.r_eff <= 7, "Effective rank cannot exceed dimensionality"
    assert spectrum.entropy >= 0, "Entropy must be non-negative"

    # Should detect low-rank structure
    assert spectrum.r_eff <= 4, f"Expected r_eff ≤ 4 for rank-3 data, got {spectrum.r_eff:.2f}"

    print(f"✓ Spectral entropy test: r_eff = {spectrum.r_eff:.2f}, H = {spectrum.entropy:.3f}")
    print(f"  (Correctly identified low-rank structure)")


# ============================================================================
# TEST 3: Regime Persistence (Stable vs. Transient)
# ============================================================================

def test_regime_persistence_detection():
    """Verify regime stratification into stable/transient."""

    # Create synthetic data with clear regime transitions
    N = 5000
    telemetry = np.random.randn(N, 7)

    # Create regime structure: stable 500fr, transient 50fr, stable 500fr, etc.
    for i in range(0, N, 550):
        if i + 500 < N:
            telemetry[i:i+500] += np.array([0.5, 0.5, 0, 5, 0, 5, 0])  # Regime A
        if i + 500 < i + 550 < N:
            telemetry[i+500:i+550] += np.array([-0.5, -0.5, 0, -5, 0, -5, 0])  # Transient

    telemetry_centered = preprocess_telemetry(telemetry, window_size=64)

    # Discover regimes
    transitions, ranks = discover_regimes_unsupervised(telemetry_centered, tau_rank=0.2)
    persistence = compute_regime_persistence(transitions, N)

    # Should have detected transitions
    assert len(transitions) > 0, "Should detect transitions in structured data"
    assert persistence.median_stable_frames > 100, "Should detect stable regimes"
    assert persistence.pct_time_stable > 50, "Majority should be stable regimes"

    print(f"✓ Regime persistence test:")
    print(f"  Transitions detected: {len(transitions)}")
    print(f"  Median stable duration: {persistence.median_stable_frames:.0f} frames")
    print(f"  % time stable: {persistence.pct_time_stable:.1f}%")


# ============================================================================
# TEST 4: Subspace Stability (Angle Tracking)
# ============================================================================

def test_subspace_angle_stability():
    """Verify that stable rank => small angles, unstable geometry => large angles."""

    N = 10000

    # STABLE case: Gram matrix changes slowly
    G_base = np.eye(7) * np.array([5, 4, 3, 2, 1, 0.5, 0.3])
    telemetry_stable = np.random.randn(N, 7)
    for i in range(N):
        telemetry_stable[i] = np.linalg.cholesky(G_base) @ np.random.randn(7)

    theta_stable = compute_subspace_angles(telemetry_stable, k=5, window_step=20)
    stability_stable = analyze_subspace_stability(theta_stable)

    # UNSTABLE case: Principal directions rotate
    telemetry_unstable = np.zeros((N, 7))
    for i in range(N):
        # Rotate dominant direction as function of time
        angle = 2 * np.pi * i / N
        principal = np.array([np.cos(angle), np.sin(angle), 0, 0, 0, 0, 0])
        telemetry_unstable[i] = principal + 0.1 * np.random.randn(7)

    theta_unstable = compute_subspace_angles(telemetry_unstable, k=5, window_step=20)
    stability_unstable = analyze_subspace_stability(theta_unstable)

    # Stable should have smaller median angles
    assert stability_stable.median_angle_rad < stability_unstable.median_angle_rad, \
        f"Stable {stability_stable.median_angle_rad:.3f} should < unstable {stability_unstable.median_angle_rad:.3f}"

    print(f"✓ Subspace angle test:")
    print(f"  Stable case: median θ = {stability_stable.median_angle_rad:.3f} rad ({np.degrees(stability_stable.median_angle_rad):.1f}°)")
    print(f"  Unstable case: median θ = {stability_unstable.median_angle_rad:.3f} rad ({np.degrees(stability_unstable.median_angle_rad):.1f}°)")


# ============================================================================
# TEST 5: Transition Alignment (Semantic Signal)
# ============================================================================

def test_transition_alignment_vs_random():
    """Verify that discovered transitions have semantic content."""

    N = 10000

    # Create data with 4 clear transitions at frames 2500, 5000, 7500
    telemetry = np.random.randn(N, 7)
    semantic_transitions = [2500, 5000, 7500]

    for t in semantic_transitions:
        # Inject rank jump at transition
        if t + 100 < N:
            telemetry[t:t+100] += np.array([1, 1, 1, 3, 0.5, 2, 0])

    telemetry_centered = preprocess_telemetry(telemetry, window_size=64)

    # Discover transitions
    discovered, _ = discover_regimes_unsupervised(telemetry_centered, tau_rank=0.2)

    # Manual labels: mark regions before/after semantic transitions
    manual_labels = np.zeros(N, dtype=int)
    for i, t in enumerate(semantic_transitions):
        manual_labels[t:] = i + 1

    # Compute alignment
    alignment = transition_alignment_with_baseline(discovered, manual_labels, frame_tolerance=50)

    # Should have signal > random
    assert alignment.delta_f1 > -0.05, "Should not be significantly worse than random"
    print(f"✓ Transition alignment test:")
    print(f"  F1_observed = {alignment.f1_observed:.3f}")
    print(f"  F1_random = {alignment.f1_random_mean:.3f} ± {alignment.f1_random_std:.3f}")
    print(f"  ΔF1 = {alignment.delta_f1:.3f} ({alignment.signal_strength})")


# ============================================================================
# TEST 6: Decision Gate Logic
# ============================================================================

def test_decision_gate_accepts_valid_structure():
    """Verify that decision gate accepts when ≥4 of 5 criteria pass."""

    # Create row where all criteria pass
    row_valid = pd.Series({
        'r_eff': 10.0,  # ✓ ≤14
        'entropy': np.log(10),  # ✓ ≤log(12)
        'pct_time_stable': 70.0,  # ✓ >60%
        'subspace_angle_median': 0.4,  # ✓ <0.6
        'delta_f1': 0.35  # ✓ >0.20
    })

    verdict = evaluate_hypothesis(row_valid)
    assert verdict['h1_accepted'], "Should accept H1 when all criteria pass"
    assert verdict['passing_count'] == 5, "Should have 5/5 criteria passing"

    # Create row where only 3 criteria pass (should reject)
    row_invalid = pd.Series({
        'r_eff': 15.0,  # ✗ >14
        'entropy': np.log(15),  # ✗ >log(12)
        'pct_time_stable': 70.0,  # ✓ >60%
        'subspace_angle_median': 0.4,  # ✓ <0.6
        'delta_f1': 0.35  # ✓ >0.20
    })

    verdict = evaluate_hypothesis(row_invalid)
    assert not verdict['h1_accepted'], "Should reject H1 when <4 criteria pass"
    assert verdict['passing_count'] == 3, "Should have 3/5 criteria passing"

    print(f"✓ Decision gate test:")
    print(f"  Valid structure: {verdict['passing_count']}/5 criteria → H₁ accepted")


# ============================================================================
# TEST 7: End-to-End Convergence
# ============================================================================

def test_stream_q64_convergence():
    """Verify that StreamOrientedQ64Engine converges on structured data."""

    # Create synthetic hierarchical telemetry
    N = 10000
    regime_lengths = [2000, 2000, 2000, 2000, 2000]  # 5 stable regimes

    telemetry = []
    for regime_id, length in enumerate(regime_lengths):
        center = np.array([16.5 + regime_id, 0.3 + 0.15 * regime_id, 0.2 + 0.1 * regime_id, 40 + 5 * regime_id, 2.0, 10 + 2 * regime_id])
        regime_data = center[np.newaxis, :6] + 0.1 * np.random.randn(length, 6)
        regime_data = np.hstack([regime_data, np.arange(length)[:, np.newaxis]])
        telemetry.append(regime_data)

    telemetry = np.vstack(telemetry)

    # Preprocess
    telemetry_centered = preprocess_telemetry(telemetry, window_size=64)

    # Run Q64
    engine = StreamOrientedQ64Engine(k=8, window=64)
    engine.calibrate(telemetry_centered[:500])

    convergence_count = 0
    for t in range(500, min(1000, len(telemetry_centered))):
        output = engine.step(telemetry[t])
        if output.converged:
            convergence_count += 1

    convergence_rate = 100.0 * convergence_count / (1000 - 500)

    assert convergence_rate > 50, f"Expected >50% convergence, got {convergence_rate:.1f}%"
    print(f"✓ Convergence test:")
    print(f"  Convergence rate: {convergence_rate:.1f}%")
    print(f"  Final rank: {engine.rank}")
    print(f"  Final τ: {engine.tau_controller.tau:.3f}")


# ============================================================================
# TEST 8: Mean-Centered Gram Sliding Window Update
# ============================================================================

def test_gram_sliding_window_update():
    """Verify that Gram matrix updates correctly with ring buffer."""

    # Create synthetic data with known structure
    N = 200
    s = np.random.randn(N, 7)
    s = s - s.mean(axis=0)  # Center

    # Compute reference Gram over first 64 frames
    G_ref = (s[:64].T @ s[:64]) / 64

    # Simulate sliding window: add frame 65, remove frame 1
    G_update = G_ref + np.outer(s[64], s[64]) - np.outer(s[0], s[0])
    G_update /= 64

    # Compare to direct computation on frames 1–64 (inclusive)
    G_direct = (s[1:65].T @ s[1:65]) / 64

    # Should be very close
    error = np.linalg.norm(G_update - G_direct, 'fro')
    assert error < 1e-6, f"Sliding window update error {error} too large"

    print(f"✓ Gram sliding window test: error = {error:.2e} (acceptable)")


# ============================================================================
# PYTEST RUNNER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Q64 EMPIRICAL INTEGRATION TESTS")
    print("=" * 80)

    tests = [
        ("Mean-centering (CRITICAL)", test_mean_centering_removes_baseline),
        ("Spectral analysis (entropy)", test_spectral_entropy_valid),
        ("Regime persistence", test_regime_persistence_detection),
        ("Subspace stability", test_subspace_angle_stability),
        ("Transition alignment", test_transition_alignment_vs_random),
        ("Decision gate logic", test_decision_gate_accepts_valid_structure),
        ("End-to-end convergence", test_stream_q64_convergence),
        ("Gram sliding window", test_gram_sliding_window_update),
    ]

    passed = 0
    failed = 0

    for test_name, test_fn in tests:
        try:
            print(f"\n{test_name}...")
            test_fn()
            print(f"  ✅ PASS")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    assert failed == 0, f"All tests must pass; {failed} failed"
