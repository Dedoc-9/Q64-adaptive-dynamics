"""
Q64 Empirical Validation Examples

Demonstrates:
1. Mean-centered telemetry preprocessing
2. Spectral analysis (r_eff, entropy)
3. Regime persistence characterization
4. Subspace angle stability
5. Transition alignment (vs. random baseline)
6. Full Q64 convergence validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from refined_protocol.core_dynamics_empirical import StreamOrientedQ64Engine, validate_stream_q64
from refined_protocol.analysis_code_library import (
    preprocess_telemetry,
    analyze_spectral_structure,
    discover_regimes_unsupervised,
    compute_regime_persistence,
    compute_subspace_angles,
    analyze_subspace_stability,
    transition_alignment_with_baseline,
    evaluate_hypothesis,
    export_analysis_summary
)


# ============================================================================
# EXAMPLE 1: Generate Synthetic Telemetry
# ============================================================================

def generate_synthetic_telemetry_hierarchical(n_samples: int = 108000,
                                             n_regimes: int = 8,
                                             noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic game telemetry with multiple stable regimes.

    Structure:
    - Regime 1: Menu (low load, low temp)
    - Regime 2: Gameplay (medium load, stable)
    - Regime 3: Combat (high load, rising temp)
    - Regime 4: Traversal (streaming, variable)
    - Regimes 5–8: Variations with transient switches

    Args:
        n_samples: Total frames (default 30 min × 60 FPS = 108k)
        n_regimes: Number of distinct regimes
        noise_level: Gaussian noise std

    Returns:
        telemetry (N, 7), scene_labels (N,)
    """

    # Regime parameters (frame_time, gpu_load, cpu_load, soc_temp, input_lag, power_draw)
    regime_centers = {
        0: [16.5, 0.3, 0.2, 40, 2.0, 8],     # Menu
        1: [16.7, 0.6, 0.4, 48, 2.5, 15],    # Gameplay
        2: [17.0, 0.85, 0.7, 58, 3.0, 24],   # Combat
        3: [18.5, 0.7, 0.5, 52, 2.8, 18],    # Traversal (streaming)
        4: [16.5, 0.3, 0.2, 40, 2.0, 8],     # Menu (repeat)
        5: [17.2, 0.65, 0.45, 50, 2.6, 16],  # Mixed gameplay
        6: [19.0, 0.8, 0.65, 55, 2.9, 22],   # High-intensity
        7: [18.0, 0.75, 0.55, 53, 2.8, 20],  # Streaming intense
    }

    regime_durations = [10000, 20000, 15000, 15000, 10000, 12000, 13000, 13000]  # Frames per regime

    telemetry = []
    scene_labels = []

    for regime_id, duration in enumerate(regime_durations):
        center = np.array(regime_centers[regime_id % len(regime_centers)])
        noise = np.random.randn(duration, 7) * noise_level
        regime_data = center[np.newaxis, :6] + noise  # 7-dim but only first 6 are the above
        # Add frame counter as 7th dimension
        regime_data = np.hstack([regime_data, np.arange(duration)[:, np.newaxis]])

        telemetry.append(regime_data)
        scene_labels.extend([f"regime_{regime_id}"] * duration)

    telemetry = np.vstack(telemetry)
    scene_labels = np.array(scene_labels)

    return telemetry, scene_labels


# ============================================================================
# EXAMPLE 2: Spectral Analysis (PRIMARY METRIC)
# ============================================================================

def example_spectral_analysis():
    """Analyze spectral structure of synthetic telemetry."""

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Spectral Analysis (PRIMARY METRICS)")
    print("=" * 80)

    # Generate synthetic data
    telemetry_raw, scene_labels = generate_synthetic_telemetry_hierarchical()

    # Preprocess: mean-center
    telemetry_centered = preprocess_telemetry(telemetry_raw, window_size=64)

    # Spectral analysis
    spectrum = analyze_spectral_structure(telemetry_centered)

    print(f"\n✓ Eigenvalues (top 7): {spectrum.eigenvalues}")
    print(f"✓ Effective rank (r_eff, entropy-based): {spectrum.r_eff:.2f}")
    print(f"  → Target: ≤14, Acceptable: ≤16, Failure: >18")
    print(f"  → Status: {'✅ PASS' if spectrum.r_eff <= 14 else ('⚠️ ACCEPTABLE' if spectrum.r_eff <= 16 else '❌ FAIL')}")

    print(f"\n✓ Spectral entropy H: {spectrum.entropy:.3f}")
    print(f"  → log(12) = {np.log(12):.3f}")
    print(f"  → Target: ≤log(12), Acceptable: ≤log(14), Failure: >log(16)")
    print(f"  → Status: {'✅ PASS' if spectrum.entropy <= np.log(12) else ('⚠️ ACCEPTABLE' if spectrum.entropy <= np.log(14) else '❌ FAIL')}")

    print(f"\n✓ Cumulative variance ranks (SECONDARY):")
    print(f"  r_90 (90% var): {spectrum.r_90}")
    print(f"  r_95 (95% var): {spectrum.r_95}")

    print(f"\n✓ Spectral decay ratio λ_1/λ_7: {spectrum.decay_ratio:.1f}")
    print(f"  → Target: >40, Acceptable: >20, Failure: <5")

    print(f"\n✓ Spectrum profile: {spectrum.profile}")
    print(f"  → 'smooth': concentrated, 'spiky': long tail, 'mixed': intermediate")

    return spectrum


# ============================================================================
# EXAMPLE 3: Regime Persistence (CRITICAL METRIC)
# ============================================================================

def example_regime_persistence():
    """Analyze regime persistence (separates meaningful states from noise)."""

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Regime Persistence (CRITICAL METRIC)")
    print("=" * 80)

    # Generate synthetic data
    telemetry_raw, scene_labels = generate_synthetic_telemetry_hierarchical()
    telemetry_centered = preprocess_telemetry(telemetry_raw, window_size=64)

    # Discover regimes unsupervised (no manual labels)
    transitions, ranks = discover_regimes_unsupervised(telemetry_centered, tau_rank=0.2)

    print(f"\n✓ Discovered {len(transitions)} rank transitions")
    print(f"  → Frame indices: {transitions[:10]}... (showing first 10)")

    # Compute persistence statistics
    persistence = compute_regime_persistence(transitions, len(telemetry_centered))

    print(f"\n✓ Stable regime (>100 frames):")
    print(f"  Median duration: {persistence.median_stable_frames:.0f} frames")
    print(f"  → Target: >300, Acceptable: >150, Failure: <50")

    print(f"\n✓ Transient regime (≤100 frames):")
    print(f"  Median duration: {persistence.median_transient_frames:.0f} frames")

    print(f"\n✓ Time distribution:")
    print(f"  % time in stable regimes: {persistence.pct_time_stable:.1f}%")
    print(f"  → Target: >70%, Acceptable: >60%, Failure: <40%")
    print(f"  → Status: {'✅ PASS' if persistence.pct_time_stable >= 70 else ('⚠️ ACCEPTABLE' if persistence.pct_time_stable >= 60 else '❌ FAIL')}")

    print(f"\n✓ Total regimes: {persistence.num_regimes}")

    return persistence


# ============================================================================
# EXAMPLE 4: Subspace Stability (NEW PRIMARY METRIC)
# ============================================================================

def example_subspace_stability():
    """Analyze principal angle stability (detects geometry rotation)."""

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Subspace Stability (NEW PRIMARY METRIC)")
    print("=" * 80)

    # Generate synthetic data
    telemetry_raw, scene_labels = generate_synthetic_telemetry_hierarchical()
    telemetry_centered = preprocess_telemetry(telemetry_raw, window_size=64)

    # Compute principal angles between subspaces
    theta_timeseries = compute_subspace_angles(telemetry_centered, k=8, window_step=10)

    print(f"\n✓ Computed {len(theta_timeseries)} principal angle measurements")

    # Analyze
    stability = analyze_subspace_stability(theta_timeseries)

    print(f"\n✓ Subspace angle statistics:")
    print(f"  Mean angle: {stability.mean_angle_rad:.3f} rad ({np.degrees(stability.mean_angle_rad):.1f}°)")
    print(f"  Median angle: {stability.median_angle_rad:.3f} rad ({np.degrees(stability.median_angle_rad):.1f}°)")
    print(f"  Max angle: {stability.max_angle_rad:.3f} rad ({np.degrees(stability.max_angle_rad):.1f}°)")

    print(f"\n✓ Stability classification:")
    print(f"  Target: <0.5 rad (<28°)")
    print(f"  Acceptable: <0.65 rad (<37°)")
    print(f"  Failure: >0.8 rad (>46°)")
    print(f"  → Status: {stability.stability.upper()}")

    if stability.stability == "stable":
        print(f"  → ✅ PASS: Subspace geometry is stable")
    else:
        print(f"  → ❌ FAIL: Subspace geometry rotates too rapidly (geometry instability)")

    return stability


# ============================================================================
# EXAMPLE 5: Transition Alignment (PRIMARY METRIC)
# ============================================================================

def example_transition_alignment():
    """Compare discovered transitions to manual labels with random baseline."""

    print("\n" + "=" * 80)
    print("EXAMPLE 5: Transition Alignment (PRIMARY METRIC)")
    print("=" * 80)

    # Generate synthetic data
    telemetry_raw, scene_labels = generate_synthetic_telemetry_hierarchical()
    telemetry_centered = preprocess_telemetry(telemetry_raw, window_size=64)

    # Discover regimes
    transitions, _ = discover_regimes_unsupervised(telemetry_centered, tau_rank=0.2)

    # Compute alignment vs. random baseline
    # Convert scene labels to numeric
    scene_numeric = np.array([hash(s) % 256 for s in scene_labels])

    alignment = transition_alignment_with_baseline(
        transitions, scene_numeric, frame_tolerance=30, n_bootstrap=1000
    )

    print(f"\n✓ Observed F1 score: {alignment.f1_observed:.3f}")
    print(f"✓ Random baseline F1 (mean ± std): {alignment.f1_random_mean:.3f} ± {alignment.f1_random_std:.3f}")
    print(f"✓ ΔF1 (signal gain over random): {alignment.delta_f1:.3f}")

    print(f"\n✓ Signal strength classification:")
    print(f"  Target: ΔF1 > 0.30 (strong)")
    print(f"  Acceptable: ΔF1 > 0.20 (moderate)")
    print(f"  Failure: ΔF1 < 0.10 (no signal)")
    print(f"  → Observed: {alignment.signal_strength.upper()}")

    if alignment.signal_strength == "strong":
        print(f"  → ✅ PASS: Discovered transitions align with semantic events")
    elif alignment.signal_strength == "weak":
        print(f"  → ⚠️ ACCEPTABLE: Weak signal; may require interpretation layer")
    else:
        print(f"  → ❌ FAIL: No semantic alignment (system detects noise, not events)")

    return alignment


# ============================================================================
# EXAMPLE 6: Full Q64 Convergence Validation
# ============================================================================

def example_q64_convergence():
    """End-to-end Q64 validation: convergence, rank stability, drift."""

    print("\n" + "=" * 80)
    print("EXAMPLE 6: Full Q64 Convergence Validation")
    print("=" * 80)

    # Generate synthetic data
    telemetry_raw, _ = generate_synthetic_telemetry_hierarchical()

    # Run Q64
    results = validate_stream_q64(telemetry_raw, k=8)

    print(f"\n✓ Convergence statistics:")
    print(f"  Convergence rate: {results['convergence_rate']:.1f}%")
    print(f"  → Target: >92%, Acceptable: >85%, Failure: <70%")

    print(f"\n✓ Time-to-convergence:")
    print(f"  Mean frames: {results['mean_ttc']:.1f}")
    print(f"  → Target: <5 frames, Acceptable: <8 frames")

    print(f"\n✓ Rank statistics:")
    print(f"  Mean rank: {float(np.mean(results['ranks'])):.1f}")
    print(f"  Rank variance: {float(np.var(results['ranks'])):.2f}")

    print(f"\n✓ Drift functional:")
    print(f"  Mean L_t: {float(np.mean(results['drifts'])):.4f}")
    print(f"  Drift std: {float(np.std(results['drifts'])):.4f}")

    print(f"\n✓ Final engine state:")
    state = results['final_state']
    for key, val in state.items():
        print(f"  {key}: {val}")

    return results


# ============================================================================
# EXAMPLE 7: Decision Gate (Accept/Reject H₁)
# ============================================================================

def example_decision_gate():
    """Evaluate whether metrics pass H₁ (structure exists)."""

    print("\n" + "=" * 80)
    print("EXAMPLE 7: Decision Gate (H₀ vs H₁)")
    print("=" * 80)

    # Run all analyses
    spectrum = analyze_spectral_structure(preprocess_telemetry(
        generate_synthetic_telemetry_hierarchical()[0], window_size=64
    ))

    telemetry_raw, scene_labels = generate_synthetic_telemetry_hierarchical()
    telemetry_centered = preprocess_telemetry(telemetry_raw, window_size=64)

    transitions, _ = discover_regimes_unsupervised(telemetry_centered)
    persistence = compute_regime_persistence(transitions, len(telemetry_centered))

    theta = compute_subspace_angles(telemetry_centered)
    stability = analyze_subspace_stability(theta)

    scene_numeric = np.array([hash(s) % 256 for s in scene_labels])
    alignment = transition_alignment_with_baseline(transitions, scene_numeric)

    # Build row for decision gate
    row = pd.Series({
        'r_eff': spectrum.r_eff,
        'entropy': spectrum.entropy,
        'pct_time_stable': persistence.pct_time_stable,
        'subspace_angle_median': stability.median_angle_rad,
        'delta_f1': alignment.delta_f1
    })

    # Evaluate
    verdict = evaluate_hypothesis(row)

    print(f"\n✓ Hypothesis evaluation:")
    print(f"  Criteria passing: {verdict['passing_count']}/5")
    for criterion, passed in verdict['criteria_passing'].items():
        status = "✅" if passed else "❌"
        print(f"    {status} {criterion}")

    print(f"\n✓ Decision:")
    if verdict['h1_accepted']:
        print(f"  → ✅ H₁ ACCEPTED (structure exists)")
        print(f"     Confidence: {verdict['confidence']:.1%}")
        print(f"     Next step: Proceed to v1.0 hardening")
    else:
        print(f"  → ❌ H₀ ACCEPTED (no structure / failure mode)")
        print(f"     Confidence: {1 - verdict['confidence']:.1%}")
        print(f"     Next step: Identify failure mode, pivot strategy")

    return verdict


# ============================================================================
# MAIN: RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Q64 EMPIRICAL VALIDATION EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate the falsifiable hypothesis testing framework.")
    print("Success requires ≥4 of 5 primary metrics passing simultaneously.")

    # Run examples
    spectrum = example_spectral_analysis()
    persistence = example_regime_persistence()
    stability = example_subspace_stability()
    alignment = example_transition_alignment()
    convergence = example_q64_convergence()
    verdict = example_decision_gate()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Collect real data on ASUS ROG Ally X (5 games, 30 min each)")
    print("  2. Apply preprocessing: mean-center with 64-frame sliding window")
    print("  3. Run analyses: spectrum, persistence, stability, alignment")
    print("  4. Evaluate decision gate (≥4 of 5 criteria)")
    print("  5. If H₁ accepted: proceed to v1.0; if H₀: pivot to failure mode strategy")
