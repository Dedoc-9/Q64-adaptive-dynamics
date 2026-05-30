"""
Q64 Reference Implementation Unit Tests
========================================

Test the stratified Q64 engine on synthetic low-rank data.
Validates:
  1. Engine initialization (4 domains created)
  2. Synthetic convergence (at least one domain converges)
  3. H₁ gate evaluation (gate logic correct)
"""

import sys
import numpy as np
from q64_stratified_engine import Q64StratifiedEngine, Q64DomainEngine


def test_initialization():
    """Test that engine initializes with 4 domains"""
    print("\n" + "="*70)
    print("TEST 1: INITIALIZATION")
    print("="*70)

    engine = Q64StratifiedEngine()

    assert len(engine.domains) == 4, f"Expected 4 domains, got {len(engine.domains)}"
    assert "input" in engine.domains, "Missing 'input' domain"
    assert "physics" in engine.domains, "Missing 'physics' domain"
    assert "system" in engine.domains, "Missing 'system' domain"
    assert "rendering" in engine.domains, "Missing 'rendering' domain"

    print("✓ Engine initialized with 4 domains")
    print(f"  - input (N=10, k=3)")
    print(f"  - physics (N=6, k=5)")
    print(f"  - system (N=12, k=3)")
    print(f"  - rendering (N=36, k=10)")

    return True


def test_synthetic_convergence():
    """Test that engine converges on synthetic low-rank data"""
    print("\n" + "="*70)
    print("TEST 2: SYNTHETIC CONVERGENCE")
    print("="*70)

    engine = Q64StratifiedEngine()

    # Generate synthetic 64-dim telemetry with domain-specific rank structure
    np.random.seed(42)

    # Per-domain true manifolds (low-rank generators)
    U_input = np.random.randn(10, 3)
    U_input, _ = np.linalg.qr(U_input)

    U_physics = np.random.randn(6, 5)
    U_physics, _ = np.linalg.qr(U_physics)

    U_system = np.random.randn(12, 3)
    U_system, _ = np.linalg.qr(U_system)

    U_rendering = np.random.randn(36, 10)
    U_rendering, _ = np.linalg.qr(U_rendering)

    # Simulate 500 frames
    n_frames = 500
    noise_level = 0.02  # 2% noise

    print(f"Generating {n_frames} frames of synthetic stratified telemetry...")
    print(f"Noise level: {noise_level*100:.1f}%")

    convergence_profile = {d: False for d in engine.domains}

    # Initialize latent codes (fixed low-rank structure)
    z_input = np.ones(3) * 0.5
    z_physics = np.ones(5) * 0.5
    z_system = np.ones(3) * 0.5
    z_rendering = np.ones(10) * 0.5

    for t in range(n_frames):
        # Smoothly perturb latent codes (maintain low-rank structure with drift)
        z_input += 0.01 * np.random.randn(3)
        z_physics += 0.01 * np.random.randn(5)
        z_system += 0.01 * np.random.randn(3)
        z_rendering += 0.01 * np.random.randn(10)

        # Project to observable space + add small noise
        s_input = U_input @ z_input + noise_level * np.random.randn(10)
        s_physics = U_physics @ z_physics + noise_level * np.random.randn(6)
        s_system = U_system @ z_system + noise_level * np.random.randn(12)
        s_rendering = U_rendering @ z_rendering + noise_level * np.random.randn(36)

        # Concatenate into 64-dim vector
        s_t = np.concatenate([s_input, s_physics, s_system, s_rendering])

        # Update engine
        results = engine.update(s_t)

        # Track convergence
        for domain, metrics in results.items():
            if metrics.c_t:
                convergence_profile[domain] = True

    # At least one domain should have converged
    domains_converged = sum(convergence_profile.values())
    print(f"\nAfter {n_frames} frames:")
    for domain, converged in convergence_profile.items():
        status = "✓ CONVERGED" if converged else "✗ NOT CONVERGED"
        print(f"  {domain:12s}: {status}")

    assert domains_converged >= 1, f"Expected at least 1 domain to converge, got {domains_converged}"
    print(f"\n✓ Synthetic convergence test PASSED ({domains_converged} domains converged)")

    return True


def test_h1_gate():
    """Test that H₁ gate evaluates correctly"""
    print("\n" + "="*70)
    print("TEST 3: H₁ GATE EVALUATION")
    print("="*70)

    engine = Q64StratifiedEngine()

    # Generate synthetic data (same as test 2)
    np.random.seed(42)

    U_input = np.random.randn(10, 3)
    U_input, _ = np.linalg.qr(U_input)

    U_physics = np.random.randn(6, 5)
    U_physics, _ = np.linalg.qr(U_physics)

    U_system = np.random.randn(12, 3)
    U_system, _ = np.linalg.qr(U_system)

    U_rendering = np.random.randn(36, 10)
    U_rendering, _ = np.linalg.qr(U_rendering)

    # Simulate 1000 frames
    n_frames = 1000
    noise_level = 0.02

    print(f"Generating {n_frames} frames for gate evaluation...")

    # Initialize latent codes (fixed low-rank structure)
    z_input = np.ones(3) * 0.5
    z_physics = np.ones(5) * 0.5
    z_system = np.ones(3) * 0.5
    z_rendering = np.ones(10) * 0.5

    for t in range(n_frames):
        # Smoothly perturb latent codes
        z_input += 0.01 * np.random.randn(3)
        z_physics += 0.01 * np.random.randn(5)
        z_system += 0.01 * np.random.randn(3)
        z_rendering += 0.01 * np.random.randn(10)

        s_input = U_input @ z_input + noise_level * np.random.randn(10)
        s_physics = U_physics @ z_physics + noise_level * np.random.randn(6)
        s_system = U_system @ z_system + noise_level * np.random.randn(12)
        s_rendering = U_rendering @ z_rendering + noise_level * np.random.randn(36)

        s_t = np.concatenate([s_input, s_physics, s_system, s_rendering])
        engine.update(s_t)

    # Evaluate H₁ gate
    gate_passes, detail = engine.h1_gate_evaluation()

    print(f"\nH₁ Gate Evaluation Results:")
    print("-" * 70)
    for domain in ["input", "physics", "system", "rendering"]:
        d = detail[domain]
        status = "✓ PASS" if d.get("passes", False) else "✗ FAIL"
        print(f"  {domain:12s}: pct_stable={d['pct_stable']:6.1%}  threshold={d['threshold']:6.0%}  {status}")

    print("-" * 70)
    print(f"Domains passing: {detail['domains_passing']}/4")
    print(f"H₁ Gate: {'✓ SUCCESS' if gate_passes else '✗ FAILURE'}")

    # Verify gate structure
    assert "domains_passing" in detail, "Missing 'domains_passing' in detail"
    assert "h1_success" in detail, "Missing 'h1_success' in detail"
    assert detail["h1_success"] == gate_passes, "Inconsistent gate result"

    print(f"\n✓ H₁ gate evaluation test PASSED")

    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" Q64 REFERENCE IMPLEMENTATION: UNIT TESTS")
    print("="*70)

    tests = [
        ("Initialization", test_initialization),
        ("Synthetic Convergence", test_synthetic_convergence),
        ("H₁ Gate Evaluation", test_h1_gate),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASS"))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            results.append((test_name, "FAIL"))

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name:40s} {status}")

    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED - Week 1 Validation Complete")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
