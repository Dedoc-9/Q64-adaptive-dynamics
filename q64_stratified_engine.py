"""
Q64 Stratified Multi-Domain Engine
====================================
Implementation of domain-stratified spectral analysis for heterogeneous telemetry.

Architecture:
  - Accepts 64-dimensional heterogeneous state vector
  - Stratifies into 4 independent domains
  - Computes convergence predicate c_t per domain
  - Returns per-domain stability profile

Week 1: Reference validation (synthetic data)
Week 2: Calibration (real game telemetry)
Week 3+: H₁ empirical validation
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DomainMetrics:
    """Per-domain convergence metrics"""
    c_t: bool                    # Convergence predicate
    rank: int                    # Estimated rank
    R_t: float                   # Residual norm
    L_t: float                   # Drift functional
    rank_stable: bool            # Rank stable over window?
    drift_stable: bool           # Drift bounded?
    time_converged: int          # Frame when c_t first triggered (-1 if never)


class Q64DomainEngine:
    """Single-domain Q64 operator with convergence detection"""

    def __init__(self, N: int, k: int, name: str, w: int = 20, tau: float = 0.4,
                 epsilon_R: float = 1e-2, delta_L: float = 0.2):
        """
        Initialize a domain-specific Q64 engine.

        Args:
            N: Dimension of this domain
            k: Max rank truncation
            name: Domain identifier (e.g., "input", "physics")
            w: Sliding window size (frames)
            tau: Rank threshold (fraction of max eigenvalue)
            epsilon_R: Residual threshold
            delta_L: Drift threshold (as fraction of L_t)
        """
        self.N = N
        self.k = min(k, N)  # Ensure k <= N
        self.name = name
        self.w = w
        self.tau = tau
        self.epsilon_R = epsilon_R
        self.delta_L = delta_L

        # State
        self.window: List[np.ndarray] = []
        self.G = np.zeros((N, N))
        self.L_prev = 0.0
        self.rank_history = [0] * 5
        self.frame_count = 0
        self.time_converged = -1
        self.converged_ever = False

    def update(self, s_t: np.ndarray) -> DomainMetrics:
        """
        Update the domain engine with a new state vector s_t.

        Args:
            s_t: (N,) state vector for this domain

        Returns:
            DomainMetrics: convergence metrics for this domain
        """
        self.frame_count += 1

        # Layer 1 & 2: Sliding window Gramian update
        self.window.append(s_t.copy())
        if len(self.window) > self.w:
            self.window.pop(0)

        if len(self.window) < 2:
            # Not enough data yet
            return DomainMetrics(
                c_t=False, rank=0, R_t=float('inf'), L_t=0.0,
                rank_stable=False, drift_stable=False,
                time_converged=-1
            )

        # Compute local covariance
        W = np.array(self.window)
        mu = np.mean(W, axis=0)
        W_centered = W - mu
        self.G = (W_centered.T @ W_centered) / len(self.window)

        # Layer 3: Spectral decomposition
        try:
            vals, vecs = np.linalg.eigh(self.G)
            idx = np.argsort(vals)[::-1]  # Descending order
            vals = vals[idx]
            vecs = vecs[:, idx]
        except np.linalg.LinAlgError:
            # Degenerate case (e.g., rank-deficient data)
            return DomainMetrics(
                c_t=False, rank=0, R_t=float('inf'), L_t=0.0,
                rank_stable=False, drift_stable=False,
                time_converged=self.frame_count
            )

        Lambda_k = vals[:self.k]
        U_k = vecs[:, :self.k]

        # Layer 4: Rank estimation
        if Lambda_k[0] > 0:
            rank_t = np.sum(Lambda_k > (self.tau * Lambda_k[0]))
        else:
            rank_t = 0

        self.rank_history.append(int(rank_t))
        self.rank_history.pop(0)

        # Layer 6 & 7: Projection operator and residual
        P_t = U_k @ np.diag(Lambda_k) @ U_k.T
        R_t = np.linalg.norm(self.G - P_t @ self.G, ord='fro')

        # Layer 8: Drift functional (trace of P_t @ P_t)
        L_t = np.trace(P_t @ P_t)

        # Layer 9: Convergence predicate (3 conditions)
        cond1 = R_t < self.epsilon_R                    # Residual bounded

        # Relaxed rank stability: allow rank to vary within ±1 of median
        median_rank = int(np.median(self.rank_history))
        rank_range = set(range(max(1, median_rank - 1), median_rank + 2))
        cond2 = all(r in rank_range for r in self.rank_history)  # Rank within ±1 of median

        cond3 = abs(L_t - self.L_prev) < (self.delta_L * L_t) if L_t > 0 else True  # Drift bounded

        c_t = cond1 and cond2 and cond3

        # Record first convergence
        if c_t and not self.converged_ever:
            self.time_converged = self.frame_count
            self.converged_ever = True

        self.L_prev = L_t

        return DomainMetrics(
            c_t=c_t,
            rank=int(rank_t),
            R_t=R_t,
            L_t=L_t,
            rank_stable=cond2,
            drift_stable=cond3,
            time_converged=self.time_converged
        )


class Q64StratifiedEngine:
    """Multi-domain Q64 engine with per-domain convergence tracking"""

    def __init__(self):
        """Initialize four independent domain engines"""
        self.domains = {
            "input": Q64DomainEngine(N=10, k=3, name="input", w=10),
            "physics": Q64DomainEngine(N=6, k=5, name="physics", w=10),
            "system": Q64DomainEngine(N=12, k=3, name="system", w=10),
            "rendering": Q64DomainEngine(N=36, k=10, name="rendering", w=10),
        }
        self.frame_count = 0
        self.metrics_history = {d: [] for d in self.domains}

    def update(self, s_t: np.ndarray) -> Dict[str, DomainMetrics]:
        """
        Update all domains with a new 64-dimensional state vector.

        Args:
            s_t: (64,) heterogeneous state vector

        Returns:
            Dict[domain_name] → DomainMetrics
        """
        if len(s_t) != 64:
            raise ValueError(f"Expected 64-dim state, got {len(s_t)}")

        self.frame_count += 1

        # Split by domain
        s_input = s_t[0:10]
        s_physics = s_t[10:16]
        s_system = s_t[16:28]
        s_rendering = s_t[28:64]

        results = {}
        for domain_name, domain_vec in [
            ("input", s_input),
            ("physics", s_physics),
            ("system", s_system),
            ("rendering", s_rendering),
        ]:
            engine = self.domains[domain_name]
            metrics = engine.update(domain_vec)
            results[domain_name] = metrics
            self.metrics_history[domain_name].append(metrics)

        return results

    def stability_profile(self) -> Dict[str, bool]:
        """Return current convergence status for all domains"""
        return {d: engine.converged_ever for d, engine in self.domains.items()}

    def h1_gate_evaluation(self) -> Tuple[bool, Dict]:
        """
        Evaluate H₁ success criteria (revised for stratified analysis).

        H₁ Success: At least 3 of 4 domains show pct_time_stable >= thresholds.

        Returns:
            (gate_passes: bool, detail: dict with per-domain metrics)
        """
        detail = {}
        domains_passing = 0

        for domain_name, engine in self.domains.items():
            if not self.metrics_history[domain_name]:
                detail[domain_name] = {"converged": False, "pct_stable": 0.0}
                continue

            metrics_list = self.metrics_history[domain_name]
            converged = engine.converged_ever
            pct_stable = sum(1 for m in metrics_list if m.c_t) / len(metrics_list) if metrics_list else 0.0

            # Domain-specific thresholds
            thresholds = {
                "input": 0.80,      # Input must be very stable
                "physics": 0.70,    # Physics stable most of the time
                "system": 0.60,     # System can vary (thermal, power)
                "rendering": 0.40,  # Rendering most complex; lower bar
            }

            passes = pct_stable >= thresholds[domain_name]
            detail[domain_name] = {
                "converged": converged,
                "pct_stable": pct_stable,
                "threshold": thresholds[domain_name],
                "passes": passes,
            }

            if passes:
                domains_passing += 1

        # H₁ success: 3+ domains pass
        gate_passes = domains_passing >= 3
        detail["domains_passing"] = domains_passing
        detail["h1_success"] = gate_passes

        return gate_passes, detail


def main():
    """
    Test the stratified engine on synthetic low-rank data.
    """
    print("\n" + "="*70)
    print(" Q64 STRATIFIED ENGINE: SYNTHETIC VALIDATION")
    print("="*70 + "\n")

    # Initialize stratified engine
    engine = Q64StratifiedEngine()

    # Generate synthetic 64-dim telemetry with domain-specific rank structure
    np.random.seed(42)

    # Per-domain true manifolds
    U_input = np.random.randn(10, 3)
    U_input, _ = np.linalg.qr(U_input)  # Orthonormalize

    U_physics = np.random.randn(6, 5)
    U_physics, _ = np.linalg.qr(U_physics)

    U_system = np.random.randn(12, 3)
    U_system, _ = np.linalg.qr(U_system)

    U_rendering = np.random.randn(36, 10)
    U_rendering, _ = np.linalg.qr(U_rendering)

    # Simulate 1000 frames
    n_frames = 1000
    noise_level = 0.02  # 2% noise

    print(f"Generating {n_frames} frames of synthetic stratified telemetry...")
    print(f"Noise level: {noise_level*100:.1f}%\n")

    for t in range(n_frames):
        # Generate latent codes per domain
        z_input = np.random.randn(3)
        z_physics = np.random.randn(5)
        z_system = np.random.randn(3)
        z_rendering = np.random.randn(10)

        # Project to observable space + add noise
        s_input = U_input @ z_input + noise_level * np.random.randn(10)
        s_physics = U_physics @ z_physics + noise_level * np.random.randn(6)
        s_system = U_system @ z_system + noise_level * np.random.randn(12)
        s_rendering = U_rendering @ z_rendering + noise_level * np.random.randn(36)

        # Concatenate
        s_t = np.concatenate([s_input, s_physics, s_system, s_rendering])

        # Update engine
        results = engine.update(s_t)

        # Print progress
        if (t + 1) % 200 == 0:
            print(f"Frame {t+1}:")
            for domain, metrics in results.items():
                status = "✓" if metrics.c_t else "✗"
                print(f"  {domain:12s}: rank={metrics.rank} R_t={metrics.R_t:.4f} {status}")
            print()

    # Evaluate H₁ gate
    print("\n" + "="*70)
    print(" H₁ GATE EVALUATION (Stratified)")
    print("="*70 + "\n")

    gate_passes, detail = engine.h1_gate_evaluation()

    for domain in ["input", "physics", "system", "rendering"]:
        d = detail[domain]
        status = "✓ PASS" if d.get("passes", False) else "✗ FAIL"
        print(f"{domain:12s}: pct_stable={d['pct_stable']:.1%} threshold={d['threshold']:.0%} {status}")

    print(f"\nDomains passing: {detail['domains_passing']}/4")
    print(f"\nH₁ Gate Result: {'✓ SUCCESS (≥3 domains pass)' if gate_passes else '✗ FAILURE (<3 domains pass)'}")

    return engine, detail


if __name__ == "__main__":
    engine, detail = main()
