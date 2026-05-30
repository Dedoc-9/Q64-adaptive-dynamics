"""
Q64 Empirical Validation: Analysis Code Library
Corrected implementations for Phase 1–7 analysis pipeline

All functions handle mean-centered telemetry.
All metrics follow operationalized hypotheses (H0 vs H1).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SpectralProfile:
    """Spectral characterization result."""
    eigenvalues: np.ndarray
    r_eff: float  # Entropy-based effective rank (PRIMARY)
    entropy: float  # Spectral entropy H (PRIMARY)
    r_90: int  # Cumulative variance at 90% (SECONDARY)
    r_95: int  # Cumulative variance at 95% (SECONDARY)
    decay_ratio: float  # λ₁/λₖ (SUPPORTING)
    profile: str  # "smooth", "spiky", "mixed"


@dataclass
class RegimePersistence:
    """Regime lifetime characterization (PRIMARY METRIC)."""
    median_stable_frames: float
    median_transient_frames: float
    pct_time_stable: float
    pct_time_transient: float
    num_regimes: int
    transition_count: int


@dataclass
class SubspaceStability:
    """Principal angle stability result (PRIMARY METRIC)."""
    mean_angle_rad: float
    median_angle_rad: float
    max_angle_rad: float
    stability: str  # "stable" (<0.6) or "unstable"


@dataclass
class TransitionAlignment:
    """F1 vs. random baseline (PRIMARY METRIC)."""
    f1_observed: float
    f1_random_mean: float
    f1_random_std: float
    delta_f1: float
    signal_strength: str  # "strong" (>0.25), "weak" (>0.10), "none"


# ============================================================================
# PHASE 2: SPECTRAL ANALYSIS
# ============================================================================

def analyze_spectral_structure(telemetry_centered: np.ndarray) -> SpectralProfile:
    """
    Comprehensive manifold structure analysis.

    Input:
    - telemetry_centered: N_samples × 7 (already mean-centered per protocol)

    Returns:
    - SpectralProfile with r_eff (PRIMARY), entropy H (PRIMARY)

    CRITICAL: Input must be mean-centered!
    """

    s = telemetry_centered
    assert s.shape[1] == 7, "Expected 7-dimensional telemetry"

    # Centered Gram matrix
    G = (s.T @ s) / len(s)

    # Eigendecomposition
    Lambda = np.linalg.eigvalsh(G)
    Lambda = np.flip(Lambda)  # Descending order

    # Cumulative variance thresholds (SECONDARY)
    cum_var = np.cumsum(Lambda) / np.sum(Lambda)
    r_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    r_95 = int(np.searchsorted(cum_var, 0.95) + 1)

    # Effective rank via entropy (PRIMARY)
    p = Lambda / np.sum(Lambda)
    p_safe = p[p > 1e-10]
    H = -np.sum(p_safe * np.log(p_safe))
    r_eff = np.exp(H)

    # Spectral decay ratio
    decay_ratio = Lambda[0] / (Lambda[-1] + 1e-10)

    # Profile classification
    if r_eff < 0.8 * r_90:
        profile = "smooth"  # Entropy much less than cumulative-variance rank
    elif r_eff > 1.2 * r_90:
        profile = "spiky"   # Long tail pulls entropy up
    else:
        profile = "mixed"

    return SpectralProfile(
        eigenvalues=Lambda,
        r_eff=r_eff,
        entropy=H,
        r_90=r_90,
        r_95=r_95,
        decay_ratio=decay_ratio,
        profile=profile
    )


# ============================================================================
# PHASE 3: REGIME DISCOVERY & PERSISTENCE
# ============================================================================

def discover_regimes_unsupervised(telemetry_centered: np.ndarray,
                                   window: int = 64,
                                   tau_rank: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unsupervised rank-transition detection.
    Does NOT use manual scene labels for transition discovery.

    Returns:
    - transitions: frame indices where Δrank ≥ 2
    - ranks: rank timeseries
    """

    ranks = []

    for t in range(len(telemetry_centered) - window):
        s_window = telemetry_centered[t:t+window]
        G = (s_window.T @ s_window) / window
        Lambda = np.linalg.eigvalsh(G)
        Lambda = np.flip(Lambda)

        rank = np.sum(Lambda > tau_rank * Lambda[0])
        ranks.append(rank)

    rank_array = np.array(ranks)

    # Detect transitions: |Δrank| ≥ 2
    rank_deltas = np.abs(np.diff(rank_array))
    transitions = np.where(rank_deltas >= 2)[0]

    return transitions, rank_array


def compute_regime_persistence(transitions: np.ndarray,
                               total_frames: int) -> RegimePersistence:
    """
    Characterize regime lifetimes.

    PRIMARY METRIC: Separates meaningful states from noise.

    Returns:
    - median_stable_frames: >100 frame regime duration
    - pct_time_stable: % of session in stable regimes (target: >60%)
    """

    regime_durations = []
    prev_t = 0

    for t in transitions:
        duration = t - prev_t
        regime_durations.append(duration)
        prev_t = t

    # Final regime
    regime_durations.append(total_frames - prev_t)

    durations = np.array(regime_durations)

    # Stratify: stable > 100 frames
    stable_mask = durations > 100
    stable_durations = durations[stable_mask]
    transient_durations = durations[~stable_mask]

    return RegimePersistence(
        median_stable_frames=float(np.median(stable_durations)) if len(stable_durations) > 0 else np.nan,
        median_transient_frames=float(np.median(transient_durations)) if len(transient_durations) > 0 else np.nan,
        pct_time_stable=100.0 * np.sum(stable_durations) / total_frames if len(stable_durations) > 0 else 0.0,
        pct_time_transient=100.0 * np.sum(transient_durations) / total_frames if len(transient_durations) > 0 else 0.0,
        num_regimes=len(regime_durations),
        transition_count=len(transitions)
    )


# ============================================================================
# PHASE 4: SUBSPACE ANGLE STABILITY
# ============================================================================

def compute_subspace_angles(telemetry_centered: np.ndarray,
                           window: int = 64,
                           k: int = 8,
                           window_step: int = 10) -> np.ndarray:
    """
    Principal angles between adjacent subspaces.

    Detects: rank may be stable, but geometry rotates rapidly.

    Returns:
    - theta_timeseries: principal angles in radians
    """

    theta_timeseries = []

    for t in range(0, len(telemetry_centered) - window - window_step, window_step):
        s_t = telemetry_centered[t:t+window]
        s_t_plus = telemetry_centered[t+window_step:t+window+window_step]

        # Covariances
        G_t = (s_t.T @ s_t) / window
        G_tp = (s_t_plus.T @ s_t_plus) / window

        # Top-k eigenvectors
        Lambda_t, U_t = np.linalg.eigh(G_t)
        Lambda_t = np.flip(Lambda_t)
        U_t = np.flip(U_t, axis=1)

        Lambda_tp, U_tp = np.linalg.eigh(G_tp)
        Lambda_tp = np.flip(Lambda_tp)
        U_tp = np.flip(U_tp, axis=1)

        U_t_k = U_t[:, :min(k, U_t.shape[1])]
        U_tp_k = U_tp[:, :min(k, U_tp.shape[1])]

        # Principal angles: arccos of singular values
        _, S, _ = np.linalg.svd(U_t_k.T @ U_tp_k, full_matrices=False)
        largest_sv = S[0] if len(S) > 0 else 1.0
        largest_angle = np.arccos(np.clip(largest_sv, -1, 1))

        theta_timeseries.append(largest_angle)

    return np.array(theta_timeseries)


def analyze_subspace_stability(theta_timeseries: np.ndarray) -> SubspaceStability:
    """
    Interpret subspace angle timeseries.

    PRIMARY METRIC: Indicates whether low-rank geometry is stable.
    """

    mean_angle = float(np.mean(theta_timeseries))
    median_angle = float(np.median(theta_timeseries))
    max_angle = float(np.max(theta_timeseries))

    stability = "stable" if median_angle < 0.6 else "unstable"

    return SubspaceStability(
        mean_angle_rad=mean_angle,
        median_angle_rad=median_angle,
        max_angle_rad=max_angle,
        stability=stability
    )


# ============================================================================
# PHASE 5: TRANSITION ALIGNMENT WITH RANDOM BASELINE
# ============================================================================

def transition_alignment_with_baseline(discovered_transitions: np.ndarray,
                                       manual_labels: np.ndarray,
                                       frame_tolerance: int = 30,
                                       n_bootstrap: int = 1000) -> TransitionAlignment:
    """
    F1 score vs. random baseline.

    PRIMARY METRIC: Semantic signal strength.

    ΔF1 > 0.25 → strong signal
    ΔF1 > 0.10 → weak signal
    ΔF1 < 0.10 → no signal
    """

    def compute_f1(discovered, manual, tolerance):
        """F1 between discovered and manual transitions."""
        if len(discovered) == 0 or len(manual) == 0:
            return 0.0

        precision = 0.0
        for t_d in discovered:
            if np.any(np.abs(manual - t_d) < tolerance):
                precision += 1
        precision /= len(discovered)

        recall = 0.0
        for t_m in manual:
            if np.any(np.abs(discovered - t_m) < tolerance):
                recall += 1
        recall /= len(manual)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1

    # Extract manual transition indices
    manual_transitions = np.where(np.diff(manual_labels) != 0)[0]

    # Observed F1
    f1_observed = compute_f1(discovered_transitions, manual_transitions, frame_tolerance)

    # Random baseline distribution
    f1_random_distribution = []
    for _ in range(n_bootstrap):
        random_transitions = np.sort(
            np.random.choice(len(manual_labels), size=len(discovered_transitions), replace=False)
        )
        f1_random = compute_f1(random_transitions, manual_transitions, frame_tolerance)
        f1_random_distribution.append(f1_random)

    f1_random_mean = np.mean(f1_random_distribution)
    f1_random_std = np.std(f1_random_distribution)
    delta_f1 = f1_observed - f1_random_mean

    # Signal strength classification
    if delta_f1 > 0.25:
        signal = "strong"
    elif delta_f1 > 0.10:
        signal = "weak"
    else:
        signal = "none"

    return TransitionAlignment(
        f1_observed=f1_observed,
        f1_random_mean=f1_random_mean,
        f1_random_std=f1_random_std,
        delta_f1=delta_f1,
        signal_strength=signal
    )


# ============================================================================
# UTILITY: MEAN CENTERING
# ============================================================================

def preprocess_telemetry(telemetry_raw: np.ndarray,
                        window_size: int = 64) -> np.ndarray:
    """
    Mean-center telemetry using sliding-window approach.

    CRITICAL: Must be applied before any spectral analysis.

    Input:
    - telemetry_raw: N_samples × 7 (raw, un-centered)
    - window_size: frames for sliding mean

    Returns:
    - telemetry_centered: N_samples × 7 (mean-centered)
    """

    N = len(telemetry_raw)
    centered = np.zeros_like(telemetry_raw)

    for t in range(N):
        start = max(0, t - window_size // 2)
        end = min(N, t + window_size // 2)
        mu_t = np.mean(telemetry_raw[start:end], axis=0)
        centered[t] = telemetry_raw[t] - mu_t

    return centered


# ============================================================================
# UTILITY: CSV EXPORT FOR DECISION GATE
# ============================================================================

def export_analysis_summary(game_name: str,
                           spectral: SpectralProfile,
                           persistence: RegimePersistence,
                           stability: SubspaceStability,
                           alignment: TransitionAlignment,
                           convergence_rate: float) -> pd.DataFrame:
    """
    Compile all metrics into single row for decision gate.
    """

    return pd.DataFrame({
        'game': [game_name],
        'r_eff': [spectral.r_eff],
        'entropy': [spectral.entropy],
        'r_90': [spectral.r_90],
        'decay_ratio': [spectral.decay_ratio],
        'profile': [spectral.profile],
        'median_stable_frames': [persistence.median_stable_frames],
        'pct_time_stable': [persistence.pct_time_stable],
        'num_regimes': [persistence.num_regimes],
        'subspace_angle_median': [stability.median_angle_rad],
        'subspace_stability': [stability.stability],
        'f1_observed': [alignment.f1_observed],
        'f1_random_mean': [alignment.f1_random_mean],
        'delta_f1': [alignment.delta_f1],
        'signal_strength': [alignment.signal_strength],
        'convergence_rate': [convergence_rate]
    })


# ============================================================================
# DECISION GATE: H0 vs H1 EVALUATION
# ============================================================================

def evaluate_hypothesis(row: pd.Series) -> Dict[str, any]:
    """
    Binary hypothesis test.

    Evaluates: Does row meet H1 (structure exists)?

    H1 criterion: ≥ 4 of 5 primary metrics pass thresholds.
    """

    criteria = {
        'r_eff_pass': row['r_eff'] <= 14,
        'entropy_pass': row['entropy'] <= np.log(12),
        'persistence_pass': row['pct_time_stable'] >= 60,
        'subspace_stable': row['subspace_angle_median'] < 0.6,
        'signal_strong': row['delta_f1'] > 0.20
    }

    passing_count = sum(criteria.values())
    h1_accepted = passing_count >= 4

    return {
        'criteria_passing': criteria,
        'passing_count': passing_count,
        'h1_accepted': h1_accepted,
        'confidence': passing_count / 5.0
    }
