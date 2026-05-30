# Q64 Empirical Validation Protocol v2.0
## Refined Systems-Performance Characterization

**Status:** Ready for deployment  
**Scope:** Determine whether game telemetry manifolds exhibit persistent low-rank structure sufficient for runtime supervision  
**Duration:** 5 weeks  
**Outcome:** Binary decision gate (H₀ vs H₁) with operational failure modes

---

## **Fundamental Question**

Not: *Can we build Q64?*

**But:** *Is runtime telemetry structured enough to justify Q64?*

This is empirical science, not engineering.

---

## **Operationalized Hypotheses**

### **H₀ (Null: No Meaningful Structure)**
System rejects H₀ if ALL of the following hold:
- Effective rank r_eff > 18 (high dimensionality)
- Spectral entropy H > log(12) (diffuse spectrum)
- < 40% of session time in stable regimes (mostly transient)
- Transition F1 vs. random baseline: ΔF1 < 0.15 (no semantic alignment)
- Subspace angle θ_t > 0.8 rad between consecutive epochs (geometry instability)

### **H₁ (Alternative: Structure Exists)**
System accepts H₁ if MOST of the following hold:
- Effective rank r_eff ≤ 14 (moderate low-rank)
- Spectral entropy H ≤ log(12) (concentrated spectrum)
- ≥ 60% of session time in stable regimes (coherent states)
- Transition F1: ΔF1 > 0.25 vs. random baseline (semantic signal)
- Subspace angle θ_t ≤ 0.6 rad (stable geometry)

**Decision rule:** If ≥ 4 of 5 criteria met → accept H₁ → Q64 viable → proceed to v1.0.

---

## **Primary Metrics (Ranked by Importance)**

| Rank | Metric | Why | Collection Cost | Sensitivity |
|------|--------|-----|-----------------|-------------|
| 1 | **Regime Persistence** | Separates meaningful states from noise | Automatic | Very high |
| 2 | **Effective Rank (r_eff)** | Honest rank capture via entropy | Automatic | High |
| 3 | **Spectral Entropy (H)** | Reveals concentration vs. diffuse | Automatic | High |
| 4 | **Subspace Angle Stability (θ_t)** | Detects geometry rotation under fixed rank | Automatic | Medium-high |
| 5 | **Transition Alignment (F1 vs. random)** | Semantic validation | Manual + automatic | Medium |
| 6 | **Convergence Rate** | Q64 applicability | Automatic | Medium |
| 7 | **Cross-Game Drift (L_drift)** | Portability feasibility | Automatic | Low (accepted at 0.5–0.9) |
| 8 | **Decay Ratio (λ₁/λₖ)** | Spectral concentration | Automatic | Low |

Secondary metrics (r₉₀, r₉₅) reported but not decision-critical.

---

## **Data Collection Specification**

### **Telemetry Vector (7 dimensions)**

```
Column  Name              Unit      Range       Notes
------  ----              ----      -----       -----
1       frame_time        ms        10–50       GPU pipeline latency
2       gpu_load          [0–1]     0.0–1.0     GPU utilization
3       cpu_load          [0–1]     0.0–1.0     CPU utilization
4       soc_temp          °C        30–80       System-on-Chip temperature
5       input_lag         ms        0–10        Input→display latency
6       power_draw        W         5–35        Total APU power consumption
7       frame_counter     count     0–∞         Monotonic frame ID
```

### **Manual Annotation (1 dimension)**

```
Column  Name              Values
------  ----              ------
8       scene_label       menu, gameplay, loading, cutscene, thermal_throttle, etc.
```

### **Sampling Protocol**

- **Rate:** 60 FPS (16.67 ms intervals)
- **Duration:** 30 minutes per game
- **Total samples:** 108,000 per game
- **Games:** 5 (esports, AAA open-world, narrative, emulation, indie)
- **Total data:** 540,000 samples

### **Preprocessing**

1. **Mean-centering** (CRITICAL):
   ```
   μ_t = mean(s[t-63:t], axis=0)  # 64-frame sliding window mean
   s̃_t = s_t - μ_t               # Center telemetry
   ```
   Without centering: first PC encodes only baseline operating point.

2. **No outlier removal** (preserve regime transitions)

3. **Preserve temporal order** (statistics depend on sequence)

---

## **Analysis Pipeline: Corrected & Refined**

### **Phase 1: Spectral Characterization**

```python
def analyze_spectral_structure(telemetry_centered, scene_labels):
    """
    Comprehensive manifold structure analysis.
    
    Returns:
    - Effective rank (primary metric)
    - Spectral entropy (primary metric)
    - Eigenvalue decay curve
    - Profile classification
    """
    
    # Full-session covariance (already mean-centered)
    s = telemetry_centered
    G = (s.T @ s) / len(s)
    Lambda = np.linalg.eigvalsh(G)
    Lambda = np.flip(Lambda)  # Descending order
    
    # Cumulative variance (secondary)
    cum_var = np.cumsum(Lambda) / np.sum(Lambda)
    r_90 = np.searchsorted(cum_var, 0.90) + 1
    r_95 = np.searchsorted(cum_var, 0.95) + 1
    
    # Effective rank via entropy (PRIMARY)
    p = Lambda / np.sum(Lambda)
    p_safe = p[p > 1e-10]
    H = -np.sum(p_safe * np.log(p_safe))
    r_eff = np.exp(H)
    
    # Spectral decay ratio
    decay_ratio = Lambda[0] / (Lambda[-1] + 1e-10)
    
    # Profile classification
    if r_eff < 0.8 * r_90:
        profile = "smooth"  # Concentrated
    elif r_eff > 1.2 * r_90:
        profile = "spiky"   # Long tail
    else:
        profile = "mixed"
    
    return {
        'eigenvalues': Lambda,
        'r_eff': r_eff,
        'entropy': H,
        'r_90': r_90,
        'r_95': r_95,
        'decay_ratio': decay_ratio,
        'profile': profile
    }
```

---

### **Phase 2: Regime Discovery & Persistence Analysis**

```python
def discover_regimes_unsupervised(telemetry_centered, window=64, tau_rank=0.2):
    """
    Discover rank-transition boundaries automatically.
    Does NOT use manual scene labels for transition detection.
    """
    
    ranks = []
    G_histories = []
    
    for t in range(len(telemetry_centered) - window):
        s_window = telemetry_centered[t:t+window]
        G = (s_window.T @ s_window) / window
        Lambda = np.linalg.eigvalsh(G)
        Lambda = np.flip(Lambda)
        
        # Rank at this τ
        rank = np.sum(Lambda > tau_rank * Lambda[0])
        ranks.append(rank)
        G_histories.append(G)
    
    # Detect transitions: rank discontinuities
    rank_array = np.array(ranks)
    rank_deltas = np.abs(np.diff(rank_array))
    transitions = np.where(rank_deltas >= 2)[0]
    
    return transitions, rank_array, G_histories

def compute_regime_persistence(transitions, total_frames):
    """
    Characterize regime lifetimes.
    
    PRIMARY METRIC: Ratio of stable vs. transient regimes.
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
    
    # Classification: stable > 100 frames, transient <= 100 frames
    stable_durations = durations[durations > 100]
    transient_durations = durations[durations <= 100]
    
    return {
        'median_stable_frames': float(np.median(stable_durations)) if len(stable_durations) > 0 else np.nan,
        'median_transient_frames': float(np.median(transient_durations)) if len(transient_durations) > 0 else np.nan,
        'pct_time_stable': 100.0 * np.sum(stable_durations) / total_frames,
        'pct_time_transient': 100.0 * np.sum(transient_durations) / total_frames,
        'num_regimes': len(regime_durations),
        'transition_count': len(transitions)
    }
```

---

### **Phase 3: Subspace Angle Stability (NEW)**

```python
def subspace_angle_stability(G_histories, k=8, window_step=10):
    """
    Compute principal angles between adjacent subspaces.
    
    Detects: rank may be stable, but geometry rotates rapidly.
    That would indicate low predictive utility despite low apparent rank.
    """
    
    theta_timeseries = []
    
    for t in range(len(G_histories) - window_step):
        G_t = G_histories[t]
        G_t_plus = G_histories[t + window_step]
        
        # Top-k eigenvectors
        Lambda_t, U_t = np.linalg.eigh(G_t)
        Lambda_t = np.flip(Lambda_t)
        U_t = np.flip(U_t, axis=1)
        
        Lambda_tp, U_tp = np.linalg.eigh(G_t_plus)
        Lambda_tp = np.flip(Lambda_tp)
        U_tp = np.flip(U_tp, axis=1)
        
        U_t_k = U_t[:, :k]
        U_tp_k = U_tp[:, :k]
        
        # Principal angles (largest singular value of U_t_k.T @ U_tp_k)
        _, S, _ = np.linalg.svd(U_t_k.T @ U_tp_k)
        largest_angle = np.arccos(np.clip(S[0], -1, 1))
        
        theta_timeseries.append(largest_angle)
    
    theta_array = np.array(theta_timeseries)
    
    return {
        'mean_angle_rad': float(np.mean(theta_array)),
        'median_angle_rad': float(np.median(theta_array)),
        'max_angle_rad': float(np.max(theta_array)),
        'stability': 'stable' if np.median(theta_array) < 0.6 else 'unstable'
    }
```

---

### **Phase 4: Transition Alignment with Random Baseline**

```python
def transition_alignment_with_baseline(discovered_transitions, manual_labels, 
                                       frame_tolerance=30, n_bootstrap=1000):
    """
    F1 score for discovered vs. manual transitions.
    Compare to random baseline for context.
    
    ΔF1 = F1_observed - E[F1_random]
    """
    
    def compute_f1(discovered, manual, tolerance):
        """F1 between two transition sets."""
        precision = 0.0
        if len(discovered) > 0:
            for t_d in discovered:
                if np.any(np.abs(manual - t_d) < tolerance):
                    precision += 1
            precision /= len(discovered)
        
        recall = 0.0
        if len(manual) > 0:
            for t_m in manual:
                if np.any(np.abs(discovered - t_m) < tolerance):
                    recall += 1
            recall /= len(manual)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1
    
    # Actual F1
    manual_transitions = np.where(np.diff(manual_labels) != 0)[0]
    f1_observed = compute_f1(discovered_transitions, manual_transitions, frame_tolerance)
    
    # Random baseline: draw equal-sized transition sets randomly
    f1_random_distribution = []
    for _ in range(n_bootstrap):
        random_transitions = np.sort(np.random.choice(
            len(manual_labels), size=len(discovered_transitions), replace=False
        ))
        f1_random = compute_f1(random_transitions, manual_transitions, frame_tolerance)
        f1_random_distribution.append(f1_random)
    
    f1_random_mean = np.mean(f1_random_distribution)
    f1_random_std = np.std(f1_random_distribution)
    delta_f1 = f1_observed - f1_random_mean
    
    return {
        'f1_observed': f1_observed,
        'f1_random_mean': f1_random_mean,
        'f1_random_std': f1_random_std,
        'delta_f1': delta_f1,
        'signal_strength': 'strong' if delta_f1 > 0.25 else ('weak' if delta_f1 > 0.10 else 'none')
    }
```

---

### **Phase 5: Convergence Validation**

Run stream-oriented Q64 with corrected Gram updates and incremental eigentracking.

```python
class StreamOrientedQ64Corrected:
    def __init__(self, k=16, window=64, tau=0.2):
        self.k = k
        self.window = window
        self.tau = tau
        self.ring = RingBuffer(window, 7)  # 7-dim telemetry
        self.G = np.zeros((7, 7))
        self.U_k = None
        self.Lambda_k = None
        self.rank = 0
        self.mu = np.zeros(7)
        self.history = []
    
    def calibrate(self, S_calib):
        """Offline calibration on first 500 frames."""
        self.mu = np.mean(S_calib, axis=0)
        S_centered = S_calib - self.mu
        self.G = (S_centered.T @ S_centered) / len(S_calib)
        
        Lambda, U = np.linalg.eigh(self.G)
        Lambda = np.flip(Lambda)
        U = np.flip(U, axis=1)
        
        self.U_k = U[:, :self.k]
        self.Lambda_k = Lambda[:self.k]
        self.rank = np.sum(Lambda > self.tau * Lambda[0])
        
        self.phi_ref = (self.G.copy(), self.rank, self.tau)
    
    def step(self, s_t):
        """Per-frame update with sliding-window Gram."""
        
        # 1. Mean-center using sliding window
        self.ring.append(s_t)
        if len(self.ring) >= self.window:
            mu_t = np.mean(self.ring.buffer, axis=0)
        else:
            mu_t = np.mean(self.ring.buffer[:len(self.ring)], axis=0)
        
        s_centered = s_t - mu_t
        
        # 2. Sliding-window Gram update (CRITICAL)
        self.G += np.outer(s_centered, s_centered)
        
        if len(self.ring) == self.window:
            # Remove oldest (now at front after rotation)
            s_old = self.ring.buffer[0] - mu_t  # Approximate
            self.G -= np.outer(s_old, s_old)
        
        # 3. Incremental eigentracking (Rayleigh-Ritz)
        if self.U_k is not None:
            H_proj = self.U_k.T @ self.G @ self.U_k
            Lambda_proj, V = np.linalg.eigh(H_proj)
            Lambda_proj = np.flip(Lambda_proj)
            V = np.flip(V, axis=1)
            
            self.U_k = self.U_k @ V
            self.Lambda_k = Lambda_proj[:self.k]
        
        # 4. Rank + convergence
        rank_new = np.sum(self.Lambda_k > self.tau * self.Lambda_k[0])
        
        # 5. τ correction with hysteresis
        if abs(rank_new - self.rank) > 2:
            if rank_new > self.k + 1:
                self.tau = min(0.5, self.tau * 0.9)
            elif rank_new < self.k - 1:
                self.tau = max(0.05, self.tau * 1.1)
        
        self.rank = rank_new
        
        # 6. Convergence tests (3 criteria)
        P_theta = self.U_k @ np.diag(self.Lambda_k) @ self.U_k.T
        R_t = np.linalg.norm(self.G - P_theta @ self.G, 'fro')
        
        L_t = (np.linalg.norm(self.G - self.phi_ref[0], 'fro') / 
               (np.linalg.norm(self.phi_ref[0], 'fro') + 1e-8))
        
        converged = (
            R_t < 1e-3 and
            self.rank == self.phi_ref[1] and
            abs(L_t - (self.history[-1]['L'] if self.history else L_t)) < 0.05 * max(L_t, 1e-8)
        )
        
        self.history.append({
            'rank': self.rank,
            'L': L_t,
            'R': R_t,
            'converged': converged,
            'tau': self.tau
        })
        
        return {'converged': converged, 'L': L_t, 'rank': self.rank}
```

---

## **Revised Success Criteria**

| Metric | Primary | Target | Acceptable | Failure | Decision Weight |
|--------|---------|--------|-----------|---------|-----------------|
| **r_eff** | Yes | ≤10 | ≤14 | >18 | Required for H₁ |
| **Entropy H** | Yes | ≤log(12) | ≤log(14) | >log(16) | Required for H₁ |
| **Stable regime %** | Yes | >70% | >60% | <40% | CRITICAL |
| **Stable regime median (frames)** | Yes | >300 | >150 | <50 | CRITICAL |
| **Subspace angle θ** | Yes | <0.5 rad | <0.65 rad | >0.8 rad | Required for H₁ |
| **Transition F1 (ΔF1 vs. random)** | Yes | >0.30 | >0.20 | <0.10 | Required for H₁ |
| **Convergence rate** | No | >92% | >85% | <70% | Supporting |
| **Decay ratio λ₁/λₖ** | No | >40 | >20 | <5 | Supporting |
| **Cross-game L_drift** | No | <0.5 | <0.8 | >1.2 | Supporting only |

**Decision Rule:** Accept H₁ (Q64 viable) if ≥ 4 of 5 primary criteria met within game.

---

## **Failure Mode Analysis**

### **Most Likely Failure Mode: Regime Fragmentation**

**Signature:**
- r_eff appears favorable (≤14)
- But median regime duration < 80 frames
- Pct stable < 50%

**Implication:** Manifold may be low-rank locally, but fragments rapidly. Convergence becomes noise-chasing.

**Consequence:** Q64 reverts to anomaly detector (detect rapid regime changes as anomalies).

---

### **Secondary Failure Mode: Geometry Instability**

**Signature:**
- r_eff favorable
- Regime persistence acceptable
- But θ_t > 0.7 rad (rapid subspace rotation)

**Implication:** Rank stable, but principal directions rotate constantly. Predictive utility collapses.

**Consequence:** Requires per-frame re-calibration; loses efficiency gains.

---

### **Tertiary Failure Mode: Semantic Misalignment**

**Signature:**
- Spectral structure favorable
- Persistence favorable
- But ΔF1 < 0.10 (transitions don't align with gameplay)

**Implication:** System detects *something*, but not operationally meaningful.

**Consequence:** Q64 becomes noise monitor; not useful for supervision.

---

## **Expected Outcome (Prior Distribution)**

Based on handheld workload structure:

| Scenario | Probability | r_eff | Stable % | θ | ΔF1 | Implication |
|----------|-------------|-------|----------|---|-----|-------------|
| **Moderate structure** | 60–70% | 8–14 | 60–75% | 0.4–0.6 | 0.25–0.50 | Q64 viable, per-game calibration |
| **Strong structure** | 10–15% | 6–9 | >75% | <0.4 | >0.50 | Q64 viable, universal Φ_ref possible |
| **Fragmentation** | 15–20% | 10–16 | 30–50% | 0.5–0.8 | 0.1–0.25 | Redesign to anomaly detection |
| **Chaos** | 5–10% | >18 | <30% | >0.8 | <0.10 | System not viable |

Most likely outcome: **Moderate structure** → proceed to v1.0 with per-game calibration.

---

## **Implementation Checklist**

```
PHASE 1: DATA COLLECTION (2 weeks)
  [ ] Deploy Ally X telemetry instrumentation (7 dimensions)
  [ ] Collect 30 min per game, 5 games, 60 FPS
  [ ] Manual scene labeling (per gameplay session)
  [ ] Export as CSV: timestamp, frame_time, gpu_load, cpu_load, temp, input_lag, power, frame_counter, scene_label
  [ ] Verify: no NaN, no gaps, monotonic timestamps

PHASE 2: SPECTRAL ANALYSIS (1 week)
  [ ] Load telemetry; mean-center per 64-frame window
  [ ] Compute full-session G = (1/N)S^T S (centered)
  [ ] Eigendecompose: get λ₁ ≥ λ₂ ≥ ... ≥ λ₇
  [ ] Compute r_eff (entropy-based) — PRIMARY
  [ ] Compute r_90, r_95 — SECONDARY
  [ ] Compute H (spectral entropy) — PRIMARY
  [ ] Compute decay_ratio = λ₁/λ₇ — SUPPORTING
  [ ] Output: spectral_report_per_game.csv

PHASE 3: REGIME DISCOVERY & PERSISTENCE (1 week)
  [ ] Unsupervised rank-transition detection (Δrank ≥ 2)
  [ ] Compute regime durations
  [ ] Stratify: stable (>100 fr) vs. transient (≤100 fr)
  [ ] Compute median_stable_frames, pct_time_stable — PRIMARY
  [ ] Output: persistence_report_per_game.csv

PHASE 4: SUBSPACE STABILITY (3 days)
  [ ] Compute principal angles θ_t between consecutive subspaces
  [ ] Report mean, median, max angles
  [ ] Classify: stable (median < 0.6 rad) vs. unstable — PRIMARY
  [ ] Output: subspace_stability_report.csv

PHASE 5: TRANSITION ALIGNMENT (3 days)
  [ ] Compute F1(discovered vs. manual) — OBSERVED
  [ ] Generate 1000 random transition sets; compute F1 distribution — BASELINE
  [ ] Compute ΔF1 = F1_obs - E[F1_random] — PRIMARY
  [ ] Output: alignment_report.csv

PHASE 6: CONVERGENCE VALIDATION (1 week)
  [ ] Implement StreamOrientedQ64Corrected (with sliding-window Gram)
  [ ] Calibrate on frames 0–500
  [ ] Run on frames 501–107,999
  [ ] Measure convergence_rate (%), mean_ttc (frames)
  [ ] Output: convergence_report_per_game.csv

PHASE 7: PORTABILITY TESTING (3 days)
  [ ] For each game pair (A, B):
      - Calibrate Φ_ref on A
      - Run Q64 on B with Φ_ref_A
      - Measure L_drift
  [ ] Create 5×5 portability matrix
  [ ] Output: portability_matrix.csv

SYNTHESIS: DECISION GATE (3 days)
  [ ] Compile all metrics into master table
  [ ] Evaluate against H₀ vs H₁ criteria
  [ ] Count passing metrics (target: ≥4 of 5 primary)
  [ ] Generate empirical_verdict_report.md
  [ ] Decision: H₀ → redesign; H₁ → v1.0 hardening
```

---

## **What Success Looks Like**

**The question is not:** "Is Q64 perfect?"

**The question is:** "Do game telemetry manifolds exhibit persistent, low-dimensional regime structure with operational meaning?"

**Success:** Yes, modulo:
- Per-game calibration required
- Regime persistence > 150 frames median
- Spectral entropy bounded
- Subspace geometry stable enough

When it holds, Q64 transitions from speculative to empirically grounded.
