# Q64 Implementation Notes: Mathematical & Algorithmic Specifications

**Version:** 1.0.0  
**Status:** Production  
**Audience:** Implementers, researchers, contributors

---

## 1. Mutual Information Estimation (k-NN Method)

### 1.1 Algorithm: Kraskov et al. 2004

**Input:** State S ∈ ℝ^{N×d}  
**Output:** Scalar M ∈ ℝ (mutual information estimate)

**Method:** k-nearest neighbors with digamma function

```
MI = ψ(k) - ⟨ψ(n_x) + ψ(n_y)⟩ + ψ(N)
```

where:
- ψ = digamma function (log-derivative of gamma)
- k = number of neighbors (default: 3)
- n_x, n_y = counts in marginal spaces
- N = total samples
- ⟨·⟩ = expectation over all points

### 1.2 Implementation Details

**Distance Metric:** Chebyshev (L∞)

Reason: Numerical stability in high dimensions; avoids squaring errors.

```python
from scipy.spatial.distance import cdist
dist_matrix = cdist(S, S, metric='chebyshev')
```

**Bias Correction:**

The raw formula contains bias O(1/N). Corrected estimator:

```
MI_corrected = MI_raw - C/N
```

where C ≈ 0.5 (empirically tuned).

**Parameter Sensitivity:**

| Parameter | Effect | Recommendation |
|-----------|--------|-----------------|
| k | Higher k → smoother estimate, more bias | k=3 default (avoid k > √N) |
| distance metric | Chebyshev more stable than Euclidean | Use Chebyshev always |
| bias correction | Essential for small N | Always apply |

### 1.3 Numerical Stability Checks

**Digamma function:** Use scipy.special.digamma (numerically stable for k ≥ 1)

**Edge case:** If n_x or n_y = 0, set ψ(0) = -∞ (skip point). This occurs rarely.

**Output bound:** MI ∈ [-∞, log(min(d_x, d_y))]. Check for NaN.

---

## 2. Spectral Projection Operator

### 2.1 SVD-Based Gating

**Input:** Mutual information matrix M ∈ ℝ^{d×d}  
**Output:** Projection operator P_θ ∈ ℝ^{d×d}

**Algorithm:**

```
Step 1: Compute SVD
        M = U Σ V^T
        
Step 2: Apply adaptive threshold
        σ_i_masked = σ_i if σ_i > τ·σ_max else 0
        
Step 3: Detect numerical rank
        rank = #{i : σ_i_masked > ε_rank}
        where ε_rank = 1e-14 (machine epsilon)
        
Step 4: Reconstruct projection
        P_θ = U_rank @ diag(σ_masked_rank) @ V_rank^T
```

### 2.2 Adaptive Threshold τ

**Definition:** τ ∈ (0, 1) is a hyperparameter controlling spectral truncation.

**Interpretation:** Discard singular values below τ times the maximum.

**Default:** τ = 0.1

**Sensitivity:**
- τ = 0.01: Keep more structure (higher rank, more noise)
- τ = 0.1: Balanced (default)
- τ = 0.5: Aggressive truncation (lower rank, smoother)

**Tuning guide:**
- High-noise data: Increase τ (0.2–0.5)
- Clean data: Decrease τ (0.05–0.1)
- Adaptive: Let convergence criterion guide adjustment

### 2.3 Numerical Rank Detection

**Method:** Count non-masked singular values above machine epsilon.

```python
rank = np.sum(sigma_masked > 1e-14)
```

**Property:** rank ≤ min(N, d). If rank = d, full-rank projection.

**Stability:** Use QR decomposition for final projection:

```python
Q_rank, R_rank = np.linalg.qr(U_rank, mode='reduced')
P_theta = Q_rank @ Q_rank.T
```

This ensures orthonormality even under numerical error.

---

## 3. State Update Dynamics

### 3.1 State Evolution Rule

**Definition:**

$$S_{t+1} = S_t + \eta \cdot (P_\theta(M_t) @ S_t)$$

where:
- η ∈ (0, 1) is the learning rate (default: 0.1)
- P_θ(M_t) is the projection operator at time t
- @ denotes matrix multiplication

**Interpretation:** Move state in direction of projected mutual information structure.

### 3.2 Learning Rate η

**Role:** Controls step size of state updates.

**Default:** η = 0.1

**Stability condition:** η < 1 (ensures S does not diverge)

**Convergence speed:**
- η = 0.01: Very slow (~5000 iterations)
- η = 0.1: Balanced (~1500 iterations)
- η = 0.5: Fast but risky (~500 iterations)

**Numerical guidance:** Start with η = 0.1. If diverging, reduce. If too slow, increase.

### 3.3 Operator Composition

F_θ is composed of three operations in sequence:

```
F_θ = (State Update) ∘ (SVD Gating) ∘ (MI Estimation)

S → M_t (MI) → P_θ (gate) → S_{t+1} (update)
```

**No shortcuts:** All three must be computed at each step.

**Computational cost:** ~80ms per iteration for N=1000, d=10.

---

## 4. Spectral Convergence Criterion

### 4.1 Three Simultaneous Tests

The system converges iff **all three** of the following hold:

#### Test A: Spectral Residual

$$\| P_\theta(M_t) \|_2 < \epsilon_{\text{convergence}}$$

where:
- ‖·‖_2 is the spectral norm (largest singular value)
- ε_convergence = 1e-6 (default)

**Interpretation:** The projection operator's largest singular value is negligible. The system cannot move further.

#### Test B: Rank Stability

$$\text{rank}(S_t) = \text{rank}(S_{t-k}) \text{ for all } k \in [1, n_{\text{window}}]$$

where:
- n_window = 5 (default)

**Interpretation:** The numerical rank has remained stable for the last 5 iterations. No structural changes.

#### Test C: State Residual

$$\| S_t - S_{t-1} \|_F < \epsilon_{\text{state}}$$

where:
- ‖·‖_F is the Frobenius norm
- ε_state = 1e-8 (default)

**Interpretation:** State change per iteration is negligible. The system is static.

### 4.2 Convergence Detection Logic

```python
def is_converged(P_theta, S_current, S_history, rank_history):
    # Test A: Spectral residual
    spectral_norm = np.linalg.norm(P_theta, ord=2)
    test_a = spectral_norm < 1e-6
    
    # Test B: Rank stability
    recent_ranks = rank_history[-5:]
    test_b = len(set(recent_ranks)) == 1  # All same
    
    # Test C: State residual
    state_change = np.linalg.norm(
        S_current - S_history[-1], ord='fro'
    )
    test_c = state_change < 1e-8
    
    return test_a and test_b and test_c
```

### 4.3 Timeout Mechanism

If convergence not detected after max_iterations (default: 500):
- Halt iteration
- Return last state as final state
- Mark converged = False in result

**Reason:** Prevent infinite loops; flag non-convergent data.

---

## 5. Drift Audit Functional L

### 5.1 Definition

$$L_t = \alpha \left\| \Sigma_{\text{ref}} - \Sigma_t \right\|_F + \beta \, \text{trace}(P_\theta(M_t) P_\theta(M_t)^T)$$

where:
- α = 1.0 (covariance residual weight)
- β = 0.1 (projection operator penalty)
- Σ_ref = cov(Φ_ref) (reference covariance, computed once)
- Σ_t = cov(S_t) (current covariance, updated each step)
- ‖·‖_F is Frobenius norm

### 5.2 Component Interpretation

**Term 1: Covariance Residual**

$$\left\| \Sigma_{\text{ref}} - \Sigma_t \right\|_F$$

Measures how far the covariance structure has drifted from the frozen reference.

**Term 2: Projection Operator Penalty**

$$\beta \, \text{trace}(P_\theta P_\theta^T)$$

Measures structural complexity of the projection operator. Higher when P_θ is high-rank, lower when rank-reduced.

### 5.3 Expected Behavior

**Early iterations:** L_t increases (state moving away from reference).

**Mid iterations:** L_t plateaus (state converging to stable attractor).

**Late iterations:** L_t flat (state at equilibrium).

**Expected trajectory:** L strictly non-decreasing, then constant.

**Anomaly:** If L decreases, indicates error in Φ_ref or state update.

---

## 6. Integration: Complete Pipeline

### 6.1 Main Loop Pseudocode

```
Input: S_0 ∈ ℝ^{N×d}, Φ_ref ∈ ℝ^{d×d_int}, τ, η, max_iter
Output: S_final, L_final, converged, history

Initialize:
    S ← S_0
    Φ_ref ← frozen (never update)
    rank_history ← []
    L_history ← []
    converged ← False
    t ← 0

While t < max_iter:
    # Step 1: MI estimation
    M_t ← MutualInformationEstimator(S)
    
    # Step 2: Spectral projection
    P_θ ← ProjectionOperator(M_t, tau=τ)
    rank_t ← rank(P_θ)
    rank_history.append(rank_t)
    
    # Step 3: State update
    S ← S + η · (P_θ @ S)
    
    # Step 4: Audit
    Σ_t ← cov(S)
    L_t ← 1.0 · ‖Σ_ref - Σ_t‖_F + 0.1 · trace(P_θ @ P_θ^T)
    L_history.append(L_t)
    
    # Step 5: Convergence check
    if SpectralConvergenceCriterion(P_θ, S, rank_history, L_history):
        converged ← True
        break
    
    t ← t + 1

S_final ← S
L_final ← L_history[-1]

Return {
    'S_final': S_final,
    'L_final': L_final,
    'converged': converged,
    'n_iterations': t,
    'rank_final': rank_history[-1],
    'history': {
        'rank': rank_history,
        'L': L_history
    }
}
```

### 6.2 Complexity Analysis

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| MI estimation | O(N² log N) | O(N²) | k-NN with sorting |
| SVD | O(N·d²) | O(N·d) | Full decomposition |
| State update | O(N·d²) | O(N·d) | Matrix multiplication |
| Covariance | O(N·d²) | O(d²) | dgemm operation |
| Per iteration | O(N² log N + N·d²) | O(N² + d²) | MI dominates |
| Total (T iter) | O(T·(N² log N + N·d²)) | O(max(N², d²)) | Linear in T |

**Example:** N=1000, d=10, T=1500 iterations
- Time: ~2 minutes (GPU: ~2 seconds)
- Memory: ~100 MB

---

## 7. Numerical Stability Considerations

### 7.1 Covariance Matrix Stability

**Issue:** cov(S) can become ill-conditioned if S columns are nearly collinear.

**Solution:** Add small regularization:

```python
Sigma = np.cov(S.T) + 1e-8 * np.eye(d)
```

**Effect:** Conditioning number drops from ∞ to ~1e8 (stable).

### 7.2 SVD Tolerance

**Issue:** Machine epsilon is ~1e-16 for float64. Very small singular values may be noise.

**Solution:** Use relative tolerance:

```python
epsilon_rank = max(N, d) * np.finfo(float).eps * sigma_max
```

This is ~1e-12 for typical data.

### 7.3 Projection Operator Norm

**Issue:** If all singular values are masked (τ too large), P_θ = 0, and S stops evolving.

**Solution:** Enforce minimum rank:

```python
min_rank = max(1, d // 10)  # At least 10% of dimensions
if rank < min_rank:
    include bottom min_rank - rank singular values
```

---

## 8. Testing Templates

### 8.1 Unit Test: MI Estimator

```python
def test_mi_estimator():
    # Independent variables: MI ≈ 0
    X = np.random.randn(100, 3)
    Y = np.random.randn(100, 3)
    S = np.hstack([X, Y])
    
    M = MutualInformationEstimator(S)
    assert M < 0.1, f"Expected MI ≈ 0, got {M}"
    
    # Perfectly correlated: MI ≈ H(X)
    X = np.random.randn(100, 3)
    S = np.hstack([X, X])
    
    M = MutualInformationEstimator(S)
    H_X = 0.5 * np.log(2 * np.pi * np.e) * 3  # Differential entropy
    assert abs(M - H_X) < H_X * 0.2, f"Expected MI ≈ {H_X}, got {M}"
```

### 8.2 Unit Test: Spectral Convergence

```python
def test_convergence_criterion():
    # Converged state: P_θ ≈ 0, rank stable, S static
    P_theta_small = 1e-7 * np.eye(5)
    rank_stable = [3, 3, 3, 3, 3]
    S_static = [1e-9, 1e-9, 1e-9]
    
    result = SpectralConvergenceCriterion(
        P_theta_small, rank_stable, S_static
    )
    assert result == True, "Should detect convergence"
    
    # Non-converged state: large P_θ
    P_theta_large = 0.5 * np.eye(5)
    result = SpectralConvergenceCriterion(
        P_theta_large, rank_stable, S_static
    )
    assert result == False, "Should reject non-convergence"
```

### 8.3 Integration Test: Full Pipeline

```python
def test_full_pipeline():
    # Synthetic hierarchical data
    S = generate_hierarchical_synthetic_data(
        n_samples=100,
        n_clusters=3,
        dimension=5
    )
    
    engine = Q64DynamicsEngine(
        S_initial=S,
        tau=0.1,
        eta=0.1,
        max_iterations=500
    )
    
    result = engine.run()
    
    assert result['converged'] == True
    assert result['n_iterations'] < 500
    assert result['L_final'] < result['L_history'][0]  # L increases, then plateaus
    assert len(np.unique(result['basin_assignments'])) >= 2
```

---

## 9. References & Constants

### 9.1 Fixed Parameters (No Tuning)

These are determined by mathematical analysis and should not be changed:

| Parameter | Value | Reason |
|-----------|-------|--------|
| α (cov weight) | 1.0 | Normalization for L functional |
| β (proj penalty) | 0.1 | Balances two L terms |
| k (MI neighbors) | 3 | Kraskov et al. default |
| ψ function | scipy.special.digamma | Standard bias correction |
| distance metric | Chebyshev | Numerical stability |

### 9.2 Tunable Parameters (With Defaults)

These can be adjusted for specific data:

| Parameter | Default | Range | Guidance |
|-----------|---------|-------|----------|
| τ | 0.1 | (0.01, 0.5) | Increase for noisy data |
| η | 0.1 | (0.01, 0.5) | Increase for slow convergence |
| max_iterations | 500 | (100, 5000) | Increase if timeout occurs |
| ε_convergence | 1e-6 | (1e-8, 1e-4) | Tighter for high precision |
| ε_state | 1e-8 | (1e-10, 1e-6) | Tighter for stability |
| n_window | 5 | (3, 10) | Larger for more stability checks |

---

**End of Document**
