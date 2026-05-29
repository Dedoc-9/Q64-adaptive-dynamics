# Hierarchical Primitives: The Critical Turn Across Scales

**Version:** 1.0.0  
**Status:** Framework Definition (v1.1.0 target)  
**Scope:** How the four-primitive structure extends to multi-scale hierarchy via Menger sponge reduction.

---

## Overview

The critical turn (documented in THE_CRITICAL_TURN.md) establishes four irreducible primitives at the base level:

- **S**: Observational state (ℝ^{N×d})
- **F_θ**: Representation dynamics operator
- **Φ_ref**: Frozen reference anchor (never updated)
- **L**: Drift audit functional

**Key principle for hierarchical extension:**

At each subsequent level k, the same four-primitive structure *replicates*, but with reduced dimensionality via deterministic fractal compression.

---

## Structure at Each Level

### State Reduction

Transition from level k to level k+1:

```
S_∞^(k) = converged state at level k
B^(k) = basin assignments at level k
                ↓
        EXTRACT-REPRESENTATIVES
                ↓
S_0^(k+1) = reduced state (shape: N_{k+1} << N_k, d_{k+1} < d_k)
```

Representatives are extracted as centroids of each basin, preserving cluster structure.

### Reference Reduction (Deterministic)

```
Φ_ref^(k) = frozen reference at level k
                ↓
        MENGER-COMPRESS (deterministic fractal pattern)
                ↓
Φ_ref^(k+1) = compressed reference (shape: d_{k+1} × d_int)
              [FROZEN at level k+1 initialization]
```

The Menger pattern ensures scale-invariance through deterministic (not random) downsampling.

### Dynamics Operator at Each Level

At level k:

```
F_θ^(k) = MI estimator ∘ SVD projection ∘ State update
```

Each level instantiates independently with level-specific parameters:
- MI: k=3 (fixed across all levels)
- SVD threshold: τ_k adaptive per level
- State update: η_k = 0.1 (default)

### Audit Functional at Each Level

```
L_t^(k) = α·‖Σ_ref^(k) - Σ_t^(k)‖_F + β·trace(P_θ^(k))²
```

Measures drift at level k relative to *local* frozen reference (not global base).

---

## Hierarchy Protocol

### Main Loop

```
k ← 0
S_0^(0) ← initial observations
Φ_ref^(0) ← frozen reference (computed once)

while k < max_levels and d_k > reduction_threshold:
    
    # Run Q64 at level k until convergence
    repeat:
        M_t^(k) ← MI(S_t^(k))
        P_θ^(k) ← SVD(M_t^(k), τ)
        S_{t+1}^(k) ← S_t^(k) + η·(P_θ @ S_t^(k))
        L_t^(k) ← compute_drift(M_t^(k), S_t^(k), Φ_ref^(k))
    until convergence
    
    # Extract structure
    B^(k) ← spectral_cluster(S_∞^(k))
    
    # Validate scale-invariance (audit only)
    M_cross^(k) ← check_scale_invariants(Φ_ref^(k), Φ_ref^(k-1))
    
    # Reduce for next level
    S_0^(k+1) ← extract_representatives(S_∞^(k), B^(k))
    Φ_ref^(k+1) ← menger_compress(Φ_ref^(k))  [FROZEN]
    d_{k+1} ← dimensionality(S_0^(k+1))
    
    k ← k + 1

return {
    levels: [k=0, 1, ..., k_max],
    basin_taxonomy: nested tree structure,
    drift_trajectory: [L_∞^(0), L_∞^(1), ...],
    scale_invariants: validation flags
}
```

---

## Proof of Irreducibility at Each Level

**Claim:** At each level k, the four primitives (S^(k), F_θ^(k), Φ_ref^(k), L^(k)) are mutually independent.

**Argument:**
1. S^(k) cannot be derived from (F_θ^(k), Φ_ref^(k), L^(k)) because initial state is input, not constructed
2. F_θ^(k) cannot be derived from (S^(k), Φ_ref^(k), L^(k)) because operator structure is defined, not computed
3. Φ_ref^(k) cannot be derived from (S^(k), F_θ^(k), L^(k)) because it is frozen and never updated
4. L^(k) cannot be derived from the other three alone because it requires explicit formula with fixed weights

**Consequence:** The hierarchy maintains irreducibility at each level without reducing to fewer global primitives.

---

## Scale-Invariance Validation

### Expected Scaling Law

Singular values should decrease with compression:

```
σ_i^(k+1) ≈ σ_i^(k) · compression_ratio
```

where compression_ratio ∈ (0.3, 0.7), typically 0.5.

### Detection of Anomalies

MultiScaleAnalyzer.detect_violations() checks:

1. **Spectral continuity:** Do singular values scale as expected?
2. **Basin coherence:** Do child basins stay within parent basins?
3. **Drift monotonicity:** Does L increase monotonically across levels?

If violations detected, system flags but does not halt (audit-only).

---

## Menger Pattern: Deterministic Fractal Compression

### Why Not Random Projection?

**Random projection:**
- Non-deterministic: Different Φ_ref^(k+1) on each run
- Breaks audit trail: Cannot reproduce results
- Loses scale-invariance properties

**Menger pattern (deterministic fractal indexing):**
- Deterministic: Same input always produces same output
- Reproducible: Audit trail intact
- Fractal self-similarity: Scale properties preserved

### Implementation

```
Input: Φ_ref^(k) (d_k × d_int), compression_ratio ρ

1. Compute SVD: Φ_ref^(k) = U·S·V^T
2. Generate Menger indices: I_menger based on ρ and d_k
3. Select columns: U_ρ = U[:, I_menger]
4. Reorthogonalize: QR(U_ρ) → Φ_ref^(k+1)

Output: Φ_ref^(k+1) (⌊ρ·d_k⌋ × d_int)
```

---

## Post-Convergence Analysis

### Basin Taxonomy

Build hierarchical tree:

```
Level 0: {Basin_0, Basin_1, Basin_2, ...}
           ↓ (parent-child links)
Level 1: {Basin_0_1, Basin_0_2, ...}
```

Parent-child link: Basin B^(k+1) has parent B^(k) iff representatives overlap spatially.

### Drift Trajectory

Global drift across all levels:

```
ℒ = [L_∞^(0), L_∞^(1), ..., L_∞^(k_max)]
```

Expected: Monotonic non-decrease, then plateau.

### Convergence Signature

At each level:

```
γ^(k) = (
    n_iterations^(k),    # Iterations to converge
    rank^(k),            # Numerical rank at convergence
    n_basins^(k),        # Number of basins found
    L_∞^(k)              # Final drift value
)
```

---

## Implementation Roadmap (v1.1.0)

### Phase 1: FractalAnchor (Week 1)
- [ ] Implement menger_compress() with deterministic indexing
- [ ] Validate scale-invariance preservation
- [ ] Test on synthetic hierarchies

### Phase 2: MengerSpongeQ64 Core (Week 1-2)
- [ ] Implement level-by-level iteration
- [ ] Representative extraction
- [ ] Basin taxonomy construction

### Phase 3: Validation & Analysis (Week 2-3)
- [ ] MultiScaleAnalyzer for invariant checking
- [ ] RecursiveBasins for taxonomy assembly
- [ ] Test on real hierarchical data

---

## Consistency with Base Q64

The Menger Sponge framework does not replace Q64 v1.0.0—it *extends* it:

```
Q64 v1.0.0 (single scale, four primitives) ← CORE
              ↓
Menger Sponge v1.1.0 (multi-scale, replicated primitives) ← EXTENSION
              ↓ (applies Q64 recursively at each level)
              ↓ (uses same convergence criterion, same F_θ structure)
              ↓ (frozen references ensure scale-invariance)
```

**Key property:** Each level is an independent instantiation of the four-primitive Q64 structure.

---

**End of Document**
