# Q64 Empirical Edition: Complete Manifest
## Falsifiable Systems Research Protocol

**Status:** Ready for Hardware Deployment  
**Timeline:** 5 weeks (data collection + analysis)  
**Decision Gate:** Accept/reject H₁ (structure exists) at Week 5  

---

## **Overview**

This folder contains the **falsifiable, empirically-grounded version** of Q64. Unlike earlier theoretical editions, this version:

1. **Specifies testable hypotheses** (H₀ vs H₁)
2. **Defines quantitative success criteria** (≥4 of 5 primary metrics must pass)
3. **Identifies failure modes a priori** (fragmentation, geometry instability, semantic misalignment)
4. **Implements all corrections** (mean-centered covariance, incremental eigentracking, hysteresis, sliding-window Gram)
5. **Provides production code** (StreamOrientedQ64Engine, analysis library, tests)

---

## **File Structure & Purpose**

### **📋 Protocols & Specifications**

| File | Purpose | Audience |
|------|---------|----------|
| **PROTOCOL_V2_EMPIRICAL_FRAMEWORK.md** | Complete 5-week protocol with weekly checklists, success criteria, failure modes, implementation checklist | Project leads, researchers, hardware validation teams |
| **IMPLEMENTATION_NOTES_EMPIRICAL.md** | Rigorous technical specifications of all corrections (mean-centering, incremental eigentracking, hysteresis, sliding-window Gram, entropy-based rank, persistence thresholds) | Implementers, algorithm engineers |
| **FAILURE_MODES_AND_V2_VISION.md** | Detailed analysis of three primary failure modes (fragmentation, geometry instability, semantic misalignment) + contingency actions + v2 research roadmap | Researchers, failure analysis, future planning |

### **💻 Implementation Code**

| File | Purpose | Audience |
|------|---------|----------|
| **core_dynamics_empirical.py** | StreamOrientedQ64Engine with all corrections: mean-centering, incremental eigentracking, hysteresis-bounded τ, sliding-window Gram, three convergence tests, hash binding | Implementers, integrators |
| **analysis_code_library.py** | Production-ready analysis functions: preprocessing, spectral analysis (r_eff, entropy), regime discovery, persistence computation, subspace angles, transition alignment, decision gate | Data analysts, researchers |
| **examples_empirical.py** | 7 worked examples: (1) synthetic data generation, (2) spectral analysis, (3) persistence, (4) subspace stability, (5) transition alignment, (6) full Q64 validation, (7) decision gate | Researchers, learning, validation |
| **test_integration_empirical.py** | 8 integration tests: mean-centering, spectral entropy, persistence detection, subspace angles, transition alignment, decision gate, convergence, Gram sliding window | QA, verification, CI/CD |

### **📖 User-Facing Documentation**

| File | Purpose | Audience |
|------|---------|----------|
| **README_EMPIRICAL.md** | Handheld-focused overview: quick-start examples, empirical success criteria table, expected outcomes, roadmap, what Q64 actually tests | End users, game developers, system integrators |

### **🔗 This Manifest**

| File | Purpose |
|------|---------|
| **MANIFEST.md** | Index of all materials, quick-reference decision flowchart, timeline, deliverables |

---

## **Quick Start: Deploy the Protocol**

### **Week 1–2: Data Collection**

```bash
# On ASUS ROG Ally X:
python examples_empirical.py --collect-data \
  --games [5 titles] \
  --duration 30min \
  --output telemetry/

# Output: game_name.csv with columns:
# timestamp, frame_time, gpu_load, cpu_load, soc_temp, input_lag, power_draw, scene_label
```

### **Week 3: Spectral Analysis**

```python
from analysis_code_library import preprocess_telemetry, analyze_spectral_structure

telemetry_raw = load_telemetry('game_name.csv')
telemetry_centered = preprocess_telemetry(telemetry_raw)  # ← CRITICAL STEP
spectrum = analyze_spectral_structure(telemetry_centered)

print(f"r_eff: {spectrum.r_eff:.2f} (target: ≤14)")
print(f"H: {spectrum.entropy:.3f} (target: ≤log(12)={np.log(12):.3f})")
```

### **Week 4: Persistence + Stability**

```python
from analysis_code_library import discover_regimes_unsupervised, compute_regime_persistence, \
    compute_subspace_angles, analyze_subspace_stability

transitions, ranks = discover_regimes_unsupervised(telemetry_centered)
persistence = compute_regime_persistence(transitions, len(telemetry_centered))

print(f"% time stable: {persistence.pct_time_stable:.1f}% (target: >60%)")

theta = compute_subspace_angles(telemetry_centered)
stability = analyze_subspace_stability(theta)
print(f"θ_median: {stability.median_angle_rad:.3f} rad (target: <0.6)")
```

### **Week 5: Decision Gate**

```python
from analysis_code_library import transition_alignment_with_baseline, evaluate_hypothesis

alignment = transition_alignment_with_baseline(transitions, scene_labels)
print(f"ΔF1: {alignment.delta_f1:.3f} (target: >0.20)")

# Evaluate
metrics = pd.Series({
    'r_eff': spectrum.r_eff,
    'entropy': spectrum.entropy,
    'pct_time_stable': persistence.pct_time_stable,
    'subspace_angle_median': stability.median_angle_rad,
    'delta_f1': alignment.delta_f1
})

verdict = evaluate_hypothesis(metrics)
print(f"H₁ accepted: {verdict['h1_accepted']} ({verdict['passing_count']}/5 criteria)")
```

---

## **Success Criteria (Primary Metrics)**

| Metric | Target | Acceptable | Failure | Decision |
|--------|--------|-----------|---------|----------|
| **r_eff (entropy-based)** | ≤10 | ≤14 | >18 | Required |
| **H (spectral entropy)** | ≤log(12) | ≤log(14) | >log(16) | Required |
| **% time stable** | >70% | >60% | <40% | Required |
| **θ_median (subspace)** | <0.5 rad | <0.65 rad | >0.8 rad | Required |
| **ΔF1 (transition signal)** | >0.30 | >0.20 | <0.10 | Required |

**Gate Logic:** Accept H₁ if **≥4 of 5** criteria pass simultaneously.

---

## **Expected Outcomes & Roadmap**

### **Most Likely Outcome (60–70% probability)**

Moderate low-rank structure:
- r_eff ≈ 8–14 ✓
- pct_time_stable ≈ 60–75% ✓
- θ ≈ 0.4–0.6 rad ✓
- ΔF1 ≈ 0.25–0.50 ✓

**Decision:** H₁ ACCEPTED  
**Next:** v1.0 hardening (4 weeks)  
**Requirements:** Per-game calibration, Φ_ref invalidation detection, conservative fallback

### **Optimistic Outcome (10–15% probability)**

Strong universal structure:
- r_eff ≈ 6–9 ✓
- pct_time_stable > 75% ✓
- θ < 0.4 rad ✓
- ΔF1 > 0.50 ✓

**Decision:** H₁ ACCEPTED (optimal)  
**Next:** Universal Φ_ref possible, skip per-game calibration  
**Benefit:** Zero cold-start overhead

### **Failure Modes (15–25% probability)**

#### **Fragmentation (15–20%)**
- r_eff ok, but pct_time_stable < 50%
- **Diagnosis:** Manifold fragments into many short-lived states
- **Pivot:** Q64 → anomaly detector (event alerting)

#### **Geometry Instability (5–8%)**
- Rank stable, but θ_median > 0.7 rad
- **Diagnosis:** Subspace rotates rapidly despite stable rank
- **Pivot:** Implement incremental subspace tracking (higher cost)

#### **Semantic Misalignment (5–8%)**
- Spectrum good, but ΔF1 < 0.10
- **Diagnosis:** Detected transitions don't align with gameplay
- **Pivot:** Add domain-specific interpretation layer (v1.1 feature)

---

## **Validation Checklist**

### **Pre-Deployment (Week 0)**
- [ ] Synthetic data generation verified
- [ ] Mean-centering implementation tested
- [ ] Integration tests pass (8/8)
- [ ] Code review complete
- [ ] Performance targets confirmed (<150μs/frame, <80KB)

### **Data Collection (Week 1–2)**
- [ ] ASUS ROG Ally X telemetry instrumentation deployed
- [ ] 5 game titles selected (esports, AAA, narrative, emulation, indie)
- [ ] 30 min per game collected at 60 FPS
- [ ] Scene labels manually annotated
- [ ] Data validation: no NaN, no gaps, monotonic timestamps

### **Analysis (Week 3–5)**
- [ ] Spectral analysis complete for all games
- [ ] Persistence analysis complete
- [ ] Subspace angles computed
- [ ] Transition alignment (vs. random baseline) done
- [ ] Decision gate evaluated
- [ ] Failure mode (if any) identified
- [ ] Report generated

### **Post-Decision**
- **If H₁ accepted:**
  - [ ] Per-game calibration automation implemented
  - [ ] Φ_ref invalidation detection added
  - [ ] Fallback logic (non-convergence → reduce power)
  - [ ] Integration with Armoury Crate
  - [ ] v1.0 hardening begins
  
- **If H₀ accepted:**
  - [ ] Failure mode documented
  - [ ] Pivot strategy selected
  - [ ] Contingency code path implemented
  - [ ] Design review for architecture change

---

## **File Dependencies**

```
PROTOCOL_V2_EMPIRICAL_FRAMEWORK.md
    ├── IMPLEMENTATION_NOTES_EMPIRICAL.md (technical details)
    ├── FAILURE_MODES_AND_V2_VISION.md (contingencies)
    └── core_dynamics_empirical.py (implementation)
        ├── analysis_code_library.py (metrics)
        ├── examples_empirical.py (tutorials)
        └── test_integration_empirical.py (validation)

README_EMPIRICAL.md (user-facing summary)
    └── PROTOCOL_V2_EMPIRICAL_FRAMEWORK.md (detailed protocol)
```

**Dependency Order:**
1. Read PROTOCOL_V2_EMPIRICAL_FRAMEWORK.md (big picture)
2. Read IMPLEMENTATION_NOTES_EMPIRICAL.md (technical corrections)
3. Read core_dynamics_empirical.py (understand implementation)
4. Run examples_empirical.py (learn framework)
5. Run test_integration_empirical.py (validate code)
6. Execute protocol on real data (Week 1–5)

---

## **Contact & Support**

**Questions on protocol:**
→ PROTOCOL_V2_EMPIRICAL_FRAMEWORK.md, section "What Q64 Actually Tests"

**Questions on implementation:**
→ IMPLEMENTATION_NOTES_EMPIRICAL.md, relevant section

**Questions on code:**
→ Comments in core_dynamics_empirical.py, analysis_code_library.py

**Failure mode diagnosis:**
→ FAILURE_MODES_AND_V2_VISION.md

**Getting started:**
→ README_EMPIRICAL.md, "Quick Start: Empirical Validation"

---

## **Version History**

- **v1.0-empirical** (2026-05-29): Initial falsifiable protocol, all corrections integrated, ready for hardware deployment

---

## **Key Insight**

> This is no longer "Can we build Q64?"
> 
> This is: "Is game telemetry structured enough to justify Q64?"
> 
> That's a much more honest question.
> 
> And if the data says **yes**, Q64 transitions from speculative to empirically grounded.

---

**Last Updated:** 2026-05-29  
**Status:** ✅ Ready for Deployment
