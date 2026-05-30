# Q64 Implementation Plan (consolidated repo next)
## Consolidated Master Document (Single Source of Truth)

**Status:** Week 1 Complete, Week 2a Ready to Start  
**Last Updated:** Post Week 1 Validation  
**Platform:** ASUS ROG Ally X (Windows), Python 3.14.5

---

## Executive Summary

**What you've accomplished (Week 1):**
- ✅ Reference engine validated (q64_stratified_engine.py)
- ✅ All unit tests passing (3/3)
- ✅ Thresholds tuned for real data (ε_R=1e-2, δ_L=0.2, τ=0.4)
- ✅ Synthetic convergence confirmed (all 4 domains converge)

**What you're building (Week 2):**
- Phase 2a (Week 2): Capture infrastructure (game harness + telemetry pipeline)
- Phase 2b (Week 2): Live runtime (real-time monitoring during game execution)
- Phase 3 (Week 2-3): Analysis & H₁ validation (forensic evaluation)

---

## Phase 1: Week 1 Validation (✅ COMPLETE)

### Completed Tasks
- [x] Install Python 3.14.5
- [x] Create requirements.txt (6 dependencies)
- [x] Create q64_stratified_engine.py (stratified multi-domain engine)
- [x] Create q64_reference_test.py (unit tests)
- [x] Run tests: 3/3 PASS
  - Test 1: Initialization ✓
  - Test 2: Synthetic convergence ✓
  - Test 3: H₁ gate evaluation ✓

### Locked Parameters (Do Not Change)
```
Thresholds:
  ε_R (residual):       1e-2
  δ_L (drift):          0.2
  τ (rank threshold):   0.4
  w (window size):      20 frames

Domain Configuration:
  Input:     N=10, k=3   (controller state)
  Physics:   N=6,  k=5   (position + velocity)
  System:    N=12, k=3   (CPU/GPU/thermal)
  Rendering: N=36, k=10  (draw calls, memory, etc.)

H₁ Success Criteria:
  ≥3 of 4 domains must pass pct_stable thresholds
  Input:     ≥80%
  Physics:   ≥70%
  System:    ≥60%
  Rendering: ≥40%
```

### Why Week 1 Matters
The reference engine **proves the math works on synthetic data**. If it didn't converge, you'd know the thresholds or operator logic is broken before touching real games. Now you can be confident Week 2 failures are about game integration, not the algorithm.

---

## Phase 2a: Week 2 Infrastructure (NEXT PHASE)

### Goal
Build the telemetry capture harness so you can collect 64-dimensional state vectors from games in real-time.

### Task Sequence (Build in This Order)

#### Task 1: Create `game_capture_config.json`
**What:** Per-game telemetry configuration (5 titles)
**Why:** Defines where each game lives and what telemetry method to use
**Dependency:** None (start here)
**Acceptance:** Config loads without errors; 5 games defined
**Time:** 1 hour
**Action:**
  1. Copy template from Appendix A
  2. Verify paths for Valorant, Portal 2, Celeste, Minecraft, Elden Ring
  3. Test: `python -c "import json; json.load(open('game_capture_config.json'))"`

---

#### Task 2: Create `capture_profile.yaml`
**What:** Global Q64 configuration (sampling rate, thresholds, output dirs)
**Why:** Central config for all captures; references locked parameters from Week 1
**Dependency:** None (independent of Task 1)
**Acceptance:** YAML loads without errors; thresholds match Week 1 locked values
**Time:** 0.5 hours
**Action:**
  1. Copy template from Appendix B
  2. Set thresholds: ε_R=1e-2, δ_L=0.2, τ=0.4
  3. Create telemetry_logs/ directory
  4. Test: `python -c "import yaml; yaml.safe_load(open('capture_profile.yaml'))"`

---

#### Task 3: Create `state_vector_mapper.py`
**What:** Assemble 64-dimensional state vector from game + OS sources
**Why:** Converts telemetry from 4 heterogeneous domains into single s_t vector
**Dependency:** Task 1 (game_capture_config.json)
**Acceptance:** Assembles 64-dim vectors without errors; shape is (64,)
**Time:** 3 hours
**Action:**
  1. Copy template from Appendix C
  2. Implement StateVectorAssembler class
  3. Test: Assembles synthetic 64-dim vector
  4. Verify: Domain splits correct (input 0-10, physics 10-16, etc.)

---

#### Task 4: Create `hardware_monitor_adapter.py`
**What:** Bridge to Open Hardware Monitor for real-time hardware metrics
**Why:** Provides system_state domain (CPU/GPU/thermal) for s_t vector
**Dependency:** None (but requires Open Hardware Monitor running during capture)
**Acceptance:** Fetches CPU temp, GPU load, fan RPM without errors
**Time:** 2 hours
**Action:**
  1. Copy template from Appendix D
  2. Implement OpenHardwareMonitorAdapter class
  3. Install Open Hardware Monitor on ASUS ROG Ally X (if not present)
  4. Test: `python hardware_monitor_adapter.py` returns real metrics

---

#### Task 5: Create `data_pipeline.py`
**What:** Streaming data flow (telemetry sources → stratified engine → storage)
**Why:** Assembles all pieces: config, state_vector_mapper, hardware_adapter, engine
**Dependency:** Tasks 1-4 (all source adapters)
**Acceptance:** Streams s_t vectors through engine without errors; logs frames to memory
**Time:** 3 hours
**Action:**
  1. Copy template from Appendix E
  2. Implement DataPipeline class
  3. Wire together: StateVectorAssembler + HardwareMonitorAdapter + Q64StratifiedEngine
  4. Test: Run for 100 frames, verify domain convergence detected

---

### Week 2a Acceptance Criteria
✓ All 5 config files created and tested  
✓ All 3 adapters working (state mapper, hardware monitor, pipeline)  
✓ Data flows: [telemetry] → [64-dim vector] → [stratified engine] → [convergence detection]  
✓ Can run synthetic test: `python data_pipeline.py --test --frames 600`

---

## Phase 2b: Week 2 Runtime (AFTER PHASE 2a)

### Goal
Integrate Q64 into live game execution so you can capture real telemetry during gameplay.

### Task Sequence

#### Task 6: Create `q64_realtime_monitor.py`
**What:** Live streaming monitor (runs alongside game)
**Dependency:** Phase 2a complete
**Acceptance:** Streams telemetry, logs convergence events, produces HDF5 file
**Time:** 5 hours

#### Task 7: Create `game_launcher_wrapper.sh` (or `.ps1`)
**What:** Bash script that launches game + Q64 monitor in parallel
**Dependency:** Task 6 (q64_realtime_monitor.py)
**Acceptance:** Launches game and monitor together; both terminate together
**Time:** 1 hour

#### Task 8: Create `telemetry_logger.py`
**What:** Persist telemetry streams to HDF5/CSV
**Dependency:** Phase 2a complete
**Acceptance:** Writes valid HDF5 with frames, convergence, ranks datasets
**Time:** 2 hours

#### Task 9: Create `convergence_visualizer.py` (OPTIONAL)
**What:** Real-time matplotlib plots of domain convergence
**Dependency:** Task 8 (data available to plot)
**Acceptance:** Plots update at 2 Hz without impacting frame rate
**Time:** 3 hours

---

## Phase 3: Analysis & H₁ Validation (AFTER PHASE 2b)

### Task 10: Create `forensic_autopsy.py`
**What:** Post-run analysis (rank evolution, stability metrics)
**Dependency:** Phase 2b complete (HDF5 logs exist)
**Acceptance:** Generates plots + stability metrics per domain
**Time:** 4 hours

### Task 11: Create `h1_gate_evaluator.py`
**What:** Compute H₁ success/failure from logged telemetry
**Dependency:** Task 10 (autopsy generates metrics)
**Acceptance:** Evaluates ≥3 of 4 domains pass; returns PASS/FAIL
**Time:** 3 hours

### Task 12: Create `spectral_report_generator.py`
**What:** Generate comprehensive H₁ validation report (per-game summary)
**Dependency:** Task 11 (H₁ evaluation complete)
**Acceptance:** HTML/PDF report showing all 5 games + domain metrics
**Time:** 4 hours

---

## File Manifest

### Core Files (Must Have)
```
q64-adaptive-dynamics/
├── requirements.txt                  ✅ Created Week 1
├── q64_stratified_engine.py          ✅ Created Week 1
├── q64_reference_test.py             ✅ Created Week 1
├── q64_sensitivity_sweep.py          ✅ Optional utility
├── Q64_IMPLEMENTATION_PLAN.md        ✅ THIS FILE (consolidated master)
│
├── PHASE 2a: Infrastructure (To Create)
│   ├── game_capture_config.json
│   ├── capture_profile.yaml
│   ├── state_vector_mapper.py
│   ├── hardware_monitor_adapter.py
│   └── data_pipeline.py
│
├── PHASE 2b: Runtime (To Create)
│   ├── q64_realtime_monitor.py
│   ├── game_launcher_wrapper.sh
│   ├── telemetry_logger.py
│   └── convergence_visualizer.py (optional)
│
└── PHASE 3: Analysis (To Create)
    ├── forensic_autopsy.py
    ├── h1_gate_evaluator.py
    └── spectral_report_generator.py
```

### Specification Files (Reference, Don't Edit)
```
refined_protocol/
├── Q64_MATHEMATICAL_FOUNDATIONS.md        (Operator T spec)
├── Q64_GOVERNANCE_KERNEL_FINAL.md         (Normative rules)
├── Q64_EXECUTION_LAYER_SPEC.md            (Operational mechanics)
└── Q64_KERNEL_BRIDGE.md                   (Mapping reference)
```

---

## Appendix A: game_capture_config.json Template

[See WEEK_1_2_PATH_FORWARD.md § "game_capture_config.json" for full template]

---

## Appendix B: capture_profile.yaml Template

[See WEEK_1_2_PATH_FORWARD.md § "capture_profile.yaml" for full template]

---

## Appendix C: state_vector_mapper.py Template

[See WEEK_1_2_PATH_FORWARD.md § "state_vector_mapper.py" for full template]

---

## Appendix D: hardware_monitor_adapter.py Template

[See WEEK_1_2_PATH_FORWARD.md § "hardware_monitor_adapter.py" for full template]

---

## Appendix E: data_pipeline.py Template

[See WEEK_1_2_PATH_FORWARD.md § "data_pipeline.py" for full template]

---

## Reference Links (Don't Duplicate Info)

**Why the operator works:**
→ Q64_MATHEMATICAL_FOUNDATIONS.md § 1 (Operator T)

**How convergence predicate works:**
→ Q64_MATHEMATICAL_FOUNDATIONS.md § 2 (Convergence predicate c_t)

**Why these thresholds are safe:**
→ Week 1 Validation results (3/3 tests pass)

**Guard evaluation rules:**
→ Q64_EXECUTION_LAYER_SPEC.md § 4 (Guard evaluation)

**Domain stratification rationale:**
→ Q64_IMPLEMENTATION_PLAN.md § "Phase 2a: Week 2 Infrastructure" (Domain definitions)

---

## Navigation Quick Links

| Need | Go to |
|------|-------|
| What to build next? | → Phase 2a Task Sequence (above) |
| How does the math work? | → Q64_MATHEMATICAL_FOUNDATIONS.md |
| What are the governance rules? | → Q64_GOVERNANCE_KERNEL_FINAL.md |
| How do I run the tests? | → "Phase 1: Week 1 Validation" (above) |
| What thresholds do I use? | → "Locked Parameters" (above) |
| Where does file X go? | → "File Manifest" (above) |

---

## Success Metrics

### Week 2a Complete:
- [ ] All 5 config/code files created
- [ ] Data pipeline runs for 600 frames without errors
- [ ] q64_stratified_engine.py receives valid s_t vectors
- [ ] Domain convergence is detectable in synthetic test

### Week 2b Complete:
- [ ] First 10-minute game capture produces valid HDF5
- [ ] File size: ~5 MB (expected for 600 frames @ 64 dims)
- [ ] Convergence timeline is visible in logs

### H₁ Success (Week 3):
- [ ] All 5 games logged for 30 min each
- [ ] ≥3 of 4 domains show pct_stable ≥ threshold
- [ ] Manifold hypothesis empirically validated
- [ ] Cleared to proceed to hardware mapping

---

## Decision Log

**Option chosen for consolidation:** B (Merge into single master document)  
**Reasoning:** Single source of truth reduces cognitive branching; references prevent duplication  
**Archive:** WEEK_1_2_PATH_FORWARD.md, IMPLEMENTATION_CHECKLIST.txt, CURRENT_REPO_LAYOUT.md → _archive/  
**Active files:** Core code + this plan + refined_protocol specs  

---

**Last edit:** Week 1 complete  
**Next action:** Start Phase 2a Task 1 (game_capture_config.json)
