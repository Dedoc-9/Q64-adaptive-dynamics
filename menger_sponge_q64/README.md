# Menger Sponge Q64: Hierarchical Extension

**Version:** 1.1.0 (v0.9 framework, v1.1.0 target release)  
**Status:** Framework Definition (stubs ready for implementation)  
**License:** AGPL-3.0

---

## Overview

**Menger Sponge Q64** extends the core Q64 system (v1.0.0) to discover structure across **multiple hierarchical scales** simultaneously.

Instead of analyzing a single scale, Menger Sponge applies Q64 recursively at each level:

```
Level 0: Raw observations S₀ (N=1000, d=64)
         ↓ Q64 converges → Basin structure detected
         ↓ Extract representatives

Level 1: Basin signatures S₁ (N=150, d=32)
         ↓ Q64 converges → Meta-basins detected
         ↓ Extract representatives

Level 2: Meta-basin signatures S₂ (N=30, d=16)
         ↓ Q64 converges → Super-meta-basins detected
         ↓ ...

Final: Hierarchical taxonomy
```

### The Problem Solved

Real-world data often exhibits structure at multiple scales. A single Q64 analysis captures only one level of organization. Menger Sponge discovers the **full hierarchy** by:

1. Running Q64 at each scale independently
2. Using frozen reference anchoring to prevent semantic drift across scales
3. Detecting basins, extracting representatives, and recursing
4. Validating scale-invariance via spectral continuity checks

---

## Key Concepts

### Four Primitives at Each Level

Each level maintains the same four irreducible primitives (see THE_CRITICAL_TURN.md):

- **S_k**: State at level k
- **F_θ^(k)**: Dynamics operator at level k
- **Φ_ref^(k)**: Frozen reference anchor (computed once, never updated)
- **L^(k)**: Drift functional at level k

### Fractal Reference Anchoring

The frozen reference compresses deterministically (not randomly) via Menger pattern:

```
Φ_ref^(k) ─(MENGER-COMPRESS)→ Φ_ref^(k+1)
```

This ensures:
- Reproducibility (deterministic compression)
- Scale-invariance (fractal self-similarity)
- Continuity (each level traces back to base)

### State Reduction

Representatives extracted from detected basins:

```
Basins B^(k) at level k
    ↓
Centroid of each basin = representative point
    ↓
S_0^(k+1) = matrix of representatives (much smaller)
```

This creates a natural dimension reduction while preserving cluster membership structure.

---

## Architecture

### Core Modules (Planned for v1.1.0)

```
menger_sponge_q64/
├── menger_sponge_core.py      Main hierarchical engine
├── fractal_reference.py       Fractal compression system
├── multi_scale_analysis.py    Cross-scale validation
└── recursive_basins.py        Basin taxonomy builder
```

### Module Specifications

#### `menger_sponge_core.py` (v1.1.0)

Main orchestrator for hierarchical analysis:

```python
class MengerSpongeQ64:
    def __init__(self, max_levels=5, reduction_threshold=0.1):
        """Initialize hierarchy parameters"""
    
    def analyze_hierarchy(self, S_initial):
        """Full multi-scale analysis
        
        Returns: {
            'levels': [Q64 results L0, L1, ...],
            'basin_taxonomy': Hierarchical structure,
            'cross_scale_invariants': Validation results,
            'drift_trajectory': [L_∞^(0), L_∞^(1), ...]
        }
        """
```

#### `fractal_reference.py` (v1.1.0)

Deterministic Menger compression:

```python
class FractalAnchor:
    def reduce(self, Phi_ref_k, compression_ratio=0.5):
        """Reduce reference from level k to k+1
        
        Uses Menger sponge fractal pattern (deterministic indexing).
        Preserves: rank, entropy, condition number.
        """
    
    def verify_scale_invariance(self, phi_levels):
        """Check that spectral properties scale correctly"""
```

#### `multi_scale_analysis.py` (v1.1.0)

Cross-level invariant validation:

```python
class MultiScaleAnalyzer:
    def detect_violations(self, levels):
        """Find scale-mismatch anomalies
        
        Returns: {
            'spectral_continuity': violations,
            'basin_coherence': violations,
            'drift_anomalies': violations
        }
        """
```

#### `recursive_basins.py` (v1.1.0)

Hierarchical taxonomy construction:

```python
class RecursiveBasins:
    def decompose_hierarchy(self, S_initial):
        """Build full basin taxonomy
        
        Returns: {
            'level_0': Basins at raw scale,
            'level_1': Meta-basins,
            ...
            'taxonomy': Tree structure with parent-child links
        }
        """
```

---

## Usage Example (Planned)

```python
from menger_sponge_q64 import MengerSpongeQ64
import numpy as np

# Hierarchical data
S = generate_hierarchical_structure(n_levels=3, n_samples=1000, dimension=64)

# Analyze hierarchy
sponge = MengerSpongeQ64(max_levels=5, reduction_threshold=0.1)
result = sponge.analyze_hierarchy(S)

# Inspect results
print(f"Levels found: {len(result['levels'])}")
print(f"Basin taxonomy:\n{result['basin_taxonomy']}")
print(f"Scale-invariant violations: {result['cross_scale_invariants']}")
print(f"Drift trajectory: {result['drift_trajectory']}")
```

---

## Implementation Roadmap

### Phase 1 (Week 1): FractalAnchor
- [ ] Implement menger_compress() with deterministic indexing
- [ ] Validate singular value preservation
- [ ] Test on synthetic hierarchies

### Phase 2 (Week 1-2): MengerSpongeQ64 Core
- [ ] Level iteration loop
- [ ] Representative extraction from basins
- [ ] Basin taxonomy assembly

### Phase 3 (Week 2-3): Validation & Analysis
- [ ] MultiScaleAnalyzer.detect_violations()
- [ ] RecursiveBasins taxonomy builder
- [ ] Integration tests on real data

### Phase 4 (Week 3-4): Documentation & Examples
- [ ] Complete examples/
- [ ] Full theory documentation
- [ ] Troubleshooting guide

---

## Integration with Q64 v1.0.0

Menger Sponge does **not** replace Q64—it *extends* it:

```
Q64 v1.0.0 (four primitives, single scale)
    ↓
Used by Menger Sponge at each level k
    ↓
Menger Sponge v1.1.0 (four primitives replicated across scales)
```

Each level runs independent Q64 instance with:
- Same convergence criterion
- Same F_θ structure
- Same audit functional L

Only difference: reference is frozen per level, not global.

---

## Performance

Estimated for N=1000, d=64 initial data:

| Level | Samples | Dimension | Q64 Time | Cumulative |
|-------|---------|-----------|----------|-----------|
| 0 | 1000 | 64 | ~10 min | 10 min |
| 1 | 150 | 32 | ~30 sec | 10.5 min |
| 2 | 30 | 16 | ~5 sec | 10.6 min |
| Total | - | - | - | **~10.6 min** |

With GPU acceleration: ~1 minute total.

---

## Testing (Planned v1.1.0)

```bash
pytest tests/test_menger_sponge.py -v

# Specific tests:
pytest tests/test_menger_sponge.py::test_level_detection
pytest tests/test_menger_sponge.py::test_fractal_reduction
pytest tests/test_menger_sponge.py::test_scale_continuity
pytest tests/test_menger_sponge.py::test_cross_scale_invariants
```

---

## Documentation

See main repo docs/:

- `THE_CRITICAL_TURN.md` - Foundational architecture (why four primitives)
- `HIERARCHICAL_PRIMITIVES.md` - Hierarchical extension details
- `IMPLEMENTATION_NOTES.md` - Mathematical specifications

---

## Status

**Framework Definition:** ✅ Complete (this README + stubs)  
**Core Implementation:** ⏳ Planned (v1.1.0, ~2-3 weeks)  
**Documentation:** ⏳ Planned (v1.1.0)  
**Tests:** ⏳ Planned (v1.1.0)

---

## License

AGPL-3.0-only. See ../LICENSE for terms.

---

**Last Updated:** 2026-05-29  
**Next Release:** v1.1.0 (2-3 weeks from v1.0.0)
