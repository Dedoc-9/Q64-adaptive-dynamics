# Contributing to Q64

Thank you for your interest in contributing to Q64: Adaptive Representational Dynamics! This document outlines the contribution process and guidelines.

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:
- Be respectful and constructive in all interactions
- Assume good intent from other contributors
- Focus on the idea, not the person
- Reach out to maintainers if you encounter any issues

---

## How to Contribute

### 1. Reporting Bugs

**Before submitting a bug report:**
- Check existing issues to avoid duplicates
- Verify the issue reproduces on latest main branch
- Gather detailed information (Python version, OS, minimal reproducible example)

**When submitting:**
- Use the bug report template
- Include Python version, NumPy version, exact error message
- Provide minimal reproducible code
- Attach generated plots or output files if relevant

### 2. Suggesting Features or Enhancements

**Before suggesting:**
- Check open issues and roadmap to avoid duplicates
- Consider whether the feature aligns with Q64's core mission (multi-scale structure discovery)

**When suggesting:**
- Describe the motivation and use case
- Provide examples of expected behavior
- Link to relevant papers or references if applicable
- Note any dependencies or breaking changes

### 3. Contributing Code

#### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/Dedoc-9/Q64-adaptive-dynamics.git
cd q64-adaptive-dynamics

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,docs,examples]"
```

#### Code Style

**Python:** Follow PEP 8 with Black formatting

```bash
# Format code
black python/

# Check style
flake8 python/
isort python/

# Type checking
mypy python/q64/
```

**Rust:** Follow standard Rust conventions

```bash
# Format
cargo fmt

# Lint
cargo clippy
```

#### Testing

**Run unit tests:**
```bash
pytest tests/ -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=q64 --cov-report=html
```

**Run benchmarks:**
```bash
cargo bench
```

#### Documentation

- Write docstrings for all public functions (NumPy style)
- Update relevant .md files in docs/
- Add examples for new features
- Run `make html` in docs/ to preview

#### Commit Guidelines

**Commit messages should:**
- Be clear and descriptive
- Reference relevant issues (#123)
- Follow format: `[TYPE] Brief description`

**Types:**
- `[FEAT]` - New feature
- `[FIX]` - Bug fix
- `[DOCS]` - Documentation
- `[TEST]` - Test addition/fix
- `[PERF]` - Performance improvement
- `[REFACTOR]` - Code refactoring

**Example:**
```
[FEAT] Add adaptive threshold tuning for spectral projection (#42)

- Implement automatic tau selection based on data dimensionality
- Add validation tests for threshold selection
- Update documentation with tuning guidelines

Closes #42
```

### 4. Pull Request Process

1. **Fork & branch:** Create feature branch from main
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Code & test:** Implement changes with full test coverage
   - Tests must pass: `pytest tests/`
   - Code must pass linting: `black`, `flake8`, `mypy`
   - Documentation must be updated

3. **Commit:** Follow commit guidelines above

4. **Push & create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```
   - Fill PR template completely
   - Reference related issues
   - Describe testing performed

5. **Review:** Respond to reviewer feedback
   - Code review may request changes
   - Maintain conversation and iterate
   - Do not force-push during review (maintain history)

6. **Merge:** Maintainers will merge after approval

---

## Priority Areas for Contribution

### High Priority (v1.0.x)

- **Unit Tests:** test_mi_estimator.py, test_projection.py, test_convergence.py
- **Example Scripts:** basic_usage.py, synthetic_basin_discovery.py, drift_monitoring.py
- **Documentation:** Expand design_decisions.md, troubleshooting guide

### Medium Priority (v1.1.0)

- **Menger Sponge Implementation:**
  - menger_sponge_core.py (hierarchical engine)
  - fractal_reference.py (fractal compression)
  - multi_scale_analysis.py (cross-scale invariants)
  - recursive_basins.py (basin taxonomy)

- **Rust FFI Bindings:** src/lib.rs for Python-Rust integration

### Lower Priority (v1.5.0+)

- **Performance Optimization:** GPU acceleration, parallelization
- **Extended Language Bindings:** Swift, C, MATLAB
- **Production Hardening:** Logging, monitoring, configuration

---

## Development Guidelines

### Mathematical Rigor

- All numerical claims must reference equations
- Include complexity analysis for new algorithms
- Test edge cases (singular matrices, near-convergence, timeouts)
- Cite relevant papers

### Code Quality

- Aim for >90% test coverage
- Write type hints for all public functions
- Use descriptive variable names (avoid single letters except in mathematical code)
- Add docstrings following NumPy style

### Performance Considerations

- Profile new code: `pytest --benchmark`
- Avoid unnecessary copies in hot loops
- Use vectorized operations (NumPy, not loops)
- For Rust code: benchmark before and after

### Documentation Standards

- Document assumptions and limitations
- Include numerical examples
- Cross-reference related sections
- Update README.md for user-facing changes

---

## Review Criteria

Code will be approved if it:

✅ **Functionality**
- Solves stated problem correctly
- Passes all tests (existing + new)
- Handles edge cases gracefully

✅ **Code Quality**
- Follows style guidelines (Black, flake8, mypy)
- Includes type hints
- Has clear docstrings

✅ **Testing**
- New features have unit tests
- Tests cover normal and edge cases
- Test names are descriptive

✅ **Documentation**
- Docstrings updated
- Relevant .md files updated
- Complex logic explained

✅ **Performance**
- No significant performance regression
- Large-scale operations benchmarked
- Memory usage reasonable

---

## Getting Help

- **Questions?** Open a discussion or issue
- **Feature ideas?** Check roadmap first, then open issue
- **Bug found?** File bug report with reproducible example
- **Design discussion?** Start discussion in issues

---

## License

By contributing to Q64, you agree that:
1. Your contributions will be licensed under AGPL-3.0-only
2. You have the right to contribute the code
3. You understand the copyleft requirements of AGPL-3.0

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Recognized in the community

Thank you for contributing!
