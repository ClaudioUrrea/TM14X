# Changelog

All notable changes to the Fatigue-Aware HRC Simulation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-01-01

### Added - Initial Release

#### Core Simulation
- Complete simulation framework for 1000-episode validation
- Dynamic threshold algorithm (`Dynamic_Threshold_Algorithm.py`)
- Tricolor semaphore system (GREEN/ORANGE/RED zones)
- Adaptive sampling rates (5.0 min → 2.0 min → 0.1 min)
- Task-specific fatigue accumulation models
- Robotic intervention logic

#### Dataset
- `simulation_data.csv`: 4000 observations (1000 episodes × 4 timepoints)
- `simulation_summary.json`: Aggregated statistical results
- Complete reproducibility with seed=42

#### Statistical Analysis
- Friedman test for repeated measures (χ²=3000.0, p<0.001)
- Wilcoxon signed-rank pairwise comparisons (6 tests, all p<0.001)
- Pearson correlation analysis (r=+0.970, p<0.001)
- Cohen's d effect size quantification (range: 1.18–2.34)
- IQR calculations for all timepoints

#### Visualization
- Figure 4: Temporal fatigue evolution trajectory
- Figure 5: Safety performance comparison vs ISO/TS 15066
- Figure 6: Collision temporal distribution
- Figure 7: Fatigue-skill paradox illustration
- Semaphore state distribution chart
- RoboDK simulation environment screenshot

#### Documentation
- Comprehensive README.md with quick start guide
- LICENSE.txt (CC-BY-4.0)
- CITATION.cff (GitHub citation format)
- CITATION.bib (BibTeX format)
- requirements.txt with pinned versions
- .gitignore for clean repository

#### Results
- 99.30% collision-free operation (993/1000 episodes)
- 0.55 percentage point gap to ISO/TS 15066 target (99.85%)
- Fatigue-skill paradox discovery (r=+0.970)
- 57% collision concentration at maximum fatigue (t=45 min)
- 100% intervention reliability when triggered
- 50.28% robot active time

### Technical Specifications
- Python: 3.10+ compatible
- Dependencies: NumPy 1.24.3, Pandas 2.0.3, SciPy 1.11.3, Matplotlib 3.7.2, Seaborn 0.12.2
- Runtime: ~16 minutes (1000 episodes)
- Memory: <2 GB
- Storage: ~50 MB for outputs

### Reproducibility Features
- Fixed random seed (42) for exact replication
- Deterministic state transitions
- Complete parameter documentation
- All source code included
- Full dataset published

### Quality Assurance
- Zero missing values in dataset
- All validation checks passed
- Statistical significance confirmed (all p<0.001)
- Consistent with published IEEE Access paper
- 100% code coverage for core functions

---

## [Unreleased]

### Planned Features
- [ ] Extended temporal horizons (6-8 hour full shifts)
- [ ] Adaptive personalized thresholds (machine learning)
- [ ] Nonlinear fatigue dynamics (exponential recovery)
- [ ] Multi-operator scenarios
- [ ] Physical validation interface for sensor data
- [ ] Real-time dashboard visualization
- [ ] Parameter sensitivity analysis tools
- [ ] Automated report generation

### Potential Improvements
- [ ] GPU acceleration for large-scale simulations
- [ ] Parallel execution for parameter sweeps
- [ ] Interactive Jupyter notebook tutorials
- [ ] Docker containerization for reproducibility
- [ ] Continuous integration/deployment (CI/CD)
- [ ] Additional statistical tests (permutation tests, bootstrapping)
- [ ] Extended documentation with use cases
- [ ] Video demonstrations

---

## Version History Summary

| Version | Date | Description | Key Changes |
|---------|------|-------------|-------------|
| 1.0.0 | 2026-01-01 | Initial release | Complete framework with IEEE Access paper |

---

## Citation

If you use this software in your research, please cite:

```bibtex
@article{urrea2026fatigue,
  author={Urrea, Claudio},
  journal={IEEE Access}, 
  title={A Simulation-Based Framework for Fatigue-Aware Task 
         Reallocation in Human-Robot Collaboration: Statistical 
         Verification with 1000-Episode Digital Twin}, 
  year={2026}
}
```

---

## Contact

**Maintainer:** Claudio Urrea  
**Email:** claudio.urrea@usach.cl  
**ORCID:** [0000-0001-7197-8928](https://orcid.org/0000-0001-7197-8928)

---

**Note:** This changelog follows semantic versioning. Given version number MAJOR.MINOR.PATCH:
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions  
- PATCH version for backwards-compatible bug fixes
