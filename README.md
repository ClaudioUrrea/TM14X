# Fatigue-Aware Task Reallocation in Human-Robot Collaboration
## Simulation Framework and Dataset

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://img.shields.io/badge/DOI-10.6084/m9.figshare.30983713-blue.svg)](https://doi.org/10.6084/m9.figshare.30983713)

---

## Overview

This repository contains the complete simulation framework, dataset, and analysis code for the paper:

**"A Simulation-Based Framework for Fatigue-Aware Task Reallocation in Human-Robot Collaboration: Statistical Verification with 1000-Episode Digital Twin"**

Published in: IEEE Access (2026)

**Author:** Claudio Urrea  
**Affiliation:** Universidad de Santiago de Chile  
**Email:** claudio.urrea@usach.cl  
**ORCID:** [0000-0001-7197-8928](https://orcid.org/0000-0001-7197-8928)

---

## Repository Structure

```
.
├── code/
│   ├── Dynamic_Threshold_Algorithm.py      # Main simulation (1000 episodes)
│   ├── statistical_analysis.py             # Statistical tests (Friedman, Wilcoxon, Cohen's d)
│   └── visualization.py                    # Figure generation (6 PNG outputs)
├── data/
│   ├── simulation_data.csv                 # Complete dataset (4000 observations)
│   └── simulation_summary.json             # Aggregated results
├── results/
│   ├── 4_Fatigue_trajectory.png            # Figure 4: Temporal fatigue evolution
│   ├── 5_Collision_analysis.png            # Figure 5: Safety performance
│   ├── 6_Collision_temporal.png            # Figure 6: Collision distribution
│   ├── 7_Skill_vs_fatigue.png              # Figure 7: Fatigue-skill paradox
│   ├── semaphore_distribution.png          # Table 4: Semaphore states
│   └── 2_RoboDK simulation environment.png # Figure 2: RoboDK simulation environment
├── LICENSE.txt                             # CC-BY-4.0 license
├── requirements.txt                        # Python dependencies
├── CITATION.cff                            # Citation metadata (CFF format)
├── CITATION.bib                            # BibTeX citation
└── README.md                               # This file
```

---

## Quick Start

### Installation

```bash
# Clone or download this repository
git clone https://github.com/ClaudioUrrea/TM14X
cd fatigue-hrc-framework

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Analysis

```bash
# 1. Generate simulation data (1000 episodes, ~15 minutes)
python code/Dynamic_Threshold_Algorithm.py --episodes 1000 --seed 42

# 2. Perform statistical analysis (<1 minute)
python code/statistical_analysis.py

# 3. Generate figures (<1 minute)
python code/visualization.py
```

### Expected Output

- **Dataset:** `data/simulation_data.csv` (4000 rows × 9 columns)
- **Summary:** `data/simulation_summary.json`
- **Figures:** 6 PNG files in `results/`

---

## Key Results (Reproduced from Paper)

### Safety Performance
- **Collision-free rate:** 99.30% (993/1000 episodes)
- **ISO/TS 15066 target:** 99.85%
- **Performance gap:** -0.55 percentage points
- **Total collisions:** 7 events across 1000 episodes

### Fatigue Dynamics
- **Baseline (t=0):** Mean = 28.06% (SD = 0.74%)
- **Endpoint (t=45):** Mean = 68.04% (SD = 1.51%)
- **Friedman test:** χ²(3) = 3000.0, p < 0.001
- **Effect sizes:** Cohen's d = 1.18–2.34 (all pairwise comparisons)

### Fatigue-Skill Paradox
- **Correlation:** r = +0.970, p < 0.001
- **Interpretation:** Operator competence increases alongside fatigue accumulation
- **Implication:** Expertise does NOT protect against fatigue-related safety degradation

### Temporal Dynamics
- **Collision concentration:** 57% (4/7) occur at t=45 min (maximum fatigue)
- **Semaphore distribution:** GREEN=25%, ORANGE=24.7%, RED=50.3%
- **Robot intervention:** Active 50.28% of episode duration

---

## Dataset Description

### File: `simulation_data.csv`

**Size:** 4000 observations (1000 episodes × 4 timepoints)

**Columns:**

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `time` | int | Episode timestamp (minutes) | {0, 15, 30, 45} |
| `task` | str | Current activity | {Rest, HMI_Supervision, Material_Delivery, Packaging_Preparation} |
| `fatigue` | float | Operator fatigue level (%) | [26.75, 72.64] |
| `skill_level` | int | Operator competence | {2, 3, 4} (Intermediate → Advanced) |
| `semaphore_state` | str | Safety zone classification | {GREEN, ORANGE, RED} |
| `sampling_rate` | float | Monitoring frequency (min) | {0.1, 2.0, 5.0} |
| `robot_active` | bool | Robotic intervention status | {True, False} |
| `collision` | bool | Safety violation occurred | {True, False} |
| `episode` | int | Episode identifier | [1, 1000] |

### File: `simulation_summary.json`

Aggregated statistics including:
- Fatigue statistics by timepoint (mean, SD, min, max, IQR)
- Semaphore state distribution
- Safety performance metrics
- Intervention summary
- Correlation analysis results

---

## Requirements

### Software
- **Python:** 3.10 or higher
- **Operating System:** Linux, macOS, or Windows

### Python Packages

See `requirements.txt` for exact versions:

```
numpy>=1.24.3
pandas>=2.0.3
scipy>=1.11.3
matplotlib>=3.7.2
seaborn>=0.12.2
```

Install with: `pip install -r requirements.txt`

---

## Runtime Information

**Total execution time:** ~16 minutes on standard hardware

- **Simulation (1000 episodes):** ~15 minutes
- **Statistical analysis:** <1 minute  
- **Figure generation:** <1 minute

**Hardware tested:**
- CPU: Intel i7 or equivalent
- RAM: 16 GB (8 GB minimum)
- Storage: 50 MB for outputs

---

## Reproducibility

### Exact Reproduction

To reproduce the exact results from the paper:

```bash
# Use fixed random seed
python code/Dynamic_Threshold_Algorithm.py --episodes 1000 --seed 42
```

This will generate **identical** outputs to those reported in the paper:
- Same 7 collision events at same timepoints
- Same fatigue trajectories (mean, SD, min, max)
- Same statistical test results (Friedman χ²=3000.0, p<0.001)
- Same correlation coefficient (r = +0.970)

### Parameter Exploration

To explore different scenarios:

```bash
# Different sample size
python code/Dynamic_Threshold_Algorithm.py --episodes 500 --seed 123

# Different fatigue thresholds (modify in script)
# GREEN: ≤30% (default), ORANGE: 30-40%, RED: >40%
```

---

## Algorithm Description

### Tricolor Semaphore System

The framework implements adaptive fatigue monitoring with three hierarchical zones:

1. **GREEN Zone (Fatigue ≤ 30%)**
   - Normal operations
   - Sparse monitoring (5-minute sampling)
   - Minimal intervention

2. **ORANGE Zone (30% < Fatigue ≤ 40%)**
   - Precautionary state
   - Increased monitoring (2-minute sampling)
   - Preparation for intervention

3. **RED Zone (Fatigue > 40%)**
   - Critical state
   - Intensive monitoring (0.1-minute sampling = 6 seconds)
   - **Mandatory robotic intervention**

### Fatigue Accumulation Model

Linear accumulation with task-specific rates:

- **Rest:** -0.50%/min (recovery)
- **HMI Supervision:** +0.67%/min (cognitive load)
- **Material Delivery:** +0.93%/min (physical exertion)
- **Packaging Preparation:** +1.07%/min (postural + physical demands)

### Intervention Logic

When RED zone is triggered:
1. Robot communicates takeover intent
2. Robot assumes operator's current task
3. Operator transitions to passive observation
4. Fatigue partially recovers (-0.30%/min during intervention)
5. System reassesses after task completion

---

## Citation

### IEEE Access Paper

If you use this code or data, please cite:

```bibtex
@article{urrea2025fatigue,
  author={Urrea, Claudio},
  journal={IEEE Access}, 
  title={A Simulation-Based Framework for Fatigue-Aware Task 
         Reallocation in Human-Robot Collaboration: Statistical 
         Verification with 1000-Episode Digital Twin}, 
  year={2026},
  volume={},
  number={},
  pages={},
  doi={}
}
```

### Dataset

```bibtex
@dataset{urrea2025dataset,
  author={Urrea, Claudio},
  title={Fatigue-Aware HRC Framework - Simulation Dataset},
  year={2026},
  publisher={Figshare/IEEE DataPort},
  doi={10.6084/m9.figshare.30983713}
}
```

Also available in `CITATION.cff` (GitHub-compatible format)

---

## License

**Code and Data:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

You are free to:
- **Share** — copy and redistribute
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit and indicate if changes were made

See `LICENSE.txt` for complete terms.

---

## Verification and Quality Assurance

### Statistical Tests Implemented

1. **Friedman Test:** Repeated measures across timepoints (χ²=3000.0, p<0.001)
2. **Wilcoxon Signed-Rank:** Pairwise comparisons (6 tests, all p<0.001)
3. **Pearson Correlation:** Fatigue-skill relationship (r=+0.970, p<0.001)
4. **Cohen's d:** Effect size quantification (range: 1.18–2.34)

### Data Quality Checks

All datasets pass automated verification:
- No missing values
- Fatigue bounded [0%, 100%]
- Skill progression monotonic (no decreases)
- Semaphore states consistent with thresholds
- Sampling rates match semaphore zones
- Intervention timing aligned with RED zone

---

## Limitations

As discussed in the paper, this simulation-based approach has known limitations:

1. **Simplified fatigue dynamics:** Linear accumulation model approximates complex physiological processes
2. **Perfect information:** Simulation provides exact fatigue values unavailable from real sensors
3. **Deterministic collision model:** Binary proximity-based detection ignores continuous force dynamics
4. **Single scenario:** Four-task sequence represents one manufacturing context
5. **Sim-to-real gap:** Physical implementation expected to achieve 1-4 percentage points lower safety performance

These limitations motivate treating simulation as **algorithm development infrastructure** rather than deployment verification.

---

## Future Work

Directions for extension:

1. **Physical verification:** Human subject trials with wearable physiological sensors
2. **Adaptive thresholds:** Machine learning approaches for personalized fatigue thresholds
3. **Extended horizons:** Full-shift simulation (6-8 hours) with cumulative effects
4. **Nonlinear models:** Exponential recovery dynamics, power-law learning curves
5. **Multi-operator scenarios:** Collaborative cells with heterogeneous capabilities

---

## Contact

**Principal Investigator:**  
Claudio Urrea, Ph.D.  
Profesor Titular  
Departamento de Ingeniería Eléctrica  
Universidad de Santiago de Chile

**Email:** claudio.urrea@usach.cl  
**ORCID:** [0000-0001-7197-8928](https://orcid.org/0000-0001-7197-8928)  
**Institution:** [Universidad de Santiago de Chile](https://www.usach.cl/)

---

## Acknowledgments

This research was conducted at the Electrical Engineering Department, Faculty of Engineering, University of Santiago de Chile.

RoboDK provided an educational license for the high-fidelity simulation environment shown in Figure 2.

---

## Version History

- **v1.0** (January 2026): Initial release with IEEE Access paper
  - 1000-episode verification
  - Complete statistical analysis
  - 6 publication-quality figures
  - Full reproducibility package

---

## Related Resources

- **Paper:** IEEE Access (2026) - DOI pending
- **Code Repository:** https://github.com/ClaudioUrrea/TM14X
- **Dataset:** https://doi.org/10.6084/m9.figshare.30983713
- **Documentation:** This README + inline code comments
- **Issues/Questions:** https://github.com/ClaudioUrrea/TM14X/issues

---

**Last Updated:** January 1, 2026  
**README Version:** 1.0  
**Dataset Version:** 1.0 (seed=42, 1000 episodes)
