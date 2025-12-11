# Fatigue-Aware Real-Time Task Reallocation in Human-Robot Collaboration: A Longitudinal Digital Twin Framework with Statistical Validation and RoboDK Integration

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fsystems-blue)](https://doi.org/10.3390/systems)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2026a-orange)](https://www.mathworks.com/)
[![RoboDK](https://img.shields.io/badge/RoboDK-5.9.2-red)](https://robodk.com/)

## 📄 Paper Information

**Title:** Fatigue-Aware Real-Time Task Reallocation in Human-Robot Collaboration: A Longitudinal Digital Twin Framework with Statistical Validation and RoboDK Integration

**Author:** Claudio Urrea  
**Affiliation:** Electrical Engineering Department, Faculty of Engineering, University of Santiago of Chile  
**Contact:** claudio.urrea@usach.cl

**Journal:** Systems (MDPI)  
**Year:** 2025  
**Status:** Submitted

## Abstract

Human fatigue in Human-Robot Collaboration (HRC) manufacturing systems remains a critical yet underaddressed factor affecting worker well-being, safety, and system performance. This repository contains the implementation of a novel fatigue-aware task reallocation framework validated through **1,000 industrial episodes** with high statistical significance (Friedman χ²(3) = 3000.00, p < 0.001).

### Key Features

- **99.30% collision-free operation** (approaching ISO/TS 15066:2016 target of 99.85%)
- **Tricolor semaphore alert system** (Green/Orange/Red)
- **Dynamic task reallocation** to collaborative robots
- **Statistical validation** with n=1000 episodes
- **Digital Twin integration** with RoboDK and MATLAB
- **Fatigue-skill paradox analysis** (ρ=0.94, p<0.001)

## Quick Start

### Prerequisites

```bash
Python 3.8+
NumPy >= 1.20.0
Pandas >= 1.3.0
Matplotlib >= 3.4.0
Seaborn >= 0.11.0
SciPy >= 1.7.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ClaudioUrrea/TM14X
cd TM14X

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy

# Run simulation
python Dynamic_Threshold_Algorithm.py
```

### Running the Simulation

```python
from Dynamic_Threshold_Algorithm import run_simulation

# Run with default parameters (1000 episodes)
run_simulation(num_episodes=1000, verbose_frequency=100)
```

## Results

### Safety Performance

| Metric | Value | Target | Status |
|--------|-------|--------|---------|
| Collision-free rate | 99.30% | 99.850% | APPROACHING TARGET |
| Safe episodes | 993/1000 | - | - |
| Total interventions | 2011 | - | - |

### Fatigue Evolution

Time Point | Mean Fatigue | Std Dev | Semaphore State | n
-----------|--------------|---------|-----------------|------
t=0 min    | 28.06%       | 0.74%   | 🟠 ORANGE       | 1000
t=15 min   | 38.07%       | 0.96%   | 🟠 ORANGE       | 1000
t=30 min   | 52.00%       | 1.29%   | 🔴 RED          | 1000
t=45 min   | 68.04%       | 1.51%   | 🔴 RED          | 1000

### Statistical Validation

- **Friedman Test:** χ²(3) = 3000.00, p < 0.001
- **Fatigue-Skill Correlation:** ρ = 0.94, p < 0.001
- **Skill Progression:** Linear (ρ = 0.97, p < 0.001)
- **Fatigue Accumulation:** Exponential (ρ = 0.99, p < 0.001)

## Repository Structure

```
fatigue-aware-hrc/
│
├── Dynamic_Threshold_Algorithm.py     # Main simulation script
├── simulation_data.csv                # Raw simulation data (1000 episodes)
├── simulation_summary.json            # Statistical summary
│
├── visualizations/
│   ├── collision_analysis.png         # Safety performance analysis
│   ├── fatigue_trajectory.png         # Fatigue evolution over time
│   ├── semaphore_distribution.png     # Alert state distribution
│   └── skill_vs_fatigue.png           # Fatigue-skill paradox
│
├── README.md                          # This file
├── LICENSE                            # CC BY 4.0 License
└── CITATION.cff                       # Citation metadata
```

## Methodology

### Fatigue Monitoring System

The system implements a **tricolor semaphore alert system**:

- 🟢 **GREEN (0-30%)**: Safe zone - Normal sampling (5 min)
- 🟠 **ORANGE (31-40%)**: Caution zone - Increased monitoring (2 min)
- 🔴 **RED (>40%)**: Critical zone - Real-time monitoring (0.1 min)

### Task Configuration

| Task | Duration | Fatigue Rate | Physical Demand | Base Fatigue |
|------|----------|--------------|-----------------|--------------|
| Rest Period | 15 min | 0.0%/min | Very Low | 15% |
| HMI Supervision | 15 min | 0.87%/min | Low | 15% |
| Material Delivery | 15 min | 0.67%/min | Moderate-High | 15% |
| Packaging Preparation | 15 min | 0.93%/min | Very High | 15% |

### Hardware Configuration

- **Robots:** 2× Omron TM14X collaborative robots (14 kg payload, 1100 mm reach)
- **Sensors:** Azure Kinect DK, Polar H10 (HR/HRV), Grove GSR sensor
- **Workstation:** AMD Ryzen 9 5950X, 64 GB RAM, NVIDIA RTX 4080 16GB
- **Software:** MATLAB 2026a, RoboDK 5.9.2, Python 3.8+

## Visualization Examples

### Fatigue Trajectory
![Fatigue Trajectory](fatigue_trajectory.png)

### Collision Analysis
![Collision Analysis](collision_analysis.png)

### Semaphore Distribution
![Semaphore Distribution](semaphore_distribution.png)

### Skill vs Fatigue
![Skill vs Fatigue](skill_vs_fatigue.png)

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{urrea2025fatigue,
  title={Fatigue-Aware Real-Time Task Reallocation in Human-Robot Collaboration: A Longitudinal Digital Twin Framework with Statistical Validation and RoboDK Integration},
  author={Urrea, Claudio},
  journal={Systems},
  volume={1},
  number={1},
  pages={0},
  year={2025},
  publisher={MDPI},
  doi={10.3390/systems}
}
```

## 📄 License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

**Claudio Urrea**  
Electrical Engineering Department  
Faculty of Engineering  
University of Santiago of Chile  
Email: claudio.urrea@usach.cl  
ORCID: [0000-0001-7197-8928](https://orcid.org/0000-0001-7197-8928)

## Acknowledgments

This research was conducted at the University of Santiago of Chile, Electrical Engineering Department. Special thanks to the robotics and human factors research community for their valuable feedback.

## Data Availability

- **Simulation Data:** Available in this repository (`simulation_data.csv`)
- **Full Dataset:** Available on FigShare at https://doi.org/10.6084/m9.figshare.30520922
- **Code:** Available on GitHub at https://github.com/ClaudioUrrea/TM14X

## Related Publications

1. Urrea, C. (2025). "Fatigue-Aware Real-Time Task Reallocation in Human-Robot Collaboration..." *Systems*, MDPI.

---

**Keywords:** Human-Robot Collaboration, Fatigue Monitoring, Digital Twin, Task Reallocation, Industry 4.0, Ergonomics, Adaptive Systems, RoboDK Integration, Statistical Validation, Semaphore Alerts, Omron TM14X

**MSC:** 93C85; 93D21; 93C40; 90C29; 90B50; 68T40; 68T05; 49J15; 93B52

---

*Last updated: November 2025*
