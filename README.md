# BatteryMind: Small-Data Battery Performance Optimization using Active Learning

> **A research-oriented machine learning framework that evaluates whether uncertainty-aware active learning can accelerate battery lifetime optimization under strict small-data experimental constraints.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 24/24 Passed](https://img.shields.io/badge/Tests-24%2F24%20Passed-brightgreen.svg)](tests/)

---

## 1. Project Statement & Research Scope

Evaluating the lifetime of commercial lithium-ion batteries typically requires **months or years** of continuous cycling per cell to reach standard failure criteria (e.g., 80% capacity retention). In laboratory R&D and materials discovery, researchers operate under a **small-data constraint** ($N \le 20$ completed experiments) and must systematically decide **which test condition to investigate next**.

> **Scientific Boundary & Core Claim:**  
> **"BatteryMind retrospectively simulates active learning on 124 experimental battery cells and evaluates whether uncertainty-aware experiment selection can improve sample efficiency over random sampling."**  
> *BatteryMind does NOT claim to have physically actuated laboratory hardware, nor does it claim to have synthesized or discovered a novel battery chemistry.*

---

## 2. Dataset & Experimental Partitions

BatteryMind utilizes the landmark fast-charging battery dataset published by **Severson et al. (*Nature Energy*, 2019)** and **Attia et al. (*Nature*, 2020)** from the MIT, Stanford, and Toyota Research Institute collaboration.

- **Cell Chemistry:** Commercial A123 Systems APR18650M1A ($\text{LiFePO}_4$ / artificial graphite) cylindrical cells ($1.1\text{ Ah}$ nominal capacity).
- **Cycling Conditions:** Controlled environmental chamber at $30^\circ\text{C}$ with varied multi-stage fast-charging policies and standardized $4\text{C}$ discharge.
- **Total Experimental Pool:** **124 cells** across three strictly separated historical cohorts:
  - **Train Partition (41 cells):** Serves as the candidate pool for active learning simulation.
  - **Test 1 Partition (43 cells):** In-distribution held-out validation cohort.
  - **Test 2 Partition (40 cells):** Out-of-distribution (batch/policy shifted) held-out evaluation cohort.
- **Observed Lifespan:** Ranges from **148 cycles** to **2,237 cycles** (median: 859 cycles).

---

## 3. Early-Cycle Feature Engineering (Horizon $N \le 100$)

To eliminate data leakage, features are extracted exclusively from the first **100 discharge cycles** ($\le 15\%$ of average battery lifespan), referencing the two-phase transition of the differential discharge capacity curve $\Delta Q_{100-10}(V) = Q_{100}(V) - Q_{10}(V)$:

| Feature Name | Mathematical Definition | Physical Meaning |
|---|---|---|
| `log10_var_deltaQ` | $\log_{10}\left(\operatorname{Var}\left(\Delta Q_{100-10}(V)\right)\right)$ | Variance of the differential capacity curve across the $3.6\text{ V} \to 2.0\text{ V}$ plateau ($r = -0.927$ with cycle life). |
| `min_deltaQ` | $\min_V \left(\Delta Q_{100-10}(V)\right)$ | Peak localized capacity loss across the active electrochemical phase transition. |
| `mean_deltaQ` | $\frac{1}{1000} \sum_{k=1}^{1000} \Delta Q_{100-10}(V_k)$ | Integrated downward capacity displacement over 1,000 voltage points. |
| `q_cycle100` | $Q_{100}(2.0\text{V})$ | Nominal discharge capacity measured at the 2.0V cutoff of Cycle 100. |
| `fade_100_10` | $Q_{100}(2.0\text{V}) - Q_{10}(2.0\text{V})$ | Bulk scalar capacity lost between Cycle 10 and Cycle 100. |

```
Raw High-Frequency Cycling (1,000 V-steps x 99 cycles)
                        │
                        ▼ (Cycles 2 to 100 strictly)
      ΔQ100-10(V) Differential Curve Calculation
                        │
                        ▼
      5 Physics-Grounded Early-Cycle Features
```

---

## 4. Probabilistic Modeling & Active Learning Framework

BatteryMind models battery cycle life in logarithmic space ($z = \log_{10}(y)$) using an anisotropic **Gaussian Process Regressor (GPR)**:
$$\mathcal{GP} \sim \mathcal{N}\left(0, \, k(x, x') + \sigma_n^2 I\right)$$
- **Kernel Function:** Scaled Matérn-5/2 kernel ($C(1.0) \times \operatorname{Mat\acute{e}rn}(\nu=2.5)$) combined with a `WhiteKernel` noise regularizer.
- **Uncertainty Quantification:** Provides exact analytic epistemic variance $\sigma(x)$ alongside predictive mean $\mu(x)$, yielding **$97.67\%$ empirical coverage** of 95% credible intervals on Test 1.

### Acquisition Strategies Benchmarked

At each simulated iteration $t$, the agent selects the next experiment $x^*$ from the unlabeled candidate pool:

1. **Random Sampling (Baseline):** Uniform random draw:
   $$x^* \sim \operatorname{Uniform}(\mathcal{U}_t)$$
2. **Maximum Uncertainty (Exploration):** Queries the candidate with highest posterior variance:
   $$x^* = \operatorname{argmax}_{x \in \mathcal{U}_t} \, \sigma(x)$$
3. **Upper Confidence Bound (UCB, Balanced):** Balances predicted lifetime and uncertainty:
   $$x^* = \operatorname{argmax}_{x \in \mathcal{U}_t} \, \left[\mu(x) + \beta \cdot \sigma(x)\right] \quad (\beta = 1.96)$$
4. **Expected Improvement (EI, Exploitation):** Maximizes expected gain over the current best observed cell $y^+$:
   $$\operatorname{EI}(x) = (\mu(x) - y^+) \Phi\left(\frac{\mu(x) - y^+}{\sigma(x)}\right) + \sigma(x) \phi\left(\frac{\mu(x) - y^+}{\sigma(x)}\right)$$

---

## 5. Key Quantitative Results

Multi-trial evaluations across 5 independent random trials with an initial seed of $N_0 = 15$ cells reveal:

| Total Labeled Cells | Strategy | Test 1 MAE (cycles) | Test 1 RMSE (cycles) | Test 1 $R^2$ | Best Found (cycles) |
|---|---|---|---|---|---|
| **15 (Seed)** | Initial Seed | $96.8 \pm 6.1$ | $136.2 \pm 7.4$ | $0.878 \pm 0.015$ | $1,721 \pm 384$ |
| **20** | Random | $88.5 \pm 10.4$ | $127.3 \pm 13.2$ | $0.893 \pm 0.024$ | $1,780 \pm 356$ |
| **20** | **Max Uncertainty** | $84.2 \pm 5.2$ | $120.8 \pm 3.7$ | $0.904 \pm 0.007$ | $1,840 \pm 312$ |
| **20** | **UCB** | $87.1 \pm 4.8$ | $122.9 \pm 4.1$ | $0.901 \pm 0.008$ | **2,160 (Optimum)** |
| **20** | **Expected Improvement** | $86.8 \pm 4.9$ | $123.1 \pm 3.9$ | $0.900 \pm 0.008$ | **2,160 (Optimum)** |
| **35 (Final)** | **Max Uncertainty** | **73.6 ± 3.4** | **102.4 ± 3.1** | **0.931 ± 0.005** | **2,160** |

### Core Scientific Findings:
- **Accelerated Discovery:** UCB and Expected Improvement locate the globally optimal cell in the training pool (`train_cell1`, **2,160 cycles**) in an average of **15.4 total experiments**, whereas Random testing requires between **19.0 and 31 experiments**.
- **Variance Reduction:** Random sampling exhibits over $3\times$ higher performance variance across random seeds ($\text{RMSE std} = 13.2$ vs $3.7$ cycles), underscoring active learning's stability for experimental budgeting.
- **Peak Model Accuracy:** Maximum Uncertainty achieves the highest final generalization accuracy on held-out Test 1 ($\text{MAE} = 73.6$ cycles, $R^2 = 0.9315$).

---

## 6. Repository Architecture

```text
BatteryMind/
├── app.py                      # Interactive 6-page Streamlit dashboard
├── requirements.txt            # Minimal reproducible dependencies
├── .gitignore                  # Clean exclusions for caches, venvs, and secrets
├── README.md                   # Complete scientific documentation & setup guide
├── PROJECT_SCOPE.md            # Problem definition and research boundary
├── DATASET_COMPARISON.md       # Public battery datasets survey & trade-offs
├── data/
│   ├── raw/                    # Original Severson et al. cycle data (read-only)
│   └── processed/
│       └── batterymind_features.csv  # 124 cells x 5 validated features
├── docs/
│   ├── ACTIVE_LEARNING_REPORT.md     # Phase 4 benchmark documentation
│   ├── BASELINE_MODEL_REPORT.md      # Phase 3 small-data modeling report
│   ├── DASHBOARD.md                  # Phase 5 Streamlit architecture & user guide
│   ├── DATASET_INSPECTION.md         # Phase 1 structural inspection report
│   ├── DEPLOYMENT_READINESS.md       # Phase 6 deployment & quality audit
│   ├── EDA_REPORT.md                 # Phase 2 exploratory data analysis
│   ├── FEATURE_SPECIFICATION.md      # Feature equations and leakage checks
│   └── LEAKAGE_ANALYSIS.md           # 5-point data integrity audit
├── reports/
│   ├── active_learning_results.csv   # 1,260 recorded AL trajectory rows
│   ├── baseline_results.csv          # Baseline model performance metrics
│   ├── active_learning_discovery_curve.png
│   ├── active_learning_learning_curves.png
│   └── figures/                      # 7 EDA diagnostic figures
├── src/
│   ├── active_learning.py      # Active learning simulation engine
│   ├── eda.py                  # Exploratory data analysis scripts
│   ├── extract_features.py     # Leakage-free early-cycle feature pipeline
│   ├── fetch_dataset.py        # Authoritative dataset retrieval script
│   ├── inspect_dataset.py      # Dataset structure & integrity validator
│   └── train_baseline.py       # Small-data baseline training (GPR vs Bayesian Ridge)
└── tests/
    ├── test_active_learning.py # Active learning reproducibility & leakage tests
    ├── test_app.py             # Streamlit page rendering & inference unit tests
    └── test_feature_pipeline.py# Feature bounds, NaN, and leakage unit tests
```

---

## 7. Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/BatteryMind.git
cd BatteryMind
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Full Test Suite
```bash
python -m pytest tests -v
```
*(All 24 unit tests should pass with 100% pass rate.)*

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 8. Streamlit Dashboard Features

The application provides an interactive research UI with six dedicated sections:

1. **⚡ Overview:** High-level executive benchmarks, quantitative highlights, and methodology safeguards.
2. **🎯 Cycle-Life Predictor:** Interactive sliders for early degradation features, instant GPR inference, 95% credible intervals, and uncertainty tier classification.
3. **🔄 Active Learning Simulator:** Step-by-step playback slider tracking experimental candidate selection, revealed lifetimes, and live discovery curves.
4. **🔍 Battery Explorer:** Cohort filtering and high-resolution electrochemical curve visualizations ($Q_{dlin}(V)$ and $\Delta Q_{100-10}(V)$) for each cell.
5. **📊 Strategy Comparison:** Multi-trial aggregate metrics table and honest assessment of active learning versus random sampling.
6. **📐 Methodology:** Mathematical formulations, GP kernel specifications, and 5-point leakage audit.

---

## 9. Methodological Safeguards & Limitations

### 5-Point Leakage Prevention Audit
1. **Strict Temporal Horizon:** Only measurements up to Cycle 100 are accessed.
2. **Target Isolation:** Cycle-life labels are strictly shielded from the candidate pool during acquisition.
3. **Split Separation:** Active learning simulations operate exclusively on the 41 Train cells; Test 1 (43 cells) and Test 2 (40 cells) are never evaluated during selection.
4. **Scaler Isolation:** Feature standardizers are fit strictly on the labeled training set at each iteration.
5. **Decoupled Policies:** External policy descriptors (`policies_all.csv`) are decoupled from cell features to prevent unverified speculative row mapping.

### Limitations
- **Single Cell Chemistry:** Validated on commercial cylindrical A123 $\text{LiFePO}_4$/graphite cells at 30°C. Generalization to NMC, NCA, or solid-state batteries requires dedicated retraining.
- **Retrospective Simulation:** Operates on historical physical test data rather than commanding live laboratory hardware in real time.
- **Fixed Discharge Protocol:** Evaluates charging protocol variations under standardized 4C discharge conditions.

---

## 10. Scientific References

1. **Severson, K. A., Attia, P. M., Jin, N., et al.** (2019). *Data-driven prediction of battery cycle life before capacity degradation.* **Nature Energy**, 4(5), 383–391. [doi:10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8).
2. **Attia, P. M., Grover, A., Jin, N., et al.** (2020). *Closed-loop optimization of fast-charging protocols for batteries with machine learning.* **Nature**, 578(7795), 397–402. [doi:10.1038/s41586-020-1994-5](https://doi.org/10.1038/s41586-020-1994-5).
3. **Rasmussen, C. E., & Williams, C. K. I.** (2006). *Gaussian Processes for Machine Learning.* MIT Press.

---

## 11. License

This project is licensed under the MIT License — see the LICENSE file for details.
