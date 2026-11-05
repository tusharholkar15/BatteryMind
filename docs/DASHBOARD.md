# BatteryMind — Interactive Streamlit Dashboard Documentation

**Project:** BatteryMind — Small-Data Battery Performance Optimization using Active Learning  
**Module:** Phase 5 Interactive Streamlit Research Application  
**File:** `app.py`  
**Author:** Deep Learning & Battery R&D Engineering  

---

## 1. Overview & Purpose

The **BatteryMind Streamlit Dashboard** translates complex active-learning algorithms and Gaussian Process uncertainty quantification into an intuitive, research-grade visual interface. Designed for battery researchers, data scientists, and engineering leadership, the application demonstrates how an active-learning framework systematically selects experiments to discover high-longevity batteries while minimizing costly testing cycles under small-data constraints.

### Key Value Proposition
- **Small-Data Efficiency:** Achieves predictive accuracy ($R^2 = 0.9052$, $\text{MAE} = 81.9$ cycles) with only $N = 20$ labeled cells.
- **Accelerated Discovery:** Identifies the longest-lived cell ($2,160$ cycles) in an average of **15.4 experiments**, compared to $19.0 - 31$ for random testing.
- **Calibrated Uncertainty:** Epistemic posterior variance $\sigma(x)$ provides $97.67\%$ empirical coverage on held-out test partitions.
- **Zero Leakage:** Strictly isolates early-cycle signatures ($N \le 100$ cycles) and separates training from test cohorts.

---

## 2. Quickstart & Execution

### Prerequisites
Ensure all dependencies from `requirements.txt` are installed:
```bash
pip install -r requirements.txt
```

### Running the Application
Launch the Streamlit dashboard from the repository root:
```bash
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

### Running Automated Tests
Run the comprehensive test suite (23 unit tests across feature pipeline, active learning engine, and dashboard):
```bash
python -m pytest tests -v
```

---

## 3. Application Architecture & Pages

The application is structured into six modular, scientifically rigorous views accessible via the sidebar navigation:

```
app.py
  ├── 1. Overview                 (Executive summary, benchmarks, methodology safeguards)
  ├── 2. Cycle-Life Predictor     (Interactive inference, 95% credible intervals, uncertainty tiers)
  ├── 3. Active Learning Sim      (Step-by-step acquisition playback, candidate pool status)
  ├── 4. Battery Explorer         (Cohort filtering, raw Qdlin curves, ΔQ100-10 degradation)
  ├── 5. Strategy Comparison      (Multi-trial quantitative tables, AL vs Random comparison)
  └── 6. Methodology              (Mathematical equations, GP kernels, 5-point leakage audit)
```

### Page 1: Overview
- **Executive Summary:** Contextualizes the battery lifetime prediction challenge and the experimental bottleneck of multi-year cycling tests.
- **Key Quantitative Metrics:**
  - Total Pool: $124$ commercial cells ($41$ Train, $43$ Test 1, $40$ Test 2).
  - Observation Horizon: Cycle $100$ ($5 - 15\%$ of total lifespan).
  - Baseline Accuracy: $\text{MAE} = 81.9$ cycles, $R^2 = 0.9052$ at $N = 20$.
  - Discovery Speedup: $15.4$ vs $19 - 31$ cells.
- **Active Learning Cycle:** Conceptual flowchart showing the loop between GP uncertainty, acquisition functions, candidate query, and retrospective retraining.
- **Core Safeguards:** Highlights strict temporal horizons, standardization isolation, and clean dataset partitioning.

### Page 2: Cycle-Life Predictor
- **Interactive Inputs:** Sliders for all five verified degradation features:
  1. `log10_var_deltaQ` ($\log_{10}(\text{Var}(\Delta Q_{100-10}))$, $[-5.50, -2.50]$)
  2. `min_deltaQ` ($\min(\Delta Q_{100-10})$, $[-0.200, 0.000]$ Ah)
  3. `mean_deltaQ` ($\text{mean}(\Delta Q_{100-10})$, $[-0.100, 0.000]$ Ah)
  4. `q_cycle100` ($Q_{100}$, $[0.850, 1.120]$ Ah)
  5. `fade_100_10` ($Q_{100} - Q_{10}$, $[-0.120, 0.010]$ Ah)
- **Preset Profiles:** One-click presets representing typical physical regimes:
  - *Long-Lived Battery:* $\sim 2,000+$ cycles.
  - *Typical Battery:* $\sim 750$ cycles.
  - *Rapid Degradation:* $\sim 300$ cycles.
- **Inference Engine:** Gaussian Process Regressor with anisotropic Matérn $\nu = 2.5$ kernel fit on the 41 training cells.
- **Outputs & Uncertainty:**
  - Point prediction $\hat{y} = 10^{\mu(x)}$ (cycles).
  - Model uncertainty $\pm \sigma$ in $\log_{10}$ space and physical percentage span.
  - Asymmetric $95\%$ credible interval $[10^{\mu - 1.96\sigma}, 10^{\mu + 1.96\sigma}]$.
  - Reliability assessment badge: Low ($<15\%$), Medium ($15 - 30\%$), or High ($>30\%$) uncertainty.
- **Scientific Disclaimer:** Clarifies that predictions reflect controlled 30°C conditions rather than unconditional manufacturer warranties.

### Page 3: Active Learning Simulator
- **Simulation Parameters:**
  - Initial Seed Size ($N_0 \in \{10, 15, 20\}$).
  - Acquisition Strategy (`EXPECTED_IMPROVEMENT`, `UCB`, `MAX_UNCERTAINTY`, `RANDOM`).
  - Experimental Trial Run (Trials 1 through 5).
- **Interactive Step Slider:** Scrub through acquisition iterations ($0$ to $26$ steps).
- **Dynamic Metric Cards:**
  - Labeled vs. Unlabeled Candidate count.
  - Identity of newly selected candidate cell.
  - Revealed ground-truth lifetime vs. model prior prediction.
  - Best lifetime discovered up to the current iteration.
- **Interactive Held-Out Evaluation Curves:**
  - *Test 1 Accuracy:* Live tracking of Test 1 MAE across iterations.
  - *Longevity Discovery:* Step-by-step progress toward locating the $2,160$-cycle optimum.

### Page 4: Battery Explorer
- **Cohort Selector:** Filter across All Cohorts, Train ($41$ cells), Test 1 ($43$ cells), or Test 2 ($40$ cells).
- **Individual Cell Inspection:** Select any of the $124$ cells to view metadata, true lifetime, and extracted feature values.
- **Electrochemical Curve Viewer:** Two-panel high-resolution plots:
  - *Discharge Curves:* Evolution of $Q_{dlin}(V)$ across Cycles 10, 20, 50, and 100 over the $3.6\text{ V} \to 2.0\text{ V}$ window.
  - *Degradation Delta Curve:* Localized capacity difference $\Delta Q_{100-10}(V)$ showing structural phase shifts.

### Page 5: Strategy Comparison
- **Multi-Trial Aggregate Table:** Compares all four acquisition strategies across standardized evaluation budgets ($N \in \{15, 20, 25, 30, 35\}$):
  - In-Distribution Accuracy (Test 1 MAE, RMSE, $R^2$).
  - Out-of-Distribution Accuracy (Test 2 MAE).
  - Longevity Discovery (Max cycles found).
- **Key Scientific Takeaways:**
  - *Discovery Advantage:* UCB and EI find the best cell within $15.4$ trials on average.
  - *Uncertainty Optimization:* Max Uncertainty achieves the lowest Test 1 error ($\text{MAE} = 73.6$ cyc).
  - *Honest Assessment:* Random sampling shows high trial variance ($>3\times$ higher std), proving active learning's stability for experimental planning.

### Page 6: Methodology & Research Foundations
- **Mathematical Formulations:**
  - Feature extraction equations: $\Delta Q_{100-10}(V)$, variance, min, mean, and capacity fade.
  - Target transformation: $z = \log_{10}(y)$.
  - GP prior and covariance functions: Matérn-5/2 kernel with WhiteKernel noise regularization.
  - Acquisition function formulations: Max Uncertainty ($\sigma(x)$), UCB ($\mu(x) + \beta \sigma(x)$), Expected Improvement ($\mathbb{E}[\max(0, y - y^+)]$).
- **5-Point Data Leakage Audit:** Comprehensive documentation of data integrity protocols.
- **Experimental Scope & Limitations:** Clear discussion of chemical scope (LFP/graphite) and retrospective experimental nature.

---

## 4. Technical Specifications & Performance

| Parameter | Specification | Notes |
|---|---|---|
| **Framework** | Streamlit $\ge 1.30.0$ | Pure Python reactive web interface |
| **Numerical Stack** | NumPy, Pandas, Scipy, Scikit-Learn | Standard scientific libraries |
| **Plotting Stack** | Matplotlib | Lightweight, statically rendered SVG/PNG plots |
| **Inference Latency** | $< 5\text{ ms}$ per query | Cached GP model evaluated on CPU |
| **Data Footprint** | $124$ processed rows, $1,260$ AL records | Instant in-memory loading via `@st.cache_data` |
| **Test Coverage** | 23 automated tests (100% pass) | Unit and integration verified |

---

## 5. Security & Boundary Compliance

- **No Remote Calls:** The application runs fully locally; no external network queries are made.
- **Read-Only Raw Data:** All raw cycling files in `data/raw/` are accessed in read-only mode.
- **No Uncontrolled Daemons:** Streamlit is executed interactively by the user and terminated cleanly upon session completion.
