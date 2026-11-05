# BatteryMind — Exploratory Data Analysis (EDA) Report (Phase 2)

---

## 1. Executive Summary
This report documents the exploratory data analysis of the **124 commercial lithium-ion cells** ($\text{LiFePO}_4$/graphite, A123 APR18650M1A) from the MIT-Stanford-TRI dataset. All diagnostic scripts are located in [`src/eda.py`](file:///c:/Users/HP/OneDrive/Documents/BatteryMind/src/eda.py) and output figures are archived in [`reports/figures/`](file:///c:/Users/HP/OneDrive/Documents/BatteryMind/reports/figures/).

### Key Quantitative Findings:
- **Cycle-Life Target ($y$):** Highly variable and positively skewed, ranging from **148 cycles to 2,237 cycles** (median: **736.5 cycles**, mean: **801.6 ± 379.7 cycles**).
- **Target Transformation:** $\log_{10}(\text{Cycle Life})$ normalizes the distribution and linearizes the relationship with early degradation features.
- **Physics-Grounded Signal:** The early-cycle degradation curve difference $\Delta Q_{100-10}(V)$ shows an exceptionally strong correlation with final cycle life (**Pearson $r = -0.927$** for $\log_{10}(\mathrm{Var}(\Delta Q_{100-10}))$).
- **Data Quality:** The linearly interpolated discharge capacity ($Q_{dlin}$) grid is completely consistent across all 124 cells (1,000 voltage points from 3.6V to 2.0V, cycles 2 to 100, zero missing values).
- **Identified Outliers:** `test1_cell22` failed prematurely at **148 cycles** (due to severe early lithium plating under high-rate charging), matching established literature findings.

---

## 2. Cycle-Life Distribution & Cohort Statistics

The target variable is the number of cycles until cell discharge capacity falls below 80% of nominal capacity ($0.88\text{ Ah}$).

![Cycle Life Distribution](file:///c:/Users/HP/OneDrive/Documents/BatteryMind/reports/figures/01_cycle_life_distribution.png)

### Summary Statistics Across Cohorts:

| Cohort | Cell Count ($N$) | Min | 25th % | Median | Mean ± Std | 75th % | Max | Skewness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 41 | 300.0 | 461.0 | 527.0 | 673.8 ± 327.1 | 788.0 | 2,160.0 | +2.05 |
| **Test 1** (Primary Test) | 43 | 148.0 | 455.5 | 561.0 | 709.3 ± 395.8 | 845.5 | 2,237.0 | +1.77 |
| **Test 2** (Secondary Test)| 40 | 541.0 | 788.8 | 964.5 | 1032.0 ± 308.6 | 1,223.8 | 1,935.0 | +0.65 |
| **Global Pool** | **124** | **148.0** | **498.8** | **736.5** | **801.6 ± 379.7** | **946.5** | **2,237.0** | **+1.41** |

![Cycle Life Boxplots](file:///c:/Users/HP/OneDrive/Documents/BatteryMind/reports/figures/02_cycle_life_boxplots.png)

### Key Distribution Insights:
1. **Right-Skewed Distribution:** Most cells have cycle lives between 450 and 950 cycles, with a long tail extending beyond 2,000 cycles.
2. **Distribution Shift in Test 2:** Cells in Test 2 have a substantially higher median (964.5 cycles) than Train (527 cycles) and Test 1 (561 cycles), reflecting different charging policy distributions in the third experimental batch. This provides a rigorous test for out-of-distribution generalization.
3. **Log-Transformation Necessity:** Modeling in $\log_{10}(y)$ reduces skewness from $+1.41$ to $+0.32$, stabilizing homoscedasticity for linear surrogate models and Gaussian Processes.

---

## 3. Candidate Charging Policy Parameter Space

The file `data/raw/policies_all.csv` defines 224 candidate experimental protocols. Each candidate policy consists of a 4-stage fast-charging current profile:

$$\mathbf{C} = [C_1, C_2, C_3, C_4]$$

![Candidate Policies](file:///c:/Users/HP/OneDrive/Documents/BatteryMind/reports/figures/03_charging_policies_distributions.png)

### Current Distributions Across the 4 Stages:
- **$C_1$ (0% to 20% SOC):** Mean = $6.08\text{C}$ (Range: $3.60\text{C} \to 8.00\text{C}$). High initial current when internal resistance and anode potential are favorable.
- **$C_2$ (20% to 40% SOC):** Mean = $5.54\text{C}$ (Range: $3.60\text{C} \to 7.00\text{C}$).
- **$C_3$ (40% to 60% SOC):** Mean = $4.84\text{C}$ (Range: $3.60\text{C} \to 5.60\text{C}$). Stepped down to avoid lithium plating as overpotential rises.
- **$C_4$ (60% to 80% SOC):** Mean = $3.86\text{C}$ (Range: $2.58\text{C} \to 4.80\text{C}$). Conservative taper before constant voltage charging.

---

## 4. Understanding $Q_{dlin}$ and Early-Cycle Curve Dynamics

### What is $Q_{dlin}(V)$?
$Q_{dlin}(V)$ is the linearly interpolated discharge capacity ($Ah$) sampled along a uniform 1,000-point discharge voltage grid:

$$V_{dlin} = \text{linspace}(3.6\text{ V}, 2.0\text{ V}, 1000), \quad \Delta V \approx 0.0016\text{ V}$$

- **Voltage Range:** $3.6\text{ V}$ (fully charged cell starting discharge) down to $2.0\text{ V}$ (discharge cutoff limit).
- **Cycle Range:** Columns correspond to early cycles from **Cycle 2 to Cycle 100** (99 cycles total). Cycle 1 is excluded because it is an initial formation cycle.
- **Value Semantics:** At $3.6\text{ V}$, $Q_{dlin} \approx 0\text{ Ah}$. As voltage drops to $2.0\text{ V}$, $Q_{dlin}$ reaches the cumulative discharge capacity of that cycle (nominal $1.1\text{ Ah}$, typically $\sim 1.05\text{--}1.07\text{ Ah}$).

![Qdlin Curves](file:///c:/Users/HP/OneDrive/Documents/BatteryMind/reports/figures/04_qdlin_curves_sample_cells.png)

### Consistency & Data Integrity Verification:
- **Grid Consistency:** All 124 cells share the identical 1,000-point voltage grid and the identical 99-cycle span.
- **No Inconsistencies or Missing Points:** Zero NaNs, zero infs, and zero missing cycles across all 124 cells.
- **No Additional Interpolation Needed:** The raw sensor measurements have already been standardized onto this grid by the original authors.

---

## 5. Early Degradation Signatures: High vs. Low Cycle Life

The physical insight discovered by Severson et al. is that subtle thermodynamic and kinetic changes occurring within the first 100 cycles manifest as non-uniform shifts in the discharge voltage curve:

$$\Delta Q_{100-10}(V) = Q_{dlin}(V, \text{cycle } 100) - Q_{dlin}(V, \text{cycle } 10)$$

![High vs Low Life Delta Q](file:///c:/Users/HP/OneDrive/Documents/BatteryMind/reports/figures/05_delta_q_high_vs_low_life.png)

### Observed Degradation Phenotypes:
1. **Long-Lived Cells ($> 1,200$ cycles):**
   - $\Delta Q_{100-10}(V)$ remains almost perfectly flat and close to zero across the entire voltage range.
   - Variance of $\Delta Q_{100-10}$ is tiny ($\sim 10^{-5}$ to $10^{-6}\text{ Ah}^2$).
   - Minimal shift in the two-phase electrochemical plateau ($3.2\text{ V} \to 2.8\text{ V}$).
2. **Short-Lived Cells ($< 500$ cycles):**
   - $\Delta Q_{100-10}(V)$ shows massive downward curvature, plunging to $-0.06\text{ Ah}$ or lower in the $2.7\text{ V} \to 2.1\text{ V}$ region.
   - Variance of $\Delta Q_{100-10}$ is 100 to 1,000 times larger ($\sim 10^{-3}\text{ Ah}^2$).
   - Reflects rapid loss of cyclable lithium inventory and accelerated solid electrolyte interphase (SEI) growth.

![Early Capacity Fade Trajectories](file:///c:/Users/HP/OneDrive/Documents/BatteryMind/reports/figures/06_early_cycle_degradation_trajectory.png)

---

## 6. Correlation Analysis & Candidate Features

![Correlations](file:///c:/Users/HP/OneDrive/Documents/BatteryMind/reports/figures/07_early_prediction_correlations.png)

### Correlation Matrix with Cycle Life ($N = 124$ cells):

| Candidate Metric | Formula | Pearson $r$ with $y$ | Pearson $r$ with $\log_{10}(y)$ | Physical Meaning |
| :--- | :--- | :---: | :---: | :--- |
| **$\log_{10}(\mathrm{Var}(\Delta Q_{100-10}))$** | $\log_{10}\left(\mathrm{Var}_V[Q_{100}(V) - Q_{10}(V)]\right)$ | **-0.890** | **-0.927** | Disruption of voltage plateau; strongest single predictor. |
| **$\min(\Delta Q_{100-10})$** | $\min_V [Q_{100}(V) - Q_{10}(V)]$ | **+0.737** | **+0.866** | Peak localized capacity fade along discharge curve. |
| **$\mathrm{Mean}(\Delta Q_{100-10})$** | $\frac{1}{1000} \sum_V \Delta Q_{100-10}(V)$ | **+0.698** | **+0.835** | Average capacity loss across all discharge potentials. |
| **Early Capacity Fade** | $Q_{100}(2.0\text{V}) - Q_{10}(2.0\text{V})$ | **+0.499** | **+0.676** | Total capacity lost between cycle 10 and cycle 100. |
| **Early Degradation Slope**| $\frac{Q_{100}(2.0\text{V}) - Q_{10}(2.0\text{V})}{90}$ | **+0.499** | **+0.676** | Linear rate of capacity fade per cycle. |
| **$Q_{100}$ Discharge Capacity** | $Q_{100}(2.0\text{V})$ | **+0.345** | **+0.477** | Raw remaining capacity at cycle 100 alone (weak predictor). |

> [!IMPORTANT]
> Raw capacity at cycle 100 ($Q_{100}$) alone has a relatively weak correlation ($r = 0.477$), whereas the **differential curve shape variance** $\Delta Q_{100-10}(V)$ achieves $r = -0.927$. This proves that **how the discharge curve evolves** is far more informative than simply looking at scalar capacity loss.

---

## 7. Outlier Analysis & Notable Cells

### 1. `test1_cell22` (Premature Failure Outlier)
- **Cycle Life:** 148 cycles (lowest in entire dataset).
- **Behavior:** By Cycle 100, its capacity dropped to $0.949\text{ Ah}$ (normal cells remain $>1.04\text{ Ah}$).
- **Literature Context:** In Severson et al. and Attia et al., this cell was cycled under an aggressive fast-charging protocol that induced early lithium plating and rapid impedance rise. In replication code by Peter Attia, it is flagged as an outlier (`data_test1_mod = np.delete(data_test1_mod, 21, axis=0)`).
- **BatteryMind Strategy:** We retain `test1_cell22` in the raw dataset but explicitly benchmark model robustness with and without this extreme point.

### 2. `test2_cell35` (Long-Life High-Rate Outlier)
- **Cycle Life:** 1,935 cycles.
- **Behavior:** Despite being in Test 2 with an aggressive protocol, it showed minimal plateau shift ($\log_{10}(\mathrm{Var}) = -4.34$), surviving far longer than expected. Demonstrates protocol resilience.
