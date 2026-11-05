# BatteryMind — Dataset Investigation & Comparison (Phase 0)

This document provides a systematic evaluation of public, peer-reviewed battery experimental datasets for **BatteryMind: Small-Data Battery Performance Optimization using Active Learning**.

---

## 1. Candidate Dataset Evaluation

### Candidate 1: MIT-Stanford Fast-Charging Battery Dataset
- **Full Citation:** Severson, K.A., Attia, P.M., Jin, N., Perkins, N., Jiang, B., Yang, Z., Chen, M.H., Aykol, M., Herring, P.K., Fraggedakis, D., Braatz, R.D., Chueh, W.C. *"Data-driven prediction of battery cycle life before capacity degradation."* Nature Energy 4, 383–391 (2019). Followed by Attia et al., *"Closed-loop optimization of fast-charging protocols for batteries with machine learning."* Nature 578, 397–402 (2020).
- **Source:** Toyota Research Institute (TRI), Massachusetts Institute of Technology (MIT), Stanford University. Hosted on [data.matr.io](https://data.matr.io) and public AWS S3 / Zenodo mirrors.
- **Number of Batteries/Cells:** 124 cells (batch 1, 2, 3) across 72 distinct multi-step fast-charging protocols. (Extended to 224+ cells with Attia 2020 closed-loop cohorts).
- **Main Measurements:** Cycle number, cell voltage, current, charge capacity, discharge capacity, internal resistance ($IR$), cell can temperature, timestamp.
- **Experimental Conditions:** Controlled thermal chamber (30°C). Cells subjected to distinct multi-step fast-charging policies (e.g., $C_1$-$\%SOC$-$C_2$, varying from 3.6C up to 6C in the first step, charging to 80% SOC in ~10 minutes, followed by 1C CC-CV finish). Uniform 4C discharge across all cells.
- **Target Variables Available:**
  - Total Battery Cycle Life (cycles until 80% nominal capacity retention, ranging from ~150 to ~2,300 cycles).
  - Capacity fade trajectory ($Q(n)$ curve).
  - Internal resistance evolution.
- **Approximate Dataset Size:**
  - Full raw time series: ~1.5 GB to 3.5 GB (JSON / MATLAB formats).
  - Cycle-level summary / feature tables: **~10 MB to 25 MB** (CSV / Parquet).
- **Suitability for Small-Data Experiments:** **Exceptional.** The dataset maps each distinct cell to an experimental protocol vector and early-cycle summary statistics. One can seed an initial model with $N = 10\text{--}20$ cells and test generalizability.
- **Support for Active Learning / Experiment Selection:** **Native / Benchmark Gold Standard.** This dataset was specifically created to solve experimental optimization via machine learning. The experimental design space consists of parameterized charging policies, making it directly suited for candidate pool evaluation, uncertainty quantification, and sequential policy recommendation.
- **License / Access Information:** Open Access under Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Advantages:**
  - High protocol diversity (72 distinct policies) providing meaningful parameter variations for active learning.
  - Standardized cell chemistry (commercial A123 Systems APR18650M1A $1.1\text{ Ah}$ $\text{LiFePO}_4$ / graphite).
  - Well-established benchmark in the scientific literature; pre-extracted early-cycle features ($\Delta Q_{100-10}$) exist and are reproducible.
  - Wide range of cycle lives (150 to 2,300 cycles) creates rich variance for regression and optimization.
- **Limitations:**
  - Raw multi-cycle sensor files require extraction if processing from scratch (mitigated by using cycle summary extracts).
  - Fixed discharge protocol (focuses strictly on charge optimization, not drive cycles).

---

### Candidate 2: NASA PCoE Battery Aging Dataset
- **Full Citation:** Saxena, S., Celaya, J., Saha, B., Saha, S., Goebel, K. *"Accelerated Aging Dataset for Lithium-ion Cells."* NASA Ames Prognostics Center of Excellence (PCoE) Data Repository (2007–2010).
- **Source:** NASA Ames Research Center Prognostics Data Repository.
- **Number of Batteries/Cells:** ~36 commercial 18650 cells organized into small batches (e.g., Set 1: B0005, B0006, B0007, B0018; Set 6: B0045, B0046, etc.).
- **Main Measurements:** Cycle number, operational mode (charge, discharge, EIS impedance), voltage, current, battery temperature, discharge capacity, charge time, impedance parameters.
- **Experimental Conditions:** 
  - Charged under standard CC-CV at room temperature (4.2V cutoff, 1.5A).
  - Discharged at constant currents (1A, 2A, 4A) or randomized loads until voltage cutoffs (2.7V, 2.5V, 2.2V) under various ambient temperatures (4°C, 24°C, 43°C).
- **Target Variables Available:**
  - Discharge Capacity per cycle ($Ah$).
  - Remaining Useful Life (RUL) cycles to 70% or 80% nominal capacity (typically ~100 to 200 cycles).
  - Real and imaginary impedance parameters.
- **Approximate Dataset Size:** ~50 MB to 200 MB (.mat / converted .csv). Very lightweight.
- **Suitability for Small-Data Experiments:** **High.** Each batch consists of 3 to 4 cells, making it a classic small-sample benchmark.
- **Support for Active Learning / Experiment Selection:** **Low to Moderate.** While each cell has complete degradation traces, there are very few cells per unique condition (often only 1 or 2 cells under a specific temperature/current setting). As a result, the candidate pool for *selecting the next experimental policy* is extremely small ($<10$ distinct conditions), making active learning selection artificially constrained.
- **License / Access Information:** Public domain (NASA Open Data Policy).
- **Advantages:**
  - Simple structure and lightweight download.
  - Very popular pedagogical benchmark for basic Remaining Useful Life (RUL) regression.
- **Limitations:**
  - Very small cell count per test matrix; lacks the design-space breadth required for a realistic pool-based active learning loop.
  - Known capacity regeneration anomalies and sensor noise across aging stages.

---

### Candidate 3: CALCE Battery Research Group Dataset
- **Full Citation:** Center for Advanced Life Cycle Engineering (CALCE), University of Maryland. *Battery Aging and Cycling Data Sets (CS2, CX2, PL series)* (Pecht et al., 2011–2018).
- **Source:** University of Maryland CALCE Battery Research Group.
- **Number of Batteries/Cells:** ~50+ prismatic and cylindrical cells (e.g., CS2 series: CS2_33, CS2_34, CS2_35, etc.).
- **Main Measurements:** Timestamp, test time, cycle count, current, voltage, charge/discharge capacity, temperature.
- **Experimental Conditions:** Cycling at diverse charge rates (0.5C, 1C) and discharge rates (0.5C to 3C), with partial cycling and constant current charging.
- **Target Variables Available:**
  - Cycle life to 80% capacity.
  - Capacity decay trajectories.
- **Approximate Dataset Size:** ~500 MB to 2 GB (Excel / CSV files).
- **Suitability for Small-Data Experiments:** **Moderate.** Cells can be analyzed individually or in small clusters.
- **Support for Active Learning / Experiment Selection:** **Moderate.** Variations exist in cycling rates, but experimental conditions across batches were not generated systematically for design-space exploration.
- **License / Access Information:** Public research access (free for academic research).
- **Advantages:**
  - Rich real-world degradation profiles with clear capacity fade curves.
  - High-quality laboratory cycling data.
- **Limitations:**
  - Inconsistent file formats across different series (CS2, CX2, etc.), requiring significant custom parsing.
  - Metadata describing the exact protocol space is fragmented across papers and test logs.

---

### Candidate 4: Oxford Battery Degradation Dataset
- **Full Citation:** Howey, D.A. et al. *"Oxford Battery Degradation Dataset 1."* Department of Engineering Science, University of Oxford (2016).
- **Source:** Oxford Energy and Power Group (EPG).
- **Number of Batteries/Cells:** 8 Kokam 740 mAh lithium-ion pouch cells.
- **Main Measurements:** Voltage, current, temperature, capacity, continuous Electrochemical Impedance Spectroscopy (EIS) frequency sweeps.
- **Experimental Conditions:** Cycled at a fixed ambient temperature (40°C) with drive-cycle dynamic discharging (Artemis urban drive cycle) and periodic characterization.
- **Target Variables Available:**
  - Capacity degradation ($Ah$).
  - Impedance increase over equivalent drive-cycle distance.
- **Approximate Dataset Size:** ~450 MB (.mat files).
- **Suitability for Small-Data Experiments:** **High** in terms of small $N$ ($N=8$), but sample diversity is near zero.
- **Support for Active Learning / Experiment Selection:** **Unsuitable.** Because all 8 cells were subjected to virtually the identical test profile at 40°C, there is no experimental candidate space to select or optimize between.
- **License / Access Information:** Academic Open Access.
- **Advantages:**
  - High-precision EIS data alongside degradation measurements.
- **Limitations:**
  - Only 8 cells and a single thermal/operating condition; completely unsuitable for active learning experiment recommendation.

---

## 2. Comparative Matrix

| Evaluation Criteria | MIT-Stanford Fast-Charging (Severson et al.) | NASA PCoE Aging Dataset | CALCE Dataset (CS2/CX2) | Oxford Degradation Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Source** | MIT / Stanford / TRI (Nature Energy 2019) | NASA Ames Research Center | Univ. of Maryland CALCE | Univ. of Oxford EPG |
| **Cell Chemistry / Form** | Commercial A123 18650 $\text{LiFePO}_4$ | Commercial 18650 $\text{LiCoO}_2$ | Prismatic / Cylindrical $\text{LiCoO}_2$ | Kokam pouch $\text{LiCoO}_2$ |
| **Number of Tested Cells** | **124** (72 distinct policies) | ~36 (in 3-4 cell batches) | ~50+ across disparate series | 8 cells |
| **Target Variable** | **Cycle Life (EOL to 80%)** | Discharge Capacity / RUL | Cycle Life / Capacity Fade | Capacity Fade / Impedance |
| **Range of Target** | **150 to ~2,300 cycles** | ~80 to 200 cycles | ~300 to 1,200 cycles | ~40 to 100 days cycling |
| **Condition Diversity** | **High** (multi-step charging current, %SOC) | Low (fixed current/temp steps) | Moderate (varying C-rates) | Negligible (uniform drive cycle) |
| **Active Learning Suitability** | **⭐⭐⭐⭐⭐ (Native fit)** | ⭐⭐ (Pool too small) | ⭐⭐⭐ (Fragmented space) | ⭐ (No condition pool) |
| **Small-Data Regime Fit** | **⭐⭐⭐⭐⭐ (Ideal candidate pool)** | ⭐⭐⭐⭐ (Small, but limited pool) | ⭐⭐⭐ | ⭐⭐ |
| **Metadata Standardization** | **High** (well-documented tabular schema) | Medium (scattered batch notes) | Medium/Low (multiple formats) | High |
| **License** | Open Access (CC BY 4.0) | Public Domain (NASA Open Data) | Academic Research Use | Open Academic License |

---

## 3. Primary Dataset Recommendation

### Selected Dataset: **MIT-Stanford Fast-Charging Battery Dataset (Severson et al., 2019 / Attia et al., 2020)**

### Why This Dataset is Selected for BatteryMind:
1. **Direct Alignment with Active Learning:**
   Active learning requires an **unlabelled candidate pool** of distinct experimental conditions to evaluate and recommend. The MIT-Stanford dataset offers **72 distinct charging protocols across 124 cells**, providing a realistic, diverse experimental landscape.
2. **Realistic Small-Data Scenario:**
   In real battery development, testing 124 cells to failure takes over a year of continuous cycler operation. By starting BatteryMind on an initial seed batch of just **15 to 20 cells**, we mirror real-world lab constraints and demonstrate how active learning steers the selection of the remaining cells.
3. **Pre-extracted Feature Feasibility:**
   Rather than wrestling with millions of raw temporal measurements, the scientific community has established compact, physics-grounded early-cycle summary features (such as $\Delta Q_{100-10}(V)$ variance, internal resistance slope, and charge time). This allows BatteryMind to remain **lightweight, clean, and beginner-friendly** without large compute overheads.
4. **Target Clarity:**
   Cycle life to 80% nominal capacity is the recognized gold standard metric for battery longevity. It provides a continuous, highly variable target ($150 \text{ to } 2,300 \text{ cycles}$) ideal for uncertainty-aware regression models.
5. **Open, Reproducible Science:**
   The dataset is governed by the permissive CC BY 4.0 license and is widely cited, ensuring complete reproducibility for our research-oriented project.
