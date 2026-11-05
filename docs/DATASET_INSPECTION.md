# BatteryMind — Dataset Inspection Report (Phase 1)

---

## 1. Source & Provenance
- **Primary Scientific Publication:** 
  - K.A. Severson, P.M. Attia, N. Jin, N. Perkins, B. Jiang, Z. Yang, M.H. Chen, M. Aykol, P.K. Herring, D. Fraggedakis, R.D. Braatz, W.C. Chueh. *"Data-driven prediction of battery cycle life before capacity degradation."* *Nature Energy* 4, 383–391 (2019).
  - P.M. Attia, A. Grover, N. Jin, K.A. Severson, T.M. Markov, Y.H. Liao, M.H. Chen, B. Cheong, N. Perkins, Z. Yang, P.K. Herring, M. Aykol, S.J. Harris, R.D. Braatz, S. Ermon, W.C. Chueh. *"Closed-loop optimization of fast-charging protocols for batteries with machine learning."* *Nature* 578, 397–402 (2020).
- **Authoritative Data Repository:** 
  - Toyota Research Institute (TRI) / Materials Data Facility (MDF): [data.matr.io/1/projects/5c48dd2bc625d700019f3204](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204).
- **Lead Computational Author Repositories:**
  - `petermattia/revisit-severson-et-al` (Dr. Peter M. Attia, Stanford University).
  - `chueh-ermon/battery-fast-charging-optimization` (Chueh and Ermon Labs, Stanford University / TRI).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 2. Download Information & Storage Architecture
The dataset was retrieved directly from the authoritative scientific sources and archived in `data/raw/` without modification:

| Component | Source / File Reference | Local Destination | Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Repository Metadata** | Girder API (`data.matr.io`) | `data/raw/matr_dataset_metadata.json` | 1.23 KB | Authoritative persistent IDs, hashes, and endpoints for the 3 raw batch files (7.70 GB total). |
| **Ground Truth Cycle Lives** | Peter Attia / Stanford | `data/raw/cycle_lives/` | 525 Bytes (3 files) | Ground-truth cycle life targets for train (41), test1 (43), and test2 (40). |
| **Candidate Experimental Space** | Chueh–Ermon / Stanford | `data/raw/policies_all.csv` | 5.25 KB | 224 candidate 4-step fast-charging experimental protocols ($C_1, C_2, C_3, C_4$). |
| **Cell Curve Series (Train)** | Peter Attia / Stanford | `data/raw/train/cell[1-41].csv` | 30.15 MB (41 files) | Early-cycle $\Delta Q(V)$ degradation curves (cycles 2–100, 1000 voltage points) for training cells. |
| **Cell Curve Series (Test 1)** | Peter Attia / Stanford | `data/raw/test1/cell[1-43].csv` | 31.63 MB (43 files) | Early-cycle $\Delta Q(V)$ degradation curves for primary test cohort. |
| **Cell Curve Series (Test 2)** | Peter Attia / Stanford | `data/raw/test2/cell[1-40].csv` | 29.33 MB (40 files) | Early-cycle $\Delta Q(V)$ degradation curves for secondary test cohort. |
| **Total Raw Footprint** | Verified Open Science Repositories | `data/raw/` | **91.12 MB (129 files)** | Clean, complete, uncompressed raw experimental series. |

---

## 3. Dataset Structure & Dimensions

```
data/raw/
├── matr_dataset_metadata.json    # Verified data.matr.io registry for all 3 batches (7.70 GB raw origin)
├── policies_all.csv              # 224 candidate experimental protocols (4 C-rate steps)
├── cycle_lives/
│   ├── train_cycle_lives.csv     # 41 cell targets (one integer per line)
│   ├── test1_cycle_lives.csv     # 43 cell targets (one integer per line)
│   └── test2_cycle_lives.csv     # 40 cell targets (one integer per line)
├── train/                        # 41 cell curve matrices: cell1.csv .. cell41.csv
├── test1/                        # 43 cell curve matrices: cell1.csv .. cell43.csv
└── test2/                        # 40 cell curve matrices: cell1.csv .. cell40.csv
```

### Dimensionality of Individual Cell Curve Matrices:
- **Matrix Dimension:** $(1000, 99)$
  - **1,000 Rows:** Linearly interpolated discharge capacity measurements across a fixed voltage window ($V_{dlin} = 3.6\text{ V} \to 2.0\text{ V}$).
  - **99 Columns:** Early degradation cycles, spanning **Cycle 2 to Cycle 100** (Cycle 1 is reserved for initial formation/conditioning).
- **Data Type:** `float64`
- **Total cells across all partitions:** **124 cells** ($41 + 43 + 40 = 124$).

---

## 4. Target Variable Inspection & Availability

### Target Definition:
- **Target Variable ($y$):** **Battery Cycle Life** — defined as the cycle number at which the cell's discharge capacity fades below **80% of its nominal capacity** ($0.88\text{ Ah}$ for these $1.1\text{ Ah}$ commercial A123 Systems APR18650M1A $\text{LiFePO}_4$/graphite cells).
- **Data Availability:** **100% available** across all 124 cells (0 missing values, 0 NaNs).
- **Target Data Type:** Non-negative integer (`int64`).

### Statistical Summary of Target ($y$):

| Cohort | Cell Count | Minimum | 25th % | Median | Mean ± Std | 75th % | Maximum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 41 | 300.0 | 461.0 | 527.0 | 673.8 ± 327.1 | 788.0 | 2,160.0 |
| **Test 1** | 43 | 148.0 | 455.5 | 561.0 | 709.3 ± 395.8 | 845.5 | 2,237.0 |
| **Test 2** | 40 | 541.0 | 788.8 | 964.5 | 1032.0 ± 308.6 | 1,223.8 | 1,935.0 |
| **Global Pool** | **124** | **148.0** | **498.8** | **736.5** | **801.6 ± 379.7** | **946.5** | **2,237.0** |

---

## 5. Candidate Experimental Variables & Protocol Space
The file `data/raw/policies_all.csv` defines the 224 candidate experimental conditions explored in the fast-charging parameter space. Each candidate experiment is characterized by a 4-step multi-stage fast-charging current policy:

$$\text{Protocol} = [C_1, C_2, C_3, C_4]$$

- **$C_1$ (Initial Current):** $3.60\text{C}$ to $8.00\text{C}$ (mean $6.08\text{C}$, 9 unique levels).
- **$C_2$ (Step 2 Current):** $3.60\text{C}$ to $7.00\text{C}$ (mean $5.54\text{C}$, 8 unique levels).
- **$C_3$ (Step 3 Current):** $3.60\text{C}$ to $5.60\text{C}$ (mean $4.84\text{C}$, 6 unique levels).
- **$C_4$ (Step 4 Current):** $2.58\text{C}$ to $4.80\text{C}$ (mean $3.86\text{C}$, 75 unique levels).
- **Missing Protocol Data:** **0%** (all 224 protocols are fully specified).

---

## 6. Sanity Checks & Quality Audit Results

All 7 mandatory data quality checks were executed by [`src/inspect_dataset.py`](file:///c:/Users/HP/OneDrive/Documents/BatteryMind/src/inspect_dataset.py):

| Check | Criterion | Result | Status |
| :--- | :--- | :--- | :---: |
| **1. Duplicate IDs** | No identical cell identifiers | 0 duplicates | ✅ PASSED |
| **2. Impossible Cycle Counts** | $0 < \text{Cycle Life} \le 10,000$ | Min = 148, Max = 2,237 | ✅ PASSED |
| **3. Negative Values** | Non-negative targets and physical currents | All $\ge 0$ | ✅ PASSED |
| **4. Missing Targets** | No NaNs, nulls, or empty lines | 0 missing | ✅ PASSED |
| **5. Consistency** | 124 curve CSVs exactly match 124 target values | Exact match (124/124) | ✅ PASSED |
| **6. File Corruption** | All files non-zero byte, valid numerical arrays | 0 corrupted files | ✅ PASSED |
| **7. Policy Bounds** | Charging currents within physical cycler bounds | $2.58\text{C} \le C \le 8.00\text{C}$ | ✅ PASSED |

---

## 7. Critical Observations & Methodological Considerations

### Potential Data Leakage Risks:
1. **Cycle Window Boundary:** All input feature extraction must strictly use data from **cycles $\le 100$**. Using summary statistics computed from cycles past 100 would constitute severe forward-looking leakage.
2. **Standardization / Scaling Leakage:** In Phase 2+, feature normalization (e.g., mean/variance or min-max scaling) must be computed **strictly on the initial seed training pool** ($N \le 20$), never globally across the unlabelled candidate pool.

### Observations Relevant to Small-Data Modeling:
- **Skewed Target Distribution:** The target variable spans over an order of magnitude ($148$ to $2,237$ cycles) with positive skewness. Modeling in $\log_{10}(\text{Cycle Life})$ space (as established by Severson and Attia) will significantly stabilize variance and linearize relationships.
- **Sample Scarcity vs. Dimensionality:** With an initial seed pool of $N = 15\text{--}20$ cells, high-dimensional models with hundreds of free parameters will overfit. Feature selection must produce a compact, physics-grounded representation ($D \le 6\text{--}8$ features).

### Observations Relevant to Future Active Learning:
- **Rich Candidate Pool:** The 124 cells provide a realistic testbed for pool-based active learning. We can designate 15 cells as the initial "seed experiments" and the remaining 109 cells as the "unlabelled candidate experiment pool".
- **Meaningful Diversity:** Because the cells were cycled under diverse fast-charging policies, the active learning acquisition function will be forced to balance **exploration** (selecting protocols in sparsely sampled parameter regions with high uncertainty) against **exploitation** (selecting protocols predicted to achieve long cycle life).
