"""
BatteryMind — Small-Data Battery Performance Optimization using Active Learning
Interactive Streamlit Research Dashboard (Phase 5)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="BatteryMind — Active Learning Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants & Paths
PROCESSED_DATA_PATH = os.path.join("data", "processed", "batterymind_features.csv")
AL_RESULTS_PATH = os.path.join("reports", "active_learning_results.csv")
BASELINE_RESULTS_PATH = os.path.join("reports", "baseline_results.csv")
RAW_DATA_DIR = os.path.join("data", "raw")
FEATURE_COLS = ['log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cycle100', 'fade_100_10']


# ==============================================================================
# CACHED DATA & MODEL LOADERS
# ==============================================================================

@st.cache_data
def load_processed_features():
    """Load the 124-cell processed feature dataset."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        return None
    return pd.read_csv(PROCESSED_DATA_PATH)


@st.cache_data
def load_al_results():
    """Load the Phase 4 active learning experimental trajectories."""
    if not os.path.exists(AL_RESULTS_PATH):
        return None
    return pd.read_csv(AL_RESULTS_PATH)


@st.cache_data
def load_baseline_results():
    """Load the Phase 3 baseline benchmark records."""
    if not os.path.exists(BASELINE_RESULTS_PATH):
        return None
    return pd.read_csv(BASELINE_RESULTS_PATH)


@st.cache_data
def load_cell_curve(dataset_split, cell_id):
    """Load the raw 1000x99 early-cycle curve matrix for a cell."""
    # cell_id is like 'train_cell1' or 'test1_cell2'
    try:
        parts = cell_id.split('_')
        split = parts[0]
        cell_num = parts[1].replace('cell', '')
        file_path = os.path.join(RAW_DATA_DIR, split, f"cell{cell_num}.csv")
        if os.path.exists(file_path):
            return np.genfromtxt(file_path, delimiter=',')
    except Exception:
        pass
    return None


@st.cache_resource
def get_trained_gp_model():
    """
    Trains and caches the primary Gaussian Process Regressor
    on all 41 designated training cells for user interactive inference.
    """
    df = load_processed_features()
    if df is None:
        return None, None
        
    train_df = df[df['dataset_split'] == 'train'].reset_index(drop=True)
    X_train = train_df[FEATURE_COLS].values
    z_train = np.log10(train_df['cycle_life'].values)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    kernel = (
        C(1.0, (1e-3, 1e3)) * 
        Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + 
        WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        n_restarts_optimizer=2,
        random_state=42
    )
    gp.fit(X_train_scaled, z_train)
    return scaler, gp


# ==============================================================================
# UI COMPONENTS & PAGE RENDERERS
# ==============================================================================

def render_sidebar():
    """Sidebar navigation and project meta."""
    st.sidebar.title("⚡ BatteryMind")
    st.sidebar.caption("Small-Data Battery Optimization with Active Learning")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Cycle-Life Predictor",
            "Active Learning Simulator",
            "Battery Explorer",
            "Strategy Comparison",
            "Methodology"
        ]
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        """
        **System Specs:**
        - **Cell Form:** Commercial A123 18650 LFP
        - **Total Pool:** 124 cells
        - **Early Horizon:** Cycle 100 ($\le 15\%$ life)
        - **Surrogate:** Gaussian Process (Matérn)
        """
    )
    st.sidebar.info("🔬 Research-oriented benchmark with strict zero-leakage protocol.")
    return page


# ------------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# ------------------------------------------------------------------------------
def page_overview():
    st.title("⚡ BatteryMind — Overview")
    st.markdown(
        """
        ### Small-Data Battery Performance Optimization using Active Learning
        Evaluating the lifetime of lithium-ion batteries typically takes **months or years** per cell across hundreds of cycles.
        In laboratory R&D, scientists operate under a **small-data constraint** ($N \le 20$ cells) and must systematically determine 
        **which experiment to run next**.
        
        **BatteryMind** demonstrates how physics-grounded early-cycle signatures coupled with Gaussian Process uncertainty quantification 
        and adaptive acquisition functions drastically accelerate optimal cell discovery while minimizing wasted experimental cycles.
        """
    )
    
    st.divider()
    st.subheader("Key Quantitative Benchmarks")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cells Analyzed", "124 Cells", "41 Train | 43 Test1 | 40 Test2")
    with col2:
        st.metric("Early-Cycle Predictor", "N = 100 Cycles", "5–15% of cell lifespan")
    with col3:
        st.metric("Baseline Model Accuracy", "R² = 0.9052", "MAE = 81.9 cyc (N=20)")
    with col4:
        st.metric("Discovery Speedup", "15.4 vs 19–31 Cells", "Global Max 2,160 cyc found")
        
    st.divider()
    
    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.subheader("The Active Learning Loop")
        st.markdown(
            """
            Instead of non-adaptively testing random battery cells, BatteryMind executes a closed feedback cycle:
            1. **Small Seed Experiment:** Train on a tiny batch of 15–20 completed cells.
            2. **Uncertainty Quantification:** The Gaussian Process outputs predictive mean $\mu(x)$ and epistemic uncertainty $\sigma(x)$.
            3. **Acquisition Function Scoring:** Evaluate candidate battery conditions using **Upper Confidence Bound (UCB)** or **Expected Improvement (EI)**.
            4. **Next Best Experiment:** Select the single most informative candidate, run it to failure, reveal the ground truth, and update model beliefs.
            """
        )
    with c_right:
        st.subheader("Core Methodological Safeguards")
        st.success("✅ **Zero Leakage:** Strictly no future cycles ($>100$) queried.")
        st.success("✅ **Split Discipline:** Test 1 and Test 2 untouched during acquisition.")
        st.success("✅ **Standardization Isolation:** Feature scaling fit strictly on labeled pool.")
        st.info("ℹ️ **Simulated/Retrospective:** Benchmark performed on verified scientific dataset.")


# ------------------------------------------------------------------------------
# PAGE 2: CYCLE-LIFE PREDICTOR
# ------------------------------------------------------------------------------
def page_predictor():
    st.title("🎯 Cycle-Life & Uncertainty Predictor")
    st.markdown(
        """
        Estimate total battery cycle life (cycles until 80% capacity retention) and evaluate **model epistemic uncertainty**
        using only the first 100 electrochemical discharge cycles.
        """
    )
    
    scaler, gp_model = get_trained_gp_model()
    if gp_model is None:
        st.error("Error: Processed dataset or model artifact not found. Please ensure `data/processed/batterymind_features.csv` exists.")
        return
        
    # Preset Profiles
    st.subheader("1. Quick Presets or Custom Inputs")
    p_col1, p_col2, p_col3 = st.columns(3)
    preset_choice = None
    if p_col1.button("Preset: Long-Lived Battery (~2,000+ cyc)"):
        preset_choice = (-5.01, -0.015, -0.005, 1.065, -0.001)
    if p_col2.button("Preset: Typical Cell (~750 cyc)"):
        preset_choice = (-3.85, -0.040, -0.019, 1.050, -0.010)
    if p_col3.button("Preset: Rapid Degradation (~300 cyc)"):
        preset_choice = (-2.75, -0.130, -0.075, 0.996, -0.060)
        
    # Input controls
    st.markdown("Adjust the 5 verified early degradation signatures:")
    col1, col2 = st.columns(2)
    
    with col1:
        v_logvar = st.slider(
            "log10(Var(ΔQ100-10)) — Voltage Plateau Variance",
            min_value=-5.50, max_value=-2.50,
            value=preset_choice[0] if preset_choice else -3.85,
            step=0.01,
            help="Log-variance of the differential discharge curve between cycle 100 and cycle 10. Strongest degradation predictor (r = -0.927)."
        )
        v_mindq = st.slider(
            "min(ΔQ100-10) — Peak Localized Capacity Drop (Ah)",
            min_value=-0.200, max_value=0.000,
            value=preset_choice[1] if preset_choice else -0.040,
            step=0.005,
            help="Minimum point of the ΔQ100-10 curve along the two-phase transition."
        )
        v_meandq = st.slider(
            "mean(ΔQ100-10) — Integrated Downward Curve Shift (Ah)",
            min_value=-0.100, max_value=0.000,
            value=preset_choice[2] if preset_choice else -0.019,
            step=0.002,
            help="Average downward capacity displacement across all 1,000 voltage points."
        )
        
    with col2:
        v_qc100 = st.slider(
            "Q_100 — Retained Discharge Capacity at Cycle 100 (Ah)",
            min_value=0.850, max_value=1.120,
            value=preset_choice[3] if preset_choice else 1.050,
            step=0.005,
            help="Scalar discharge capacity measured at the 2.0V cutoff of Cycle 100."
        )
        v_fade = st.slider(
            "fade_100_10 — End Capacity Fade Q100 - Q10 (Ah)",
            min_value=-0.120, max_value=0.010,
            value=preset_choice[4] if preset_choice else -0.010,
            step=0.002,
            help="Net bulk capacity lost between Cycle 10 and Cycle 100 at 2.0V cutoff."
        )

    # Perform Inference
    input_vec = np.array([[v_logvar, v_mindq, v_meandq, v_qc100, v_fade]])
    input_scaled = scaler.transform(input_vec)
    z_pred, z_std = gp_model.predict(input_scaled, return_std=True)
    
    y_pred = float(10.0 ** z_pred[0])
    std_val = float(z_std[0])
    y_lower = float(10.0 ** (z_pred[0] - 1.96 * std_val))
    y_upper = float(10.0 ** (z_pred[0] + 1.96 * std_val))
    rel_margin = (y_upper - y_lower) / (2.0 * y_pred) * 100.0
    
    # Uncertainty tier
    if rel_margin < 15.0:
        tier_label = "LOW UNCERTAINTY"
        tier_color = "normal"
    elif rel_margin <= 30.0:
        tier_label = "MEDIUM UNCERTAINTY"
        tier_color = "off"
    else:
        tier_label = "HIGH UNCERTAINTY"
        tier_color = "inverse"
        
    st.divider()
    st.subheader("2. Probabilistic Prediction Results")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Predicted Cycle Life", f"{y_pred:.0f} cycles")
    with m_col2:
        st.metric("Model Uncertainty (σ)", f"±{std_val:.4f} log₁₀ cyc", f"±{rel_margin:.1f}% physical span")
    with m_col3:
        st.metric("95% Credible Interval", f"[{y_lower:.0f}, {y_upper:.0f}] cycles")
    with m_col4:
        st.metric("Reliability Assessment", tier_label)
        
    st.caption("ℹ️ **Scientific Disclaimer:** Cycle life prediction is a probabilistic estimate derived from early electrochemical signatures under controlled 30°C conditions. It does not represent a guaranteed physical manufacturer lifetime.")


# ------------------------------------------------------------------------------
# PAGE 3: ACTIVE LEARNING SIMULATOR
# ------------------------------------------------------------------------------
def page_al_simulator():
    st.title("🔄 Active Learning Simulator")
    st.markdown(
        """
        Simulate a real laboratory testing campaign using the **41 Train cells** as the available experimental pool.
        The system chooses which candidate experiment to evaluate next, reveals its true lifetime, and retrains the model.
        """
    )
    st.warning("🔬 **Methodological Note:** Retrospective simulation using historical scientific data. No physical laboratory hardware is actuated.")
    
    al_df = load_al_results()
    if al_df is None:
        st.error("Active learning results not found. Please run `src/active_learning.py` first.")
        return
        
    # Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        seed_choice = st.selectbox("Initial Labeled Seed Size (N₀)", [10, 15, 20], index=1)
    with ctrl_col2:
        strategy_choice = st.selectbox(
            "Acquisition Strategy",
            ["EXPECTED_IMPROVEMENT", "UCB", "MAX_UNCERTAINTY", "RANDOM"]
        )
    with ctrl_col3:
        trial_choice = st.selectbox("Experimental Trial Run", [0, 1, 2, 3, 4], format_func=lambda x: f"Trial #{x+1}")
        
    # Filter trajectory data
    traj = al_df[
        (al_df['initial_N'] == seed_choice) &
        (al_df['strategy'] == strategy_choice) &
        (al_df['trial_idx'] == trial_choice)
    ].sort_values('step').reset_index(drop=True)
    
    if len(traj) == 0:
        st.info("No trajectory found for selected parameters.")
        return
        
    # Step Slider
    max_steps = len(traj) - 1
    current_step = st.slider("Simulation Step (Acquisitions Made)", 0, max_steps, min(5, max_steps))
    
    row_now = traj.iloc[current_step]
    
    st.divider()
    st.subheader(f"Status at Step {current_step} ({row_now['n_labeled']} Total Labeled Cells)")
    
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Labeled Cells Count", f"{int(row_now['n_labeled'])} / 41", f"{41 - int(row_now['n_labeled'])} candidates left")
    with sc2:
        st.metric("Selected Cell", str(row_now['selected_cell_id']))
    with sc3:
        if current_step > 0:
            st.metric("Revealed True Lifetime", f"{int(row_now['true_cycle_life'])} cycles", f"Pred: {10**row_now['pred_mean_log10']:.0f} cyc")
        else:
            st.metric("Revealed True Lifetime", "Seed Batch", "N/A")
    with sc4:
        st.metric("Best Discovered Cell", f"{int(row_now['best_cycle_life_discovered'])} cycles", "Target: 2,160 cyc")
        
    # Interactive Curves
    st.subheader("Held-Out Generalization Curves")
    tab1, tab2 = st.tabs(["Test 1 (In-Distribution) Accuracy", "Longevity Discovery Progression"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        # Plot all 4 strategies for comparison in background
        sub_all = al_df[(al_df['initial_N'] == seed_choice) & (al_df['trial_idx'] == trial_choice)]
        for strat in ["RANDOM", "MAX_UNCERTAINTY", "UCB", "EXPECTED_IMPROVEMENT"]:
            s_data = sub_all[sub_all['strategy'] == strat].sort_values('n_labeled')
            is_active = (strat == strategy_choice)
            ax.plot(
                s_data['n_labeled'], s_data['test1_mae'],
                marker='o' if is_active else None,
                linewidth=2.5 if is_active else 1.2,
                alpha=1.0 if is_active else 0.4,
                label=f"{strat} (Active)" if is_active else strat
            )
        # Highlight current point
        ax.scatter([row_now['n_labeled']], [row_now['test1_mae']], color='red', s=120, zorder=5, label='Current State')
        ax.set_title(f"Test 1 MAE vs Number of Labeled Cells (Trial #{trial_choice+1})")
        ax.set_xlabel("Total Labeled Cells")
        ax.set_ylabel("Test 1 MAE (cycles)")
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 4))
        for strat in ["RANDOM", "MAX_UNCERTAINTY", "UCB", "EXPECTED_IMPROVEMENT"]:
            s_data = sub_all[sub_all['strategy'] == strat].sort_values('n_labeled')
            is_active = (strat == strategy_choice)
            ax.plot(
                s_data['n_labeled'], s_data['best_cycle_life_discovered'],
                marker='s' if is_active else None,
                linewidth=2.5 if is_active else 1.2,
                alpha=1.0 if is_active else 0.4,
                label=f"{strat} (Active)" if is_active else strat
            )
        ax.axhline(2160, color='blue', linestyle='--', alpha=0.5, label='Global Max (2,160 cyc)')
        ax.scatter([row_now['n_labeled']], [row_now['best_cycle_life_discovered']], color='red', s=120, zorder=5)
        ax.set_title("Best Observed Cycle Life vs Number of Labeled Cells")
        ax.set_xlabel("Total Labeled Cells")
        ax.set_ylabel("Max Cycle Life Found (cycles)")
        ax.legend()
        st.pyplot(fig)
        plt.close()


# ------------------------------------------------------------------------------
# PAGE 4: BATTERY EXPLORER
# ------------------------------------------------------------------------------
def page_explorer():
    st.title("🔍 Battery Cell Explorer")
    st.markdown("Inspect individual battery cells, raw degradation curves, extracted features, and predictions.")
    
    df_features = load_processed_features()
    if df_features is None:
        st.error("Processed features file not found.")
        return
        
    col_filter, col_select = st.columns([1, 2])
    with col_filter:
        split_filter = st.selectbox("Filter by Cohort", ["All Cohorts", "train (41 cells)", "test1 (43 cells)", "test2 (40 cells)"])
    
    filtered_df = df_features.copy()
    if "train" in split_filter:
        filtered_df = filtered_df[filtered_df['dataset_split'] == 'train']
    elif "test1" in split_filter:
        filtered_df = filtered_df[filtered_df['dataset_split'] == 'test1']
    elif "test2" in split_filter:
        filtered_df = filtered_df[filtered_df['dataset_split'] == 'test2']
        
    with col_select:
        cell_options = list(filtered_df['cell_id'])
        chosen_cell_id = st.selectbox("Select Battery Cell ID", cell_options)
        
    cell_row = df_features[df_features['cell_id'] == chosen_cell_id].iloc[0]
    
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Cell Identifier", str(cell_row['cell_id']))
    with c2:
        st.metric("Cohort Partition", str(cell_row['dataset_split']).upper())
    with c3:
        st.metric("True Cycle Life", f"{int(cell_row['cycle_life'])} cycles")
    with c4:
        st.metric("log10(Var(ΔQ100-10))", f"{cell_row['log10_var_deltaQ']:.4f}")
        
    # Curve visualization
    curve_mat = load_cell_curve(cell_row['dataset_split'], cell_row['cell_id'])
    if curve_mat is not None and curve_mat.shape == (1000, 99):
        st.subheader(f"Early Electrochemical Curves ({chosen_cell_id})")
        v_grid = np.linspace(3.6, 2.0, 1000)
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        
        # Plot Qdlin
        cycles_to_plot = [8, 18, 48, 98] # Cycles 10, 20, 50, 100
        labels = ['Cycle 10', 'Cycle 20', 'Cycle 50', 'Cycle 100']
        for c_idx, lbl in zip(cycles_to_plot, labels):
            axes[0].plot(v_grid, curve_mat[:, c_idx], label=lbl, linewidth=1.5)
        axes[0].set_title(f"Discharge Capacity $Q_{{dlin}}(V)$ vs Voltage")
        axes[0].set_xlabel("Voltage (V)")
        axes[0].set_ylabel("Capacity (Ah)")
        axes[0].set_xlim(3.65, 1.95)
        axes[0].legend()
        
        # Plot Delta Q100-10
        dq = curve_mat[:, 98] - curve_mat[:, 8]
        axes[1].plot(v_grid, dq, color='crimson', linewidth=2.0, label="$\Delta Q_{100-10}(V)$")
        axes[1].axhline(0, color='grey', linestyle='--', linewidth=0.8)
        axes[1].set_title(f"Degradation Shift Curve: $\Delta Q_{{100-10}}(V)$")
        axes[1].set_xlabel("Voltage (V)")
        axes[1].set_ylabel("Capacity Difference (Ah)")
        axes[1].set_xlim(3.65, 1.95)
        axes[1].legend()
        
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Raw curve file not found for this cell; displaying tabular features only.")


# ------------------------------------------------------------------------------
# PAGE 5: STRATEGY COMPARISON
# ------------------------------------------------------------------------------
def page_comparison():
    st.title("📊 Acquisition Strategy Comparison")
    st.markdown("Quantitative comparison across **Random, Maximum Uncertainty, UCB, and Expected Improvement**.")
    
    al_df = load_al_results()
    if al_df is None:
        st.error("Results file `reports/active_learning_results.csv` not found.")
        return
        
    st.subheader("1. Comprehensive Multi-Trial Summary (N₀ = 15 Seed)")
    
    sub = al_df[(al_df['initial_N'] == 15) & (al_df['n_labeled'].isin([15, 20, 25, 30, 35]))]
    agg = sub.groupby(['n_labeled', 'strategy'])[['test1_mae', 'test1_rmse', 'test1_r2', 'test2_mae', 'best_cycle_life_discovered']].agg(['mean', 'std']).reset_index()
    agg.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in agg.columns]
    
    # Format columns for display
    display_rows = []
    for _, r in agg.iterrows():
        display_rows.append({
            'Labeled Cells': int(r['n_labeled']),
            'Strategy': str(r['strategy']),
            'Test 1 MAE': f"{r['test1_mae_mean']:.1f} ± {r['test1_mae_std']:.1f} cyc",
            'Test 1 RMSE': f"{r['test1_rmse_mean']:.1f} ± {r['test1_rmse_std']:.1f} cyc",
            'Test 1 R²': f"{r['test1_r2_mean']:.3f}",
            'Test 2 MAE': f"{r['test2_mae_mean']:.1f} cyc",
            'Best Found': f"{r['best_cycle_life_discovered_mean']:.0f} cyc"
        })
    df_disp = pd.DataFrame(display_rows)
    st.dataframe(df_disp, use_container_width=True)
    
    st.divider()
    st.subheader("2. Key Scientific Findings")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            #### 🏆 Discovery & Efficiency (Active Learning Advantage)
            - **Fastest Record Discovery:** **UCB** and **Expected Improvement** identify the maximum longevity cell (2,160 cycles) within an average of **15.4 tests** (1 step after initial seed).
            - **Random Inefficiency:** Random sampling requires **19.0 to 31 tests** to locate the same optimal battery.
            - **Peak Accuracy:** **Maximum Uncertainty** achieves the lowest final Test 1 error at budget 35 (**MAE = 73.6 cyc, R² = 0.9315**).
            """
        )
    with col_b:
        st.markdown(
            """
            #### ⚖️ Honest Assessment of Random Baseline
            - **Intermediate Parity:** Around $N = 20$, Random sampling is surprisingly competitive in mean MAE (~88 cycles).
            - **High Random Volatility:** However, Random sampling exhibits **over 3x higher variance** across trials (RMSE std = 13.2 cyc vs ~3.5 cyc for active learning).
            - **Conclusion:** Active learning eliminates the high risk of uninformative batch draws in physical lab testing.
            """
        )


# ------------------------------------------------------------------------------
# PAGE 6: METHODOLOGY
# ------------------------------------------------------------------------------
def page_methodology():
    st.title("📐 Methodology & Research Foundations")
    st.markdown(
        """
        ### End-to-End Scientific Architecture
        BatteryMind is built on the peer-reviewed methodology of **Severson et al. (Nature Energy 2019)** 
        and **Attia et al. (Nature 2020)**.
        """
    )
    
    st.markdown(
        """
        ```
        Raw Battery Cycler Data (124 Cells, A123 18650 LFP)
                       ↓
        Early-Cycle Extraction (Cycles 2 to 100, V: 3.6V → 2.0V)
                       ↓
        Approved Features: ΔQ100-10(V) Variance, Min, Mean, Q100, Fade
                       ↓
        Gaussian Process Surrogate (Matérn ν=2.5 Anisotropic Kernel)
                       ↓
        Uncertainty Quantification (Epistemic Posterior σ(x))
                       ↓
        Acquisition Function (Random / Max Uncertainty / UCB / Expected Improvement)
                       ↓
        Candidate Experiment Selection & Retrospective Label Reveal
        ```
        """
    )
    
    st.divider()
    st.subheader("1. Data Leakage Prevention (5-Point Audit)")
    st.markdown(
        """
        1. **Strict Temporal Horizon:** Only measurements from Cycle $\le 100$ are accessed.
        2. **Target Isolation:** Cycle-life ($y$) is strictly unobservable to the model before acquisition.
        3. **Split Separation:** The 41 Train cells form the simulation pool; Test 1 (43 cells) and Test 2 (40 cells) are never evaluated during selection.
        4. **Pipeline Encapsulation:** `StandardScaler` is fit **only on the active labeled set**.
        5. **No Speculative Mapping:** Charging policies in `policies_all.csv` are decoupled to prevent unverified row matching.
        """
    )
    
    st.divider()
    st.subheader("2. Methodological Limitations")
    st.markdown(
        """
        - **Cell Chemistry Specificity:** Benchmarked on commercial A123 $\text{LiFePO}_4$/graphite cylindrical cells.
        - **Retrospective Setup:** Simulations draw from historical experimental records rather than commanding a live physical cycler.
        - **Fixed Discharge Protocol:** Features focus on charging-induced degradation under standardized 4C discharge.
        """
    )


# ==============================================================================
# MAIN APPLICATION ENTRYPOINT
# ==============================================================================

def main():
    selected_page = render_sidebar()
    
    if selected_page == "Overview":
        page_overview()
    elif selected_page == "Cycle-Life Predictor":
        page_predictor()
    elif selected_page == "Active Learning Simulator":
        page_al_simulator()
    elif selected_page == "Battery Explorer":
        page_explorer()
    elif selected_page == "Strategy Comparison":
        page_comparison()
    elif selected_page == "Methodology":
        page_methodology()


if __name__ == '__main__':
    main()
