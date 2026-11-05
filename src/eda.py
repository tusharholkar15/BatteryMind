"""
Script: eda.py
Description: Exploratory Data Analysis (EDA) for BatteryMind (Phase 2).
Performs statistical analysis of cycle life, charging policy parameter space,
Qdlin early-cycle curve behaviors, and early degradation signatures.
Generates publication-quality diagnostic plots in reports/figures/.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def load_dataset(raw_dir='data/raw'):
    """Load cell metadata, targets, curve matrices, and candidate policies."""
    cycle_dir = os.path.join(raw_dir, 'cycle_lives')
    splits = {
        'train': os.path.join(cycle_dir, 'train_cycle_lives.csv'),
        'test1': os.path.join(cycle_dir, 'test1_cycle_lives.csv'),
        'test2': os.path.join(cycle_dir, 'test2_cycle_lives.csv')
    }
    
    cells_meta = []
    curves_dict = {}
    
    for split_name, path in splits.items():
        targets = pd.read_csv(path, header=None)[0].values
        split_dir = os.path.join(raw_dir, split_name)
        for idx, y in enumerate(targets, start=1):
            cell_id = f"{split_name}_cell{idx}"
            curve_file = os.path.join(split_dir, f"cell{idx}.csv")
            if os.path.exists(curve_file):
                curve_mat = np.genfromtxt(curve_file, delimiter=',')
                curves_dict[cell_id] = curve_mat
            else:
                curve_mat = None
                
            cells_meta.append({
                'cell_id': cell_id,
                'split': split_name,
                'cell_num': idx,
                'cycle_life': int(y),
                'log_cycle_life': np.log10(y)
            })
            
    df_cells = pd.DataFrame(cells_meta)
    
    # Load candidate experimental policies
    policy_file = os.path.join(raw_dir, 'policies_all.csv')
    df_policies = pd.read_csv(policy_file, header=None)
    df_policies.columns = ['C1_current_A', 'C2_current_A', 'C3_current_A', 'C4_current_A']
    
    return df_cells, curves_dict, df_policies


def plot_cycle_life_distribution(df_cells, out_dir):
    """Plot 1: Cycle-life distribution across cohorts."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Linear scale
    sns.histplot(
        data=df_cells, x='cycle_life', hue='split', multiple='stack',
        bins=25, palette='tab10', ax=axes[0], edgecolor='white'
    )
    axes[0].set_title('Cycle Life Distribution (Linear Scale)')
    axes[0].set_xlabel('Cycle Life to 80% Capacity Retention')
    axes[0].set_ylabel('Cell Count')
    axes[0].axvline(df_cells['cycle_life'].median(), color='red', linestyle='--', linewidth=1.5,
                    label=f'Global Median ({df_cells["cycle_life"].median():.0f})')
    axes[0].legend()
    
    # Log10 scale
    sns.histplot(
        data=df_cells, x='log_cycle_life', hue='split', multiple='stack',
        bins=25, palette='tab10', ax=axes[1], edgecolor='white'
    )
    axes[1].set_title('Cycle Life Distribution (Log10 Scale)')
    axes[1].set_xlabel('log10(Cycle Life)')
    axes[1].set_ylabel('Cell Count')
    axes[1].axvline(df_cells['log_cycle_life'].median(), color='red', linestyle='--', linewidth=1.5,
                    label=f'Median ({df_cells["log_cycle_life"].median():.2f})')
    axes[1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, '01_cycle_life_distribution.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_cycle_life_boxplots(df_cells, out_dir):
    """Plot 2: Cycle life boxplots & strip plots per cohort."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # Boxplot
    sns.boxplot(data=df_cells, x='split', y='cycle_life', hue='split', legend=False, palette='Set2', ax=ax, width=0.45, boxprops=dict(alpha=0.8))
    sns.stripplot(data=df_cells, x='split', y='cycle_life', color='black', alpha=0.6, jitter=0.2, size=6, ax=ax)
    
    ax.set_title('Battery Longevity Cohort Comparison')
    ax.set_xlabel('Cohort Split')
    ax.set_ylabel('Cycle Life (cycles to 80% EOL)')
    
    # Annotate summary stats on the plot
    for i, split in enumerate(['train', 'test1', 'test2']):
        sub = df_cells[df_cells['split'] == split]['cycle_life']
        ax.text(i, sub.max() + 50, f"n={len(sub)}\nMed={sub.median():.0f}", 
                ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
    plt.tight_layout()
    save_path = os.path.join(out_dir, '02_cycle_life_boxplots.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_candidate_policies(df_policies, out_dir):
    """Plot 3: Distribution of candidate charging currents (C1, C2, C3, C4)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    stages = [('C1_current_A', 'Stage 1 Current (0-20% SOC)'),
              ('C2_current_A', 'Stage 2 Current (20-40% SOC)'),
              ('C3_current_A', 'Stage 3 Current (40-60% SOC)'),
              ('C4_current_A', 'Stage 4 Current (60-80% SOC)')]
    
    for idx, (col, title) in enumerate(stages):
        ax = axes[idx]
        sns.histplot(df_policies[col], bins=15, kde=True, color=colors[idx], ax=ax, edgecolor='white')
        ax.set_title(title)
        ax.set_xlabel('Current Rate (C-rate)')
        ax.set_ylabel('Policy Count')
        ax.axvline(df_policies[col].mean(), color='black', linestyle=':', label=f'Mean: {df_policies[col].mean():.2f}C')
        ax.legend()
        
    plt.suptitle('Candidate Experimental Policy Space (224 Protocols)', fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(out_dir, '03_charging_policies_distributions.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_qdlin_curves(curves_dict, df_cells, out_dir):
    """Plot 4: Qdlin curves across voltage for representative cycles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    v_grid = np.linspace(3.6, 2.0, 1000)
    
    # Pick a representative cell (e.g. train_cell1)
    rep_cell_id = 'train_cell1'
    mat = curves_dict[rep_cell_id]
    # In curves matrix, col 0 = cycle 2, col 8 = cycle 10, col 98 = cycle 100
    cycle_indices = [8, 18, 48, 78, 98] # cycle numbers: index+2 => cycles 10, 20, 50, 80, 100
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cycle_indices)))
    
    for c_idx, color in zip(cycle_indices, colors):
        cycle_num = c_idx + 2 # col 0 is cycle 2
        axes[0].plot(v_grid, mat[:, c_idx], label=f'Cycle {cycle_num}', color=color, linewidth=1.5)
        
    axes[0].set_title(f'Discharge Capacity $Q_{{dlin}}(V)$ vs Voltage ({rep_cell_id})')
    axes[0].set_xlabel('Discharge Voltage (V)')
    axes[0].set_ylabel('Linearly Interpolated Capacity $Q_{dlin}$ (Ah)')
    axes[0].set_xlim(3.65, 1.95) # Inverted axis from 3.6 down to 2.0
    axes[0].legend(loc='upper right')
    
    # Delta Q curves relative to cycle 10 (index 8)
    cycle_10_idx = 8 # cycle 10 is at index 8 (cycle 2 is index 0)
    q10 = mat[:, cycle_10_idx]
    for c_idx, color in zip(cycle_indices[1:], colors[1:]):
        cycle_num = c_idx + 2
        delta_q = mat[:, c_idx] - q10
        axes[1].plot(v_grid, delta_q, label=f'Cycle {cycle_num} - Cycle 10', color=color, linewidth=1.5)
        
    axes[1].set_title(f'Degradation Shift $\Delta Q_{{n-10}}(V)$ ({rep_cell_id})')
    axes[1].set_xlabel('Discharge Voltage (V)')
    axes[1].set_ylabel('Capacity Difference $\Delta Q$ (Ah)')
    axes[1].set_xlim(3.65, 1.95)
    axes[1].axhline(0, color='grey', linestyle='--', linewidth=0.8)
    axes[1].legend(loc='lower left')
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, '04_qdlin_curves_sample_cells.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_high_vs_low_life(curves_dict, df_cells, out_dir):
    """Plot 5: Comparison of Delta Q100-10 curves for long-lived vs short-lived cells."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    v_grid = np.linspace(3.6, 2.0, 1000)
    
    # Define thresholds
    long_lived = df_cells[df_cells['cycle_life'] >= 1200].sort_values('cycle_life', ascending=False)
    short_lived = df_cells[df_cells['cycle_life'] <= 500].sort_values('cycle_life')
    
    c10_idx = 8  # Cycle 10
    c100_idx = 98 # Cycle 100
    
    # Plot top 5 long lived
    for _, row in long_lived.head(5).iterrows():
        cid = row['cell_id']
        mat = curves_dict[cid]
        dq = mat[:, c100_idx] - mat[:, c10_idx]
        ax.plot(v_grid, dq, color='#1f77b4', alpha=0.85, linewidth=1.5,
                label=f"Long-lived ({row['cycle_life']} cyc)" if _ == long_lived.index[0] else "")
        
    # Plot bottom 5 short lived
    for _, row in short_lived.head(5).iterrows():
        cid = row['cell_id']
        mat = curves_dict[cid]
        dq = mat[:, c100_idx] - mat[:, c10_idx]
        ax.plot(v_grid, dq, color='#d62728', alpha=0.85, linewidth=1.5,
                label=f"Short-lived ({row['cycle_life']} cyc)" if _ == short_lived.index[0] else "")
        
    ax.set_title(r'Scientific Signature: $\Delta Q_{100-10}(V)$ for Long vs. Short-Lived Batteries')
    ax.set_xlabel('Discharge Voltage (V)')
    ax.set_ylabel(r'$\Delta Q_{100-10}$ (Ah) = $Q_{100}(V) - Q_{10}(V)$')
    ax.set_xlim(3.65, 1.95)
    ax.axhline(0, color='black', linestyle=':', linewidth=0.8)
    ax.legend(frameon=True, facecolor='white', loc='lower left')
    
    # Annotate key voltage region
    ax.axvspan(2.7, 2.1, color='yellow', alpha=0.15, label='High Discrimination Region (Plateau shift)')
    ax.text(2.4, -0.06, 'Distinct Degradation Spread\n(Loss of active lithium)', ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, '05_delta_q_high_vs_low_life.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_early_capacity_trajectories(curves_dict, df_cells, out_dir):
    """Plot 6: Capacity fade across cycles 2 to 100."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Nominal discharge capacity at end of discharge (voltage = 2.0V, last row of curve matrix)
    cycles = np.arange(2, 101) # 99 cycles
    
    long_lived = df_cells[df_cells['cycle_life'] >= 1200]
    short_lived = df_cells[df_cells['cycle_life'] <= 500]
    
    for _, row in long_lived.iterrows():
        mat = curves_dict[row['cell_id']]
        end_cap = mat[-1, :] # row 999 is 2.0V
        ax.plot(cycles, end_cap, color='#1f77b4', alpha=0.3, linewidth=1.0)
        
    for _, row in short_lived.iterrows():
        mat = curves_dict[row['cell_id']]
        end_cap = mat[-1, :]
        ax.plot(cycles, end_cap, color='#d62728', alpha=0.3, linewidth=1.0)
        
    # Plot means
    long_caps = np.array([curves_dict[row['cell_id']][-1, :] for _, row in long_lived.iterrows()])
    short_caps = np.array([curves_dict[row['cell_id']][-1, :] for _, row in short_lived.iterrows()])
    
    ax.plot(cycles, np.mean(long_caps, axis=0), color='#1f77b4', linewidth=2.5, label=f'Mean Long-Lived (n={len(long_lived)})')
    ax.plot(cycles, np.mean(short_caps, axis=0), color='#d62728', linewidth=2.5, label=f'Mean Short-Lived (n={len(short_lived)})')
    
    ax.set_title('Early Capacity Fade Trajectories (Cycles 2 to 100)')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Discharge Capacity at 2.0V Cutoff (Ah)')
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, '06_early_cycle_degradation_trajectory.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def compute_and_plot_correlations(curves_dict, df_cells, out_dir):
    """Plot 7: Early candidate summary metrics vs cycle life."""
    feature_rows = []
    c10_idx = 8
    c100_idx = 98
    
    for _, row in df_cells.iterrows():
        cid = row['cell_id']
        mat = curves_dict[cid]
        
        # 1. Delta Q 100-10 vector
        dq = mat[:, c100_idx] - mat[:, c10_idx]
        var_dq = np.var(dq)
        log_var_dq = np.log10(np.abs(var_dq) + 1e-12)
        min_dq = np.min(dq)
        mean_dq = np.mean(dq)
        
        # 2. Capacity at cycle 2, 10, 100
        q_cyc2 = mat[-1, 0]
        q_cyc10 = mat[-1, c10_idx]
        q_cyc100 = mat[-1, c100_idx]
        fade_100_10 = q_cyc100 - q_cyc10
        
        # 3. Degradation slope cycles 10 to 100
        slope_10_100 = (q_cyc100 - q_cyc10) / 90.0
        
        feature_rows.append({
            'cell_id': cid,
            'split': row['split'],
            'cycle_life': row['cycle_life'],
            'log_cycle_life': row['log_cycle_life'],
            'log10_var_deltaQ': log_var_dq,
            'min_deltaQ': min_dq,
            'mean_deltaQ': mean_dq,
            'q_cyc100': q_cyc100,
            'fade_100_10': fade_100_10,
            'slope_10_100': slope_10_100
        })
        
    df_feat = pd.DataFrame(feature_rows)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Scatter 1: log10(var(delta Q)) vs log10(cycle_life)
    r1 = np.corrcoef(df_feat['log10_var_deltaQ'], df_feat['log_cycle_life'])[0, 1]
    sns.scatterplot(
        data=df_feat, x='log10_var_deltaQ', y='log_cycle_life', hue='split',
        palette='tab10', s=55, alpha=0.85, ax=axes[0]
    )
    sns.regplot(
        data=df_feat, x='log10_var_deltaQ', y='log_cycle_life',
        scatter=False, ax=axes[0], color='black', line_kws={'linestyle': '--', 'linewidth': 1.2}
    )
    axes[0].set_title(f'$\log_{{10}}(\mathrm{{Var}}(\Delta Q_{{100-10}}))$ vs $\log_{{10}}(\mathrm{{Cycle\ Life}})$ (Pearson r = {r1:.2f})')
    axes[0].set_xlabel(r'$\log_{10}(\mathrm{Variance\ of\ } \Delta Q_{100-10})$')
    axes[0].set_ylabel(r'$\log_{10}(\mathrm{Cycle\ Life})$')
    
    # Scatter 2: Early slope vs log10(cycle_life)
    r2 = np.corrcoef(df_feat['fade_100_10'], df_feat['log_cycle_life'])[0, 1]
    sns.scatterplot(
        data=df_feat, x='fade_100_10', y='log_cycle_life', hue='split',
        palette='tab10', s=55, alpha=0.85, ax=axes[1]
    )
    sns.regplot(
        data=df_feat, x='fade_100_10', y='log_cycle_life',
        scatter=False, ax=axes[1], color='black', line_kws={'linestyle': '--', 'linewidth': 1.2}
    )
    axes[1].set_title(f'Early Capacity Fade $(Q_{{100}} - Q_{{10}})$ vs $\log_{{10}}(\mathrm{{Cycle\ Life}})$ (Pearson r = {r2:.2f})')
    axes[1].set_xlabel('Capacity Fade (Ah) between Cycle 10 and Cycle 100')
    axes[1].set_ylabel(r'$\log_{10}(\mathrm{Cycle\ Life})$')
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, '07_early_prediction_correlations.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")
    
    return df_feat


def run_eda():
    out_dir = os.path.join('reports', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data for EDA...")
    df_cells, curves_dict, df_policies = load_dataset()
    print(f"Loaded metadata for {len(df_cells)} cells and {len(df_policies)} candidate policies.")
    
    print("\n--- Generating Diagnostic Figures ---")
    plot_cycle_life_distribution(df_cells, out_dir)
    plot_cycle_life_boxplots(df_cells, out_dir)
    plot_candidate_policies(df_policies, out_dir)
    plot_qdlin_curves(curves_dict, df_cells, out_dir)
    plot_high_vs_low_life(curves_dict, df_cells, out_dir)
    plot_early_capacity_trajectories(curves_dict, df_cells, out_dir)
    df_feat = compute_and_plot_correlations(curves_dict, df_cells, out_dir)
    
    # Compute summary statistics
    print("\n--- SUMMARY OF CANDIDATE FEATURES VS TARGET ---")
    corr_matrix = df_feat[['cycle_life', 'log_cycle_life', 'log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cyc100', 'fade_100_10', 'slope_10_100']].corr()
    print(corr_matrix[['cycle_life', 'log_cycle_life']].to_string())
    
    print("\nEDA Execution Complete!")


if __name__ == '__main__':
    run_eda()
