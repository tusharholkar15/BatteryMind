"""
Script: active_learning.py
Description: Active Learning & Experiment Selection Engine for BatteryMind (Phase 4).
Simulates closed-loop sequential experimentation using the 41 Train cells as an
experimental pool and evaluates generalization on untouched Test 1 and Test 2 cohorts.

Compares:
  1. Random (Baseline)
  2. Maximum Uncertainty (Exploration)
  3. Upper Confidence Bound (UCB - Optimistic Exploitation/Exploration)
  4. Expected Improvement (EI - Longevity Discovery Optimization)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error


# Visual styling for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


class ActiveLearningEnvironment:
    """
    Simulates a laboratory testing environment using the 41 Train cells.
    Strictly encapsulates candidate labels to prevent data leakage prior to selection.
    """
    def __init__(self, df_train, feature_cols):
        self.df_train = df_train.copy().reset_index(drop=True)
        self.feature_cols = feature_cols
        self.all_cell_ids = self.df_train['cell_id'].values
        self.X_all = self.df_train[feature_cols].values
        self.y_all = self.df_train['cycle_life'].values
        self.z_all = np.log10(self.y_all)
        
        self.labeled_indices = []
        self.candidate_indices = []
        
    def reset(self, initial_seed_indices):
        """Initializes the environment with a starting labeled seed set."""
        self.labeled_indices = list(initial_seed_indices)
        self.candidate_indices = [i for i in range(len(self.df_train)) if i not in self.labeled_indices]
        
    def get_candidate_features(self):
        """Returns feature matrix X for current candidates. (Zero target leakage)."""
        if not self.candidate_indices:
            return None, []
        X_cand = self.X_all[self.candidate_indices]
        return X_cand, list(self.candidate_indices)
        
    def get_labeled_data(self):
        """Returns X, z (log10 target), and y (physical cycles) for labeled cells."""
        X_lab = self.X_all[self.labeled_indices]
        z_lab = self.z_all[self.labeled_indices]
        y_lab = self.y_all[self.labeled_indices]
        return X_lab, z_lab, y_lab
        
    def reveal_label(self, selected_pool_idx):
        """
        Transfers selected candidate from candidate pool to labeled set,
        and reveals its ground truth cycle life.
        """
        assert selected_pool_idx in self.candidate_indices, "Selected index not in candidate pool!"
        self.candidate_indices.remove(selected_pool_idx)
        self.labeled_indices.append(selected_pool_idx)
        
        cell_id = self.all_cell_ids[selected_pool_idx]
        true_y = self.y_all[selected_pool_idx]
        return cell_id, true_y


class GaussianProcessSurrogate:
    """
    Gaussian Process surrogate model with Matérn nu=2.5 kernel.
    Scales features strictly on current labeled observations.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.model = None
        
    def fit(self, X_train, z_train):
        """Fit scaler and GP on current labeled set."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        kernel = (
            C(1.0, (1e-3, 1e3)) * 
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + 
            WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,
            random_state=self.random_state
        )
        self.model.fit(X_scaled, z_train)
        
    def predict(self, X):
        """Predicts mean and standard deviation in log10(cycle_life) space."""
        X_scaled = self.scaler.transform(X)
        z_mean, z_std = self.model.predict(X_scaled, return_std=True)
        return z_mean, z_std


def compute_acquisition_scores(strategy, z_mean, z_std, best_z_labeled, beta=1.96, xi=0.01, rng=None):
    """
    Computes acquisition utility scores for candidate cells:
      - RANDOM: Uniform random selection
      - MAX_UNCERTAINTY: a(x) = sigma(x)
      - UCB: a(x) = mu(x) + beta * sigma(x)
      - EXPECTED_IMPROVEMENT: analytical EI for maximization
    """
    if strategy == 'RANDOM':
        return rng.uniform(0.0, 1.0, size=len(z_mean))
        
    elif strategy == 'MAX_UNCERTAINTY':
        return z_std.copy()
        
    elif strategy == 'UCB':
        return z_mean + beta * z_std
        
    elif strategy == 'EXPECTED_IMPROVEMENT':
        # EI(x) = (mu - best_z - xi)*Phi(u) + sigma*phi(u)
        diff = z_mean - best_z_labeled - xi
        scores = np.zeros_like(z_mean)
        valid = z_std > 1e-9
        
        if np.any(valid):
            u = np.zeros_like(z_mean)
            u[valid] = diff[valid] / z_std[valid]
            scores[valid] = diff[valid] * norm.cdf(u[valid]) + z_std[valid] * norm.pdf(u[valid])
            
        scores[~valid] = 0.0
        return scores
        
    else:
        raise ValueError(f"Unknown acquisition strategy: {strategy}")


def evaluate_test_cohort(surrogate, X_eval, y_eval):
    """Evaluates surrogate predictions and uncertainty on held-out test cohort."""
    z_pred, z_std = surrogate.predict(X_eval)
    y_pred = 10.0 ** z_pred
    
    mae = mean_absolute_error(y_eval, y_pred)
    rmse = root_mean_squared_error(y_eval, y_pred)
    r2 = r2_score(y_eval, y_pred)
    medae = median_absolute_error(y_eval, y_pred)
    
    # 95% predictive interval in physical space
    y_lower = 10.0 ** (z_pred - 1.96 * z_std)
    y_upper = 10.0 ** (z_pred + 1.96 * z_std)
    cov_95 = float(np.mean((y_eval >= y_lower) & (y_eval <= y_upper)) * 100.0)
    mean_std = float(np.mean(z_std))
    
    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2': round(r2, 4),
        'medae': round(medae, 2),
        'mean_std': round(mean_std, 4),
        'cov_95_pct': round(cov_95, 2)
    }


def run_active_learning_simulation(processed_csv='data/processed/batterymind_features.csv',
                                   reports_dir='reports',
                                   seed_sizes=[10, 15, 20],
                                   max_budget=35,
                                   n_trials=5,
                                   beta=1.96,
                                   xi=0.01):
    """
    Executes the comprehensive Active Learning benchmark across:
      - Seed sizes: 10, 15, 20
      - Strategies: RANDOM, MAX_UNCERTAINTY, UCB, EXPECTED_IMPROVEMENT
      - Multiple independent random trials with controlled seed matching
    """
    print("=" * 80)
    print("BATTERYMIND: ACTIVE LEARNING & EXPERIMENT SELECTION BENCHMARK (PHASE 4)")
    print("=" * 80)
    
    df = pd.read_csv(processed_csv)
    feature_cols = ['log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cycle100', 'fade_100_10']
    
    # Isolate Train (41 cells) and Test sets
    df_train = df[df['dataset_split'] == 'train'].reset_index(drop=True)
    df_test1 = df[df['dataset_split'] == 'test1'].reset_index(drop=True)
    df_test2 = df[df['dataset_split'] == 'test2'].reset_index(drop=True)
    
    X_test1 = df_test1[feature_cols].values
    y_test1 = df_test1['cycle_life'].values
    
    X_test2 = df_test2[feature_cols].values
    y_test2 = df_test2['cycle_life'].values
    
    strategies = ['RANDOM', 'MAX_UNCERTAINTY', 'UCB', 'EXPECTED_IMPROVEMENT']
    trial_random_seeds = [42, 101, 202, 303, 404][:n_trials]
    
    records = []
    
    for initial_N in seed_sizes:
        print(f"\n=======================================================")
        print(f"BENCHMARK: Initial Seed Size N0 = {initial_N} cells (Target Budget: {max_budget})")
        print(f"=======================================================")
        
        for trial_idx, seed_val in enumerate(trial_random_seeds):
            rng_trial = np.random.RandomState(seed_val)
            
            # Select identical initial seed indices for ALL strategies in this trial
            initial_seed = rng_trial.choice(len(df_train), size=initial_N, replace=False)
            
            for strategy in strategies:
                env = ActiveLearningEnvironment(df_train, feature_cols)
                env.reset(initial_seed)
                
                # Independent RNG for strategy decisions
                rng_strat = np.random.RandomState(seed_val + 1000)
                
                # Step 0: Initial model evaluation before any acquisitions
                surrogate = GaussianProcessSurrogate(random_state=seed_val)
                X_lab, z_lab, y_lab = env.get_labeled_data()
                surrogate.fit(X_lab, z_lab)
                
                m1 = evaluate_test_cohort(surrogate, X_test1, y_test1)
                m2 = evaluate_test_cohort(surrogate, X_test2, y_test2)
                best_y = int(np.max(y_lab))
                
                records.append({
                    'initial_N': initial_N,
                    'trial_idx': trial_idx,
                    'trial_seed': seed_val,
                    'strategy': strategy,
                    'step': 0,
                    'n_labeled': initial_N,
                    'selected_cell_id': 'INITIAL_SEED',
                    'pred_mean_log10': np.nan,
                    'pred_std_log10': np.nan,
                    'true_cycle_life': np.nan,
                    'best_cycle_life_discovered': best_y,
                    'test1_mae': m1['mae'],
                    'test1_rmse': m1['rmse'],
                    'test1_r2': m1['r2'],
                    'test1_medae': m1['medae'],
                    'test1_cov95': m1['cov_95_pct'],
                    'test2_mae': m2['mae'],
                    'test2_rmse': m2['rmse'],
                    'test2_r2': m2['r2']
                })
                
                # Active learning sequential acquisition loop
                step = 1
                while len(env.labeled_indices) < max_budget and len(env.candidate_indices) > 0:
                    # 1. Get candidate features (ZERO target access)
                    X_cand, cand_indices = env.get_candidate_features()
                    if X_cand is None or len(cand_indices) == 0:
                        break
                        
                    # 2. Predict distribution on candidates
                    z_mean, z_std = surrogate.predict(X_cand)
                    best_z = float(np.max(z_lab))
                    
                    # 3. Compute acquisition scores and select candidate
                    scores = compute_acquisition_scores(
                        strategy, z_mean, z_std, best_z, beta=beta, xi=xi, rng=rng_strat
                    )
                    selected_cand_idx = int(np.argmax(scores))
                    selected_pool_idx = cand_indices[selected_cand_idx]
                    
                    pred_m = float(z_mean[selected_cand_idx])
                    pred_s = float(z_std[selected_cand_idx])
                    
                    # 4. Reveal ground truth label (Virtual Experimentation)
                    cell_id, true_y = env.reveal_label(selected_pool_idx)
                    
                    # 5. Retrain surrogate on expanded labeled pool
                    X_lab, z_lab, y_lab = env.get_labeled_data()
                    surrogate.fit(X_lab, z_lab)
                    
                    # 6. Evaluate on held-out test cohorts
                    m1 = evaluate_test_cohort(surrogate, X_test1, y_test1)
                    m2 = evaluate_test_cohort(surrogate, X_test2, y_test2)
                    best_y = int(np.max(y_lab))
                    
                    records.append({
                        'initial_N': initial_N,
                        'trial_idx': trial_idx,
                        'trial_seed': seed_val,
                        'strategy': strategy,
                        'step': step,
                        'n_labeled': len(env.labeled_indices),
                        'selected_cell_id': cell_id,
                        'pred_mean_log10': round(pred_m, 4),
                        'pred_std_log10': round(pred_s, 4),
                        'true_cycle_life': int(true_y),
                        'best_cycle_life_discovered': best_y,
                        'test1_mae': m1['mae'],
                        'test1_rmse': m1['rmse'],
                        'test1_r2': m1['r2'],
                        'test1_medae': m1['medae'],
                        'test1_cov95': m1['cov_95_pct'],
                        'test2_mae': m2['mae'],
                        'test2_rmse': m2['rmse'],
                        'test2_r2': m2['r2']
                    })
                    step += 1
                    
        print(f"  Finished {n_trials} trials for N0 = {initial_N} across all 4 strategies.")
        
    df_results = pd.DataFrame(records)
    
    # Save results to CSV
    os.makedirs(reports_dir, exist_ok=True)
    out_csv = os.path.join(reports_dir, 'active_learning_results.csv')
    df_results.to_csv(out_csv, index=False)
    print(f"\n-> Successfully saved trajectory results: {out_csv} ({len(df_results)} rows)")
    
    # Generate Publication-Quality Learning Curves
    generate_active_learning_plots(df_results, reports_dir)
    
    # Print Executive Summary Table
    print_benchmark_summary(df_results)
    
    return df_results


def generate_active_learning_plots(df_results, reports_dir):
    """Generates learning curves and discovery curves."""
    palette = {
        'RANDOM': '#7f7f7f',
        'MAX_UNCERTAINTY': '#1f77b4',
        'UCB': '#2ca02c',
        'EXPECTED_IMPROVEMENT': '#d62728'
    }
    
    # Focus on primary initial seed size N0 = 15 for primary visual comparison
    df_plot = df_results[df_results['initial_N'] == 15].copy()
    
    # -------------------------------------------------------------
    # Plot 1: Learning Curves (Test 1 MAE & RMSE vs Number of Cells)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Test 1 MAE
    sns.lineplot(
        data=df_plot, x='n_labeled', y='test1_mae', hue='strategy',
        palette=palette, marker='o', linewidth=2.0, ax=axes[0], errorbar=('ci', 68)
    )
    axes[0].set_title('Test 1 (In-Distribution) MAE Learning Curve (N0 = 15)')
    axes[0].set_xlabel('Total Number of Labeled Cells')
    axes[0].set_ylabel('Held-Out Test 1 MAE (cycles)')
    axes[0].axhline(80.0, color='grey', linestyle=':', label='80-cycle Error Benchmark')
    axes[0].legend(title='Acquisition Strategy')
    
    # Right: Test 1 RMSE
    sns.lineplot(
        data=df_plot, x='n_labeled', y='test1_rmse', hue='strategy',
        palette=palette, marker='s', linewidth=2.0, ax=axes[1], errorbar=('ci', 68)
    )
    axes[1].set_title('Test 1 (In-Distribution) RMSE Learning Curve (N0 = 15)')
    axes[1].set_xlabel('Total Number of Labeled Cells')
    axes[1].set_ylabel('Held-Out Test 1 RMSE (cycles)')
    axes[1].legend(title='Acquisition Strategy')
    
    plt.tight_layout()
    plot1_path = os.path.join(reports_dir, 'active_learning_learning_curves.png')
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    print(f"Saved: {plot1_path}")
    
    # -------------------------------------------------------------
    # Plot 2: Longevity Discovery Curve (Best Cycle Life Discovered)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.lineplot(
        data=df_plot, x='n_labeled', y='best_cycle_life_discovered', hue='strategy',
        palette=palette, marker='o', linewidth=2.2, ax=ax, errorbar=('ci', 68)
    )
    ax.set_title('Experiment Efficiency: Maximum Cycle Life Discovered (N0 = 15)')
    ax.set_xlabel('Total Number of Labeled Cells')
    ax.set_ylabel('Best Cycle Life Found in Labeled Set (cycles)')
    ax.axhline(2160, color='blue', linestyle='--', alpha=0.5, label='Global Max Train Cycle Life (2,160 cyc)')
    ax.legend(title='Acquisition Strategy', loc='lower right')
    
    plt.tight_layout()
    plot2_path = os.path.join(reports_dir, 'active_learning_discovery_curve.png')
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    print(f"Saved: {plot2_path}")


def print_benchmark_summary(df_results):
    """Prints a quantitative comparison table of acquisition efficiency."""
    print("\n" + "=" * 80)
    print("ACTIVE LEARNING QUANTITATIVE SUMMARY (Mean across trials)")
    print("=" * 80)
    
    # Compare performance at key milestones: initial (N0=15), mid (N=20), final (N=30)
    sub = df_results[(df_results['initial_N'] == 15) & (df_results['n_labeled'].isin([15, 20, 25, 30]))]
    agg = sub.groupby(['n_labeled', 'strategy'])[['test1_mae', 'test1_rmse', 'test1_r2', 'best_cycle_life_discovered']].mean().reset_index()
    
    print(f"{'Labeled Cells':<14} | {'Strategy':<20} | {'Test 1 MAE':<12} | {'Test 1 RMSE':<12} | {'Test 1 R2':<10} | {'Best Found (cyc)':<16}")
    print("-" * 95)
    for _, r in agg.iterrows():
        print(f"{int(r['n_labeled']):<14} | {r['strategy']:<20} | {r['test1_mae']:>10.2f} cyc | {r['test1_rmse']:>10.2f} cyc | {r['test1_r2']:>8.4f} | {r['best_cycle_life_discovered']:>14.1f}")
    print()


if __name__ == '__main__':
    run_active_learning_simulation()
