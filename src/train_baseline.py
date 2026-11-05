"""
Script: train_baseline.py
Description: Small-data probabilistic baseline modeling for BatteryMind (Phase 3).
Trains Bayesian Ridge Regression and Gaussian Process Regression on seed batches
of size N = 10, 15, 20 from designated training cells.
Evaluates predictions and calibrated epistemic uncertainty on held-out Test 1 and Test 2.
Saves results to reports/baseline_results.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error


def load_processed_data(data_path='data/processed/batterymind_features.csv'):
    """Loads the validated processed dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run extract_features.py first.")
    df = pd.read_csv(data_path)
    return df


def evaluate_probabilistic_predictions(y_true, z_pred, z_std):
    """
    Computes regression and probabilistic uncertainty calibration metrics.
    z_pred and z_std are in log10(cycle_life) space.
    y_true is in physical cycle life units.
    """
    # Back-transform point predictions to physical cycles
    y_pred = 10.0 ** z_pred
    
    # Standard evaluation metrics in physical cycles
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    
    # 95% predictive interval in physical space: [10^(z - 1.96*std), 10^(z + 1.96*std)]
    y_lower = 10.0 ** (z_pred - 1.96 * z_std)
    y_upper = 10.0 ** (z_pred + 1.96 * z_std)
    coverage_95 = np.mean((y_true >= y_lower) & (y_true <= y_upper)) * 100.0
    
    # Mean predictive uncertainty in log10 space and average ratio in physical space
    mean_std_log10 = float(np.mean(z_std))
    mean_relative_uncert = float(np.mean((y_upper - y_lower) / (2.0 * y_pred)))
    
    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2': round(r2, 4),
        'medae': round(medae, 2),
        'mean_std_log10': round(mean_std_log10, 4),
        'coverage_95_pct': round(coverage_95, 2),
        'mean_rel_uncert': round(mean_relative_uncert, 4)
    }


def train_and_evaluate_baselines(data_path='data/processed/batterymind_features.csv',
                                 reports_dir='reports',
                                 random_seed=42):
    """
    Executes the small-data benchmark experiments for N in [10, 15, 20].
    """
    print("=" * 80)
    print("BATTERYMIND: SMALL-DATA BASELINE MODELING PIPELINE (PHASE 3)")
    print("=" * 80)
    
    df = load_processed_data(data_path)
    
    feature_cols = ['log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cycle100', 'fade_100_10']
    
    # 1. Isolate designated training cells (41 cells)
    df_train_full = df[df['dataset_split'] == 'train'].reset_index(drop=True)
    X_train_pool = df_train_full[feature_cols].values
    y_train_pool = df_train_full['cycle_life'].values
    z_train_pool = np.log10(y_train_pool)
    
    # 2. Isolate untouched evaluation cohorts
    df_test1 = df[df['dataset_split'] == 'test1'].reset_index(drop=True)
    X_test1 = df_test1[feature_cols].values
    y_test1 = df_test1['cycle_life'].values
    
    df_test2 = df[df['dataset_split'] == 'test2'].reset_index(drop=True)
    X_test2 = df_test2[feature_cols].values
    y_test2 = df_test2['cycle_life'].values
    
    df_test_comb = df[df['dataset_split'].isin(['test1', 'test2'])].reset_index(drop=True)
    X_comb = df_test_comb[feature_cols].values
    y_comb = df_test_comb['cycle_life'].values
    
    test_cohorts = {
        'Test 1 (Primary)': (X_test1, y_test1),
        'Test 2 (Out-of-Dist)': (X_test2, y_test2),
        'Combined Test (83 cells)': (X_comb, y_comb)
    }
    
    seed_sizes = [10, 15, 20]
    results_records = []
    
    print(f"Candidate Features ({len(feature_cols)}) : {feature_cols}")
    print(f"Training Pool Size     : {len(df_train_full)} cells")
    print(f"Test 1 (In-Dist) Size  : {len(df_test1)} cells")
    print(f"Test 2 (Shifted) Size  : {len(df_test2)} cells")
    print(f"Evaluated Seed Sizes   : N = {seed_sizes}\n")
    
    for N in seed_sizes:
        print(f">>> Running Experiment: Seed Batch Size N = {N} cells (Random Seed = {random_seed}) <<<")
        
        # Sample N seed cells from the 41 training cells
        rng = np.random.RandomState(random_seed)
        seed_indices = rng.choice(len(df_train_full), size=N, replace=False)
        
        X_seed = X_train_pool[seed_indices]
        z_seed = z_train_pool[seed_indices]
        y_seed = y_train_pool[seed_indices]
        
        # Fit scaler ONLY on the seed training set (Strict Zero-Leakage)
        scaler = StandardScaler()
        X_seed_scaled = scaler.fit_transform(X_seed)
        
        # Model 1: Bayesian Ridge Regression
        br_model = BayesianRidge(max_iter=300, tol=1e-4, compute_score=True)
        br_model.fit(X_seed_scaled, z_seed)
        
        # Model 2: Gaussian Process Regressor with Matérn kernel
        gp_kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        gp_model = GaussianProcessRegressor(
            kernel=gp_kernel,
            alpha=0.0,
            n_restarts_optimizer=5,
            random_state=random_seed
        )
        gp_model.fit(X_seed_scaled, z_seed)
        
        models = {
            'BayesianRidge': br_model,
            'GaussianProcess': gp_model
        }
        
        for model_name, model in models.items():
            for cohort_name, (X_eval, y_eval) in test_cohorts.items():
                X_eval_scaled = scaler.transform(X_eval)
                z_pred, z_std = model.predict(X_eval_scaled, return_std=True)
                
                metrics = evaluate_probabilistic_predictions(y_eval, z_pred, z_std)
                
                results_records.append({
                    'model': model_name,
                    'seed_size_N': N,
                    'test_cohort': cohort_name,
                    'mae_cycles': metrics['mae'],
                    'rmse_cycles': metrics['rmse'],
                    'r2_score': metrics['r2'],
                    'medae_cycles': metrics['medae'],
                    'mean_std_log10': metrics['mean_std_log10'],
                    'coverage_95_pct': metrics['coverage_95_pct'],
                    'mean_rel_uncert': metrics['mean_rel_uncert']
                })
                
                if cohort_name == 'Test 1 (Primary)':
                    print(f"  [{model_name:<16}] N={N:2d} -> Test 1: MAE={metrics['mae']:>5.1f} cyc | "
                          f"RMSE={metrics['rmse']:>5.1f} cyc | R2={metrics['r2']:>6.3f} | "
                          f"MedAE={metrics['medae']:>5.1f} cyc | 95% Cov={metrics['coverage_95_pct']:>5.1f}% | "
                          f"Mean std={metrics['mean_std_log10']:.4f}")
        print()
        
    df_results = pd.DataFrame(results_records)
    
    # Save results to CSV
    os.makedirs(reports_dir, exist_ok=True)
    out_csv = os.path.join(reports_dir, 'baseline_results.csv')
    df_results.to_csv(out_csv, index=False)
    print(f"-> Successfully saved baseline experimental results: {out_csv}\n")
    
    # Display full summary table
    print("=" * 80)
    print("EXPERIMENTAL BENCHMARK SUMMARY (ALL SPLITS & SEED SIZES)")
    print("=" * 80)
    summary_cols = ['model', 'seed_size_N', 'test_cohort', 'mae_cycles', 'rmse_cycles', 'r2_score', 'medae_cycles', 'coverage_95_pct']
    print(df_results[summary_cols].to_string(index=False))
    print()
    
    return df_results


if __name__ == '__main__':
    train_and_evaluate_baselines()
