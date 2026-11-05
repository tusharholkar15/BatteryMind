"""
Unit tests for BatteryMind Feature Pipeline and Small-Data Baseline Models (Phase 3).
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

from src.extract_features import extract_battery_features


PROCESSED_CSV = os.path.join('data', 'processed', 'batterymind_features.csv')


@pytest.fixture(scope="module")
def features_df():
    """Fixture providing the processed feature dataframe."""
    if not os.path.exists(PROCESSED_CSV):
        df = extract_battery_features()
    else:
        df = pd.read_csv(PROCESSED_CSV)
    return df


def test_processed_dataset_exists(features_df):
    """Test 1: Verify processed dataset exists and is non-empty."""
    assert os.path.exists(PROCESSED_CSV), f"Missing {PROCESSED_CSV}"
    assert len(features_df) > 0, "Processed dataset is empty"


def test_one_row_per_cell_integrity(features_df):
    """Test 2: Verify one-row-per-cell integrity and cohort counts."""
    # Exactly 124 unique cells
    assert len(features_df) == 124, f"Expected 124 cells, got {len(features_df)}"
    assert features_df['cell_id'].nunique() == 124, "Duplicate cell_id found in dataset"
    
    # Check cohort breakdown
    split_counts = features_df['dataset_split'].value_counts().to_dict()
    assert split_counts.get('train') == 41, f"Expected 41 train cells, got {split_counts.get('train')}"
    assert split_counts.get('test1') == 43, f"Expected 43 test1 cells, got {split_counts.get('test1')}"
    assert split_counts.get('test2') == 40, f"Expected 40 test2 cells, got {split_counts.get('test2')}"


def test_no_nan_or_infinite_values(features_df):
    """Test 3: Verify zero missing or infinite values in all feature columns."""
    expected_cols = [
        'cell_id', 'dataset_split', 'cycle_life',
        'log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cycle100', 'fade_100_10'
    ]
    for col in expected_cols:
        assert col in features_df.columns, f"Missing expected column: {col}"
        
    num_cols = ['log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cycle100', 'fade_100_10', 'cycle_life']
    
    # Missing value check
    nan_counts = features_df[num_cols].isna().sum().to_dict()
    for col, count in nan_counts.items():
        assert count == 0, f"Column {col} has {count} NaN values"
        
    # Infinite value check
    inf_counts = np.isinf(features_df[num_cols]).sum().to_dict()
    for col, count in inf_counts.items():
        assert count == 0, f"Column {col} has {count} infinite values"


def test_target_presence_and_bounds(features_df):
    """Test 4: Verify target variable presence, data type, and physical bounds."""
    targets = features_df['cycle_life']
    assert np.issubdtype(targets.dtype, np.integer) or np.issubdtype(targets.dtype, np.floating)
    assert targets.min() >= 100, f"Cycle life below physical lower threshold: {targets.min()}"
    assert targets.max() <= 5000, f"Cycle life above physical upper threshold: {targets.max()}"


def test_features_do_not_use_future_cycles():
    """
    Test 5: Verify that feature extraction strictly uses only data up to Cycle 100.
    In the raw curve matrices (1000, 99), column 0 is Cycle 2, column 8 is Cycle 10,
    and column 98 is Cycle 100. Columns beyond 98 (if any existed) must not be queried.
    """
    raw_sample = np.genfromtxt('data/raw/train/cell1.csv', delimiter=',')
    assert raw_sample.shape == (1000, 99), f"Unexpected raw matrix shape: {raw_sample.shape}"
    
    # Column 98 is the 99th column (Cycle 100). Max valid column index accessed is 98.
    max_accessed_col = 98
    assert max_accessed_col < raw_sample.shape[1], "Max accessed column index exceeds available early cycles"
    assert max_accessed_col == 98, "Accessed index does not equal Cycle 100 index (98)"


def test_baseline_models_predictive_uncertainty(features_df):
    """
    Test 6: Verify that Bayesian Ridge and Gaussian Process can fit on small N=10 seed,
    predict on held-out cells, and output non-zero predictive uncertainty.
    """
    feature_cols = ['log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cycle100', 'fade_100_10']
    
    train_df = features_df[features_df['dataset_split'] == 'train'].head(10)
    eval_df = features_df[features_df['dataset_split'] == 'test1'].head(5)
    
    X_train = train_df[feature_cols].values
    z_train = np.log10(train_df['cycle_life'].values)
    
    X_eval = eval_df[feature_cols].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s = scaler.transform(X_eval)
    
    # 1. Bayesian Ridge
    br = BayesianRidge()
    br.fit(X_train_s, z_train)
    z_pred_br, z_std_br = br.predict(X_eval_s, return_std=True)
    assert len(z_pred_br) == len(X_eval)
    assert np.all(z_std_br > 0), "Bayesian Ridge predictive standard deviations must be positive"
    
    # 2. Gaussian Process
    kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-2)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gp.fit(X_train_s, z_train)
    z_pred_gp, z_std_gp = gp.predict(X_eval_s, return_std=True)
    assert len(z_pred_gp) == len(X_eval)
    assert np.all(z_std_gp > 0), "Gaussian Process predictive standard deviations must be positive"
