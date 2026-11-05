"""
Unit tests for BatteryMind Active Learning & Experiment Selection Engine (Phase 4).
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pandas as pd

from src.active_learning import (
    ActiveLearningEnvironment,
    GaussianProcessSurrogate,
    compute_acquisition_scores
)


@pytest.fixture(scope="module")
def processed_data():
    csv_path = os.path.join('data', 'processed', 'batterymind_features.csv')
    assert os.path.exists(csv_path), "Missing processed dataset"
    df = pd.read_csv(csv_path)
    return df


@pytest.fixture(scope="module")
def train_df(processed_data):
    df_tr = processed_data[processed_data['dataset_split'] == 'train'].reset_index(drop=True)
    assert len(df_tr) == 41, f"Expected 41 train cells, got {len(df_tr)}"
    return df_tr


def test_environment_candidate_pool_integrity(train_df):
    """Test 1: Verify environment pool shrinkage and zero duplicate selection."""
    feature_cols = ['log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cycle100', 'fade_100_10']
    env = ActiveLearningEnvironment(train_df, feature_cols)
    
    initial_seed = [0, 1, 2, 3, 4]
    env.reset(initial_seed)
    
    assert len(env.labeled_indices) == 5
    assert len(env.candidate_indices) == 36
    assert set(env.labeled_indices).isdisjoint(set(env.candidate_indices))
    
    # Reveal one candidate
    cand_idx = env.candidate_indices[0]
    cell_id, true_y = env.reveal_label(cand_idx)
    
    assert len(env.labeled_indices) == 6
    assert len(env.candidate_indices) == 35
    assert cand_idx in env.labeled_indices
    assert cand_idx not in env.candidate_indices
    assert true_y > 0


def test_no_candidate_label_access_before_selection(train_df):
    """Test 2: Verify candidate feature getter yields ONLY features, zero target access."""
    feature_cols = ['log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cycle100', 'fade_100_10']
    env = ActiveLearningEnvironment(train_df, feature_cols)
    env.reset([0, 1, 2])
    
    X_cand, cand_indices = env.get_candidate_features()
    
    # Must have shape (38, 5) — exactly 5 features, no target column
    assert X_cand.shape == (38, 5)
    assert len(cand_indices) == 38
    # Ensure returned array contains only floating feature values
    assert np.issubdtype(X_cand.dtype, np.floating)


def test_acquisition_strategies_validity():
    """Test 3: Verify all 4 acquisition strategies produce valid ranking indices."""
    z_mean = np.array([2.5, 3.1, 2.8, 3.0])
    z_std = np.array([0.05, 0.12, 0.08, 0.02])
    best_z = 3.0
    
    # 1. RANDOM
    rng = np.random.RandomState(42)
    s_rand = compute_acquisition_scores('RANDOM', z_mean, z_std, best_z, rng=rng)
    assert len(s_rand) == 4
    assert np.all(s_rand >= 0.0)
    
    # 2. MAX_UNCERTAINTY: should pick index 1 (highest std = 0.12)
    s_uncert = compute_acquisition_scores('MAX_UNCERTAINTY', z_mean, z_std, best_z)
    assert np.argmax(s_uncert) == 1
    
    # 3. UCB: mu + 1.96*std
    s_ucb = compute_acquisition_scores('UCB', z_mean, z_std, best_z, beta=1.96)
    expected_ucb = z_mean + 1.96 * z_std
    assert np.allclose(s_ucb, expected_ucb)
    assert np.argmax(s_ucb) == 1  # 3.1 + 1.96*0.12 = 3.335
    
    # 4. EXPECTED_IMPROVEMENT: should be non-negative and finite
    s_ei = compute_acquisition_scores('EXPECTED_IMPROVEMENT', z_mean, z_std, best_z, xi=0.01)
    assert len(s_ei) == 4
    assert np.all(s_ei >= 0.0)
    assert not np.any(np.isnan(s_ei))
    assert not np.any(np.isinf(s_ei))


def test_uncertainty_output_validity(train_df):
    """Test 4: Verify Gaussian Process surrogate produces strictly positive uncertainty."""
    feature_cols = ['log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cycle100', 'fade_100_10']
    X_train = train_df[feature_cols].values[:10]
    z_train = np.log10(train_df['cycle_life'].values[:10])
    
    surrogate = GaussianProcessSurrogate(random_state=42)
    surrogate.fit(X_train, z_train)
    
    X_test = train_df[feature_cols].values[10:15]
    z_pred, z_std = surrogate.predict(X_test)
    
    assert len(z_pred) == 5
    assert len(z_std) == 5
    assert np.all(z_std > 0.0), "Predictive standard deviation must be positive"
    assert np.all(np.isfinite(z_pred))
    assert np.all(np.isfinite(z_std))


def test_active_learning_reproducibility(train_df):
    """Test 5: Verify fixed random seeds yield deterministic candidate selection."""
    feature_cols = ['log10_var_deltaQ', 'min_deltaQ', 'mean_deltaQ', 'q_cycle100', 'fade_100_10']
    
    initial_seed = [5, 12, 19, 23, 31]
    
    # Run 1
    env1 = ActiveLearningEnvironment(train_df, feature_cols)
    env1.reset(initial_seed)
    surrogate1 = GaussianProcessSurrogate(random_state=42)
    X1, z1, _ = env1.get_labeled_data()
    surrogate1.fit(X1, z1)
    Xc1, cand1 = env1.get_candidate_features()
    zm1, zs1 = surrogate1.predict(Xc1)
    scores1 = compute_acquisition_scores('UCB', zm1, zs1, float(np.max(z1)), beta=1.96)
    pick1 = cand1[int(np.argmax(scores1))]
    
    # Run 2
    env2 = ActiveLearningEnvironment(train_df, feature_cols)
    env2.reset(initial_seed)
    surrogate2 = GaussianProcessSurrogate(random_state=42)
    X2, z2, _ = env2.get_labeled_data()
    surrogate2.fit(X2, z2)
    Xc2, cand2 = env2.get_candidate_features()
    zm2, zs2 = surrogate2.predict(Xc2)
    scores2 = compute_acquisition_scores('UCB', zm2, zs2, float(np.max(z2)), beta=1.96)
    pick2 = cand2[int(np.argmax(scores2))]
    
    assert pick1 == pick2, f"Acquisition not deterministic: {pick1} != {pick2}"


def test_no_test_set_leakage(processed_data):
    """Test 6: Verify Test 1 and Test 2 are never included in the Train candidate pool."""
    train_ids = set(processed_data[processed_data['dataset_split'] == 'train']['cell_id'])
    test1_ids = set(processed_data[processed_data['dataset_split'] == 'test1']['cell_id'])
    test2_ids = set(processed_data[processed_data['dataset_split'] == 'test2']['cell_id'])
    
    assert len(train_ids) == 41
    assert len(test1_ids) == 43
    assert len(test2_ids) == 40
    
    assert train_ids.isdisjoint(test1_ids), "Train pool overlaps with Test 1!"
    assert train_ids.isdisjoint(test2_ids), "Train pool overlaps with Test 2!"
    assert test1_ids.isdisjoint(test2_ids), "Test 1 overlaps with Test 2!"
