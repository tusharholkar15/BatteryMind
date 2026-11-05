"""
Unit Tests for Streamlit Dashboard (Phase 5)
BatteryMind — Small-Data Battery Performance Optimization using Active Learning
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app


def test_app_file_exists():
    """Verify that app.py exists and is a valid file."""
    app_path = os.path.join(os.path.dirname(__file__), '..', 'app.py')
    assert os.path.exists(app_path), "app.py does not exist"
    assert os.path.getsize(app_path) > 1000, "app.py is unexpectedly small"


def test_required_data_and_report_files_exist():
    """Verify all prerequisite processed data and experiment reports exist."""
    assert os.path.exists(app.PROCESSED_DATA_PATH), f"Missing {app.PROCESSED_DATA_PATH}"
    assert os.path.exists(app.AL_RESULTS_PATH), f"Missing {app.AL_RESULTS_PATH}"
    assert os.path.exists(app.BASELINE_RESULTS_PATH), f"Missing {app.BASELINE_RESULTS_PATH}"


def test_load_processed_features():
    """Verify load_processed_features returns the 124-cell dataframe."""
    df = app.load_processed_features()
    assert df is not None, "load_processed_features returned None"
    assert len(df) == 124, f"Expected 124 rows, got {len(df)}"
    for col in app.FEATURE_COLS + ['cell_id', 'cycle_life', 'dataset_split']:
        assert col in df.columns, f"Missing expected column {col}"
    assert df[app.FEATURE_COLS].isna().sum().sum() == 0, "Found NaN in feature columns"


def test_load_al_results():
    """Verify load_al_results loads active learning trajectories."""
    al_df = app.load_al_results()
    assert al_df is not None, "load_al_results returned None"
    assert len(al_df) > 0, "Active learning results DataFrame is empty"
    expected_cols = ['step', 'n_labeled', 'strategy', 'initial_N', 'trial_idx', 'best_cycle_life_discovered']
    for c in expected_cols:
        assert c in al_df.columns, f"Expected column {c} missing in AL results"


def test_load_baseline_results():
    """Verify load_baseline_results loads baseline metrics."""
    base_df = app.load_baseline_results()
    assert base_df is not None, "load_baseline_results returned None"
    assert len(base_df) > 0, "Baseline results DataFrame is empty"
    for c in ['model', 'seed_size_N', 'test_cohort', 'mae_cycles', 'rmse_cycles', 'r2_score']:
        assert c in base_df.columns, f"Expected column {c} missing in baseline results"


def test_load_cell_curve_valid_and_invalid():
    """Verify load_cell_curve loads valid curves and gracefully returns None on invalid IDs."""
    # Test valid cell
    curve = app.load_cell_curve("train", "train_cell1")
    assert curve is not None, "Failed to load train_cell1 curve"
    assert curve.shape == (1000, 99), f"Expected shape (1000, 99), got {curve.shape}"

    # Test invalid cell ID
    invalid_curve = app.load_cell_curve("invalid_split", "nonexistent_cell")
    assert invalid_curve is None, "Expected None for nonexistent cell curve"


def test_trained_gp_model_loads():
    """Verify that get_trained_gp_model initializes, trains on 41 train cells, and returns scaler & model."""
    scaler, gp = app.get_trained_gp_model()
    assert scaler is not None, "Scaler is None"
    assert gp is not None, "GP model is None"
    assert hasattr(gp, "predict"), "gp does not have predict method"


def test_predictor_model_valid_inference():
    """Verify predictor model returns physical cycle life and valid positive uncertainty."""
    scaler, gp = app.get_trained_gp_model()
    
    # Typical feature vector within nominal range
    x_test = np.array([[-3.85, -0.040, -0.019, 1.050, -0.010]])
    x_scaled = scaler.transform(x_test)
    z_pred, z_std = gp.predict(x_scaled, return_std=True)

    y_pred = float(10.0 ** z_pred[0])
    std_val = float(z_std[0])

    # Assert physical reasonableness
    assert 100 <= y_pred <= 3000, f"Predicted cycle life {y_pred} outside reasonable bounds"
    assert std_val > 0.0, f"Uncertainty must be strictly positive, got {std_val}"
    
    # Assert credible interval ordering
    y_lower = 10.0 ** (z_pred[0] - 1.96 * std_val)
    y_upper = 10.0 ** (z_pred[0] + 1.96 * std_val)
    assert y_lower < y_pred < y_upper, f"Credible interval ordering invalid: {y_lower} < {y_pred} < {y_upper}"


def test_predictor_handles_extreme_or_out_of_bounds_inputs():
    """Verify that extreme/unusual inputs produce predictions without crashing."""
    scaler, gp = app.get_trained_gp_model()

    # Extreme degraded cell
    x_extreme_bad = np.array([[-1.5, -0.4, -0.2, 0.5, -0.3]])
    x_bad_scaled = scaler.transform(x_extreme_bad)
    z_pred_bad, z_std_bad = gp.predict(x_bad_scaled, return_std=True)
    assert not np.isnan(z_pred_bad[0]), "GP produced NaN on extreme input"
    assert z_std_bad[0] > 0.0, "GP produced non-positive std"

    # Extreme high-longevity cell
    x_extreme_good = np.array([[-6.5, -0.001, -0.0001, 1.15, 0.0]])
    x_good_scaled = scaler.transform(x_extreme_good)
    z_pred_good, z_std_good = gp.predict(x_good_scaled, return_std=True)
    assert not np.isnan(z_pred_good[0]), "GP produced NaN on extreme input"
    assert z_std_good[0] > 0.0, "GP produced non-positive std"


def test_all_six_dashboard_pages_exist():
    """Verify that all 6 required dashboard sections exist as callable functions in app.py."""
    expected_pages = [
        "page_overview",
        "page_predictor",
        "page_al_simulator",
        "page_explorer",
        "page_comparison",
        "page_methodology"
    ]
    for page_fn in expected_pages:
        assert hasattr(app, page_fn), f"Missing page function: {page_fn}"
        assert callable(getattr(app, page_fn)), f"{page_fn} is not callable"


def test_sidebar_navigation_options():
    """Verify the navigation options defined in app.py match the 6 required pages."""
    import inspect
    sidebar_code = inspect.getsource(app.render_sidebar)
    required_titles = [
        "Overview",
        "Cycle-Life Predictor",
        "Active Learning Simulator",
        "Battery Explorer",
        "Strategy Comparison",
        "Methodology"
    ]
    for title in required_titles:
        assert title in sidebar_code, f"Missing page title in sidebar: {title}"


def test_all_pages_execute_without_exception():
    """Verify each page rendering function executes without unhandled exceptions."""
    pages = [
        app.page_overview,
        app.page_predictor,
        app.page_al_simulator,
        app.page_explorer,
        app.page_comparison,
        app.page_methodology,
    ]
    for page_fn in pages:
        try:
            page_fn()
        except Exception as e:
            pytest.fail(f"Page function {page_fn.__name__} raised unexpected exception: {e}")

