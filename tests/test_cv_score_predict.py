"""
Comprehensive test suite for cv_score_predict library.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.compose import ColumnTransformer
from cv_score_predict import cv_score_predict


# =============================================================================
# TEST 1: Robust Categorical Handling
# =============================================================================

def test_robust_categorical_handling():
    X_train = pd.DataFrame({
        "num": [1.0, 2.5, 3.1, 4.8, 5.2, 6.0, 7.3, 8.9, 9.5, 10.2],
        "cat_low_card": ["A", "B", "A", "C", "B", "A", "C", "D", "B", np.nan],
        "cat_high_card": [f"cat_{i}" for i in range(10)],
    })
    y_train = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    
    X_test = pd.DataFrame({
        "num": [11.0, 12.5, 13.1],
        "cat_low_card": ["E", np.nan, "F"],
        "cat_high_card": ["cat_10", "cat_11", np.nan],
    })
    
    processor = ColumnTransformer(
        [("num", StandardScaler(), ["num"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    
    oof, test_preds, artifacts = cv_score_predict(
        X=X_train,
        y=y_train,
        X_test=X_test,
        pred_type="classification",
        processor=processor,
        models=["lgb", "xgb", "cb"],
        random_state=[42],
        n_splits=2,
        return_trained=True,
        verbose=0,
    )
    
    assert oof.shape == (10, 3)
    assert test_preds.shape == (3, 3)
    assert not oof.isna().any().any()
    assert np.all((oof >= 0) & (oof <= 1))
    assert len(artifacts) == 6
    
    fold_processor, _ = artifacts[0]
    X_test_transformed = fold_processor.transform(X_test)
    
    for col in ["cat_low_card", "cat_high_card"]:
        assert isinstance(X_test_transformed[col].dtype, pd.CategoricalDtype)
        assert -1 in X_test_transformed[col].cat.categories
    
    print("✅ test_robust_categorical_handling PASSED")


# =============================================================================
# TEST 2: Custom CV Splitter Functionality
# =============================================================================

def test_custom_cv_splitter():
    np.random.seed(42)
    n_samples = 100
    
    X_train = pd.DataFrame({
        "num": np.random.randn(n_samples),
        "cat": [f"cat_{i % 10}" for i in range(n_samples)],
    })
    y_train = np.random.randint(0, 2, n_samples)
    
    X_test = pd.DataFrame({
        "num": np.random.randn(20),
        "cat": [f"cat_{i % 10}" for i in range(20)],
    })
    
    quarter_groups = np.repeat([0, 1, 2, 3], 25)
    
    processor = ColumnTransformer(
        [("num", StandardScaler(), ["num"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    
    print("\n[TEST 1] cv_splitter=None (default) → regular CV")
    
    oof_default, test_preds_default, _ = cv_score_predict(
        X=X_train,
        y=y_train,
        X_test=X_test,
        pred_type="classification",
        processor=processor,
        models=["lgb", "xgb"],
        random_state=[42],
        n_splits=5,
        cv_splitter=None,
        cv_groups=None,
        verbose=0,
    )
    
    assert oof_default.shape == (100, 2)
    print("✓ Default behavior (cv_splitter=None) works correctly")
    
    print("\n[TEST 2] cv_groups without cv_splitter → raises ValueError")
    
    try:
        cv_score_predict(
            X=X_train,
            y=y_train,
            X_test=X_test,
            pred_type="classification",
            models=["lgb"],
            random_state=[42],
            n_splits=5,
            cv_splitter=None,
            cv_groups=quarter_groups,
            verbose=0,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e).lower()
        assert "cv_groups" in error_msg and "cv_splitter" in error_msg
        print(f"✓ Correctly raises ValueError: {e}")
    
    print("\n[TEST 3] Custom splitter (GroupKFold) with cv_groups")
    
    custom_splitter = GroupKFold(n_splits=4)
    
    oof_grouped, test_preds_grouped, artifacts_grouped = cv_score_predict(
        X=X_train,
        y=y_train,
        X_test=X_test,
        pred_type="classification",
        processor=processor,
        models=["lgb", "xgb"],
        random_state=[42],
        n_splits=5,
        cv_splitter=custom_splitter,
        cv_groups=quarter_groups,
        return_trained=True,
        verbose=0,
    )
    
    assert oof_grouped.shape == (100, 2)
    assert test_preds_grouped.shape == (20, 2)
    assert len(artifacts_grouped) == 8
    print("✓ Custom splitter with cv_groups works correctly")
    
    print("\n[TEST 4] Multiple seeds with custom splitter")
    
    oof_multi_seed, test_preds_multi_seed, _ = cv_score_predict(
        X=X_train,
        y=y_train,
        X_test=X_test,
        pred_type="classification",
        processor=processor,
        models=["lgb"],
        random_state=[42, 99],
        cv_splitter=GroupKFold(n_splits=3),
        cv_groups=quarter_groups,
        verbose=0,
    )
    
    assert oof_multi_seed.shape == (100, 2)
    expected_cols = ["lgb_seed_42", "lgb_seed_99"]
    assert list(oof_multi_seed.columns) == expected_cols
    print("✓ Multiple seeds with custom splitter works correctly")
    
    print("\n[TEST 5] Custom splitter without groups (standard KFold)")
    
    standard_splitter = KFold(n_splits=3, shuffle=True, random_state=42)
    
    oof_standard, test_preds_standard, _ = cv_score_predict(
        X=X_train,
        y=y_train,
        X_test=X_test,
        pred_type="classification",
        processor=processor,
        models=["lgb"],
        random_state=[42],
        cv_splitter=standard_splitter,
        cv_groups=None,
        verbose=0,
    )
    
    assert oof_standard.shape == (100, 1)
    print("✓ Custom splitter without groups works correctly")
    
    print("\n[TEST 6] Regression with custom splitter")
    
    y_reg = np.random.randn(n_samples)
    
    oof_reg, test_preds_reg, _ = cv_score_predict(
        X=X_train,
        y=y_reg,
        X_test=X_test,
        pred_type="regression",
        processor=processor,
        models=["lgb"],
        random_state=[42],
        cv_splitter=GroupKFold(n_splits=4),
        cv_groups=quarter_groups,
        verbose=0,
    )
    
    assert oof_reg.shape == (100, 1)
    assert test_preds_reg.shape == (20, 1)
    assert np.all(np.isfinite(oof_reg.values))
    print("✓ Regression with custom splitter works correctly")
    
    print("\n✅ test_custom_cv_splitter PASSED")


# =============================================================================
# TEST 3: Prediction Structures and Modes
# =============================================================================

def test_cv_prediction_structures_and_modes():
    X_clf = pd.DataFrame({"num": range(20), "cat": list("ABCDE") * 4})
    y_clf = [0, 1] * 10
    X_test_clf = pd.DataFrame({"num": [20, 21], "cat": ["F", "G"]})
    
    oof_clf, test_clf_raw, _ = cv_score_predict(
        X=X_clf,
        y=y_clf,
        X_test=X_test_clf,
        pred_type="classification",
        models=["lgb", "xgb"],
        random_state=[42, 99],
        n_splits=3,
        return_raw_test_preds=True,
        verbose=0,
    )
    
    assert oof_clf.shape == (20, 4)
    expected_oof_cols = ["lgb_seed_42", "xgb_seed_42", "lgb_seed_99", "xgb_seed_99"]
    assert list(oof_clf.columns) == expected_oof_cols
    
    assert test_clf_raw.shape == (2, 12)
    expected_raw_cols = [
        f"{m}_seed_{seed}_fold_{fold}"
        for seed in [42, 99]
        for fold in range(3)
        for m in ["lgb", "xgb"]
    ]
    assert list(test_clf_raw.columns) == expected_raw_cols
    
    _, test_clf_avg, _ = cv_score_predict(
        X=X_clf,
        y=y_clf,
        X_test=X_test_clf,
        pred_type="classification",
        models=["lgb", "xgb"],
        random_state=[42, 99],
        n_splits=3,
        return_raw_test_preds=False,
        verbose=0,
    )
    
    assert test_clf_avg.shape == (2, 4)
    assert list(test_clf_avg.columns) == expected_oof_cols
    
    for col in expected_oof_cols:
        raw_cols_for_col = [c for c in test_clf_raw.columns if c.startswith(col)]
        avg_from_raw = test_clf_raw[raw_cols_for_col].mean(axis=1)
        pd.testing.assert_series_equal(
            test_clf_avg[col], 
            avg_from_raw, 
            check_names=False,
            rtol=1e-5,
        )
    
    oof_thresh, test_thresh, _ = cv_score_predict(
        X=X_clf,
        y=y_clf,
        X_test=X_test_clf,
        pred_type="classification",
        models=["lgb"],
        random_state=[42],
        n_splits=2,
        predict_proba=False,
        decision_threshold=0.5,
        verbose=0,
    )
    
    oof_vals = set(oof_thresh.values.flatten())
    assert oof_vals.issubset({0, 1})
    
    test_vals = set(test_thresh.values.flatten())
    assert test_vals.issubset({0, 1})
    
    y_reg = np.random.randn(20)
    X_test_reg = pd.DataFrame({"num": [20, 21], "cat": ["F", "G"]})
    
    oof_reg, test_reg, artifacts_reg = cv_score_predict(
        X=X_clf,
        y=y_reg,
        X_test=X_test_reg,
        pred_type="regression",
        models=["lgb", "xgb"],
        random_state=[42],
        n_splits=2,
        return_trained=True,
        verbose=0,
    )
    
    assert np.all(np.isfinite(oof_reg.values))
    assert np.all(np.isfinite(test_reg.values))
    assert len(artifacts_reg) == 4
    for proc, model in artifacts_reg:
        assert hasattr(proc, "transform")
        assert hasattr(model, "predict")
    
    print("✅ test_cv_prediction_structures_and_modes PASSED")


# =============================================================================
# TEST 4: Edge Cases and Error Handling
# =============================================================================

def test_edge_cases_and_errors():
    X = pd.DataFrame({"num": range(10)})
    y = [0, 1] * 5
    
    print("\n[TEST 1] Empty X_test → raises ValueError")
    try:
        cv_score_predict(
            X=X,
            y=y,
            X_test=pd.DataFrame(),
            pred_type="classification",
            models=["lgb"],
            verbose=0,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "x_test" in str(e).lower() and "empty" in str(e).lower()
        print(f"✓ Correctly raises ValueError: {e}")
    
    print("\n[TEST 2] X/y length mismatch → raises ValueError")
    try:
        cv_score_predict(
            X=X,
            y=[0, 1] * 3,
            pred_type="classification",
            models=["lgb"],
            verbose=0,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "same number of samples" in str(e).lower()
        print(f"✓ Correctly raises ValueError: {e}")
    
    print("\n[TEST 3] Invalid pred_type → raises ValueError")
    try:
        cv_score_predict(
            X=X,
            y=y,
            pred_type="invalid",
            models=["lgb"],
            verbose=0,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "classification" in str(e).lower() or "regression" in str(e).lower()
        print(f"✓ Correctly raises ValueError: {e}")
    
    print("\n[TEST 4] Invalid model keys → raises ValueError")
    try:
        cv_score_predict(
            X=X,
            y=y,
            pred_type="classification",
            models=["invalid_model"],
            verbose=0,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_model" in str(e).lower() or "allowed" in str(e).lower()
        print(f"✓ Correctly raises ValueError: {e}")
    
    print("\n[TEST 5] Invalid processor → raises TypeError")
    try:
        cv_score_predict(
            X=X,
            y=y,
            pred_type="classification",
            models=["lgb"],
            processor="not_a_processor",
            verbose=0,
        )
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "fit_transform" in str(e).lower() or "transform" in str(e).lower()
        print(f"✓ Correctly raises TypeError: {e}")
    
    print("\n✅ test_edge_cases_and_errors PASSED")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Running cv_score_predict Test Suite")
    print("=" * 70)
    
    test_robust_categorical_handling()
    print()
    test_custom_cv_splitter()
    print()
    test_cv_prediction_structures_and_modes()
    print()
    test_edge_cases_and_errors()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)