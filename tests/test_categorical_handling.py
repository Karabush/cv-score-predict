import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from cv_score_predict import cv_score_predict

def test_robust_categorical_handling():
    """
    Validates categorical handling robustness across critical edge cases:
      • Unseen categories in test data (XGBoost's strict validation)
      • Missing values (NaN) in categoricals
      • All-missing categorical column
      • High-cardinality categoricals (stress test OrdinalEncoder)
      • Processor that DYNAMICALLY CREATES new columns (binning) without breaking categorical detection
      • All three boosters (LightGBM/XGBoost/CatBoost) with unseen categories
    
    This test ensures the -1 sentinel pattern works correctly:
      • XGBoost: -1 exists in .cat.categories → passes strict validation
      • LightGBM: Handles unseen via internal mechanisms
      • CatBoost: Accepts integer categories via explicit cat_features
    """
    # === Setup: Training data with categoricals + missing values ===
    X_train = pd.DataFrame({
        "num": [1.0, 2.5, 3.1, 4.8, 5.2, 6.0, 7.3, 8.9, 9.5, 10.2],
        "cat_low_card": ["A", "B", "A", "C", "B", "A", "C", "D", "B", np.nan],  # Low cardinality + NaN
        "cat_high_card": [f"cat_{i}" for i in range(10)],  # High cardinality (10 unique)
    })
    y_train = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    
    # Test data with UNSEEN categories + missing values
    X_test = pd.DataFrame({
        "num": [11.0, 12.5, 13.1],
        "cat_low_card": ["E", np.nan, "F"],  # 'E','F' unseen in training
        "cat_high_card": ["cat_10", "cat_11", np.nan],  # Unseen + NaN
    })
    
    # === Test 1: Base processor with passthrough (ColumnTransformer) ===
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
    
    # Verify prediction shapes
    assert oof.shape == (10, 3), f"OOF shape mismatch: {oof.shape}"
    assert test_preds.shape == (3, 3), f"Test preds shape mismatch: {test_preds.shape}"
    
    # Verify no NaNs in OOF (complete coverage)
    assert not oof.isna().any().any(), "OOF contains NaNs"
    
    # Verify probabilities in [0,1]
    assert np.all((oof >= 0) & (oof <= 1)), "OOF predictions outside [0,1]"
    assert np.all((test_preds >= 0) & (test_preds <= 1)), "Test predictions outside [0,1]"
    
    # Verify artifacts count: 3 models × 2 folds × 1 seed = 6
    assert len(artifacts) == 6, f"Expected 6 artifacts, got {len(artifacts)}"
    
    # Verify categorical conversion happened correctly on test data
    fold_processor, _ = artifacts[0]
    X_test_transformed = fold_processor.transform(X_test)
    
    # All categorical columns must be category dtype
    for col in ["cat_low_card", "cat_high_card"]:
        assert pd.api.types.is_categorical_dtype(X_test_transformed[col]), \
            f"Column '{col}' not converted to category dtype (got {X_test_transformed[col].dtype})"
        
        # CRITICAL: Check CATEGORIES (display values), not codes
        assert -1 in X_test_transformed[col].cat.categories, \
            f"Column '{col}' missing -1 sentinel in categories: {X_test_transformed[col].cat.categories.tolist()}"
        
        # Verify all categories are integers (no strings) — ensures proper booster handling
        categories = X_test_transformed[col].cat.categories
        assert all(isinstance(c, (int, np.integer)) for c in categories if not pd.isna(c)), \
            f"Column '{col}' has non-integer categories: {categories.tolist()}"
    
    # === Test 2: Processor that DYNAMICALLY CREATES new columns (binning) ===
    # This validates detection happens AFTER base processor runs AND handles naming conflicts
    binning_processor = ColumnTransformer(
        [
            ("num", StandardScaler(), ["num"]),
            ("bin", KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform"), ["num"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=True,  # ← FIX: Prevents 'num' column name conflict
    ).set_output(transform="pandas")
    
    # Original categoricals ("cat_low_card", "cat_high_card") should still be detected/handled
    oof2, test_preds2, _ = cv_score_predict(
        X=X_train,
        y=y_train,
        X_test=X_test,
        pred_type="classification",
        processor=binning_processor,
        models=["lgb", "xgb"],
        random_state=[42],
        n_splits=2,
        verbose=0,
    )
    
    assert oof2.shape == (10, 2), f"Binning OOF shape mismatch: {oof2.shape}"
    assert not oof2.isna().any().any(), "Binning OOF contains NaNs"
    assert np.all((oof2 >= 0) & (oof2 <= 1)), "Binning OOF predictions outside [0,1]"
    
    # === Test 3: All-missing categorical column (edge case) ===
    X_all_missing = X_train.copy()
    X_all_missing["cat_all_missing"] = np.nan  # Entire column missing
    
    oof3, _, _ = cv_score_predict(
        X=X_all_missing,
        y=y_train,
        X_test=None,
        pred_type="classification",
        models=["lgb", "xgb", "cb"],
        random_state=[42],
        n_splits=2,
        verbose=0,
    )
    
    assert oof3.shape == (10, 3), f"All-missing OOF shape mismatch: {oof3.shape}"
    assert not oof3.isna().any().any(), "All-missing OOF contains NaNs"
    
    print("✅ test_robust_categorical_handling PASSED: All categorical edge cases handled correctly")