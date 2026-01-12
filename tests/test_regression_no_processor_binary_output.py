def test_regression_without_preprocessor():
    import pandas as pd
    import numpy as np
    from cv_score_predict import cv_score_predict

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
    y = X.sum(axis=1) + np.random.randn(30) * 0.1  # linear target + noise
    X_test = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])

    # Run regression CV with multiple seeds and models
    oof_df, test_df, trained_artifacts = cv_score_predict(
        X=X,
        y=y,
        X_test=X_test,
        pred_type="regression",
        processor=None,          # no preprocessing
        models=["lgb", "cb"],    # two models
        random_state=[1, 2, 3],  # 3 seeds
        n_splits=3,              # 3 folds
        predict_proba=False,     # ignored in regression
        return_trained=True,
        verbose=0
    )

    # --- Shape validation ---
    # OOF: (n_samples, n_models × n_seeds) = (30, 2 × 3) = (30, 6)
    assert oof_df.shape == (30, 6), f"Expected OOF shape (30, 6), got {oof_df.shape}"
    # Test: (n_test, n_models × n_folds × n_seeds) = (5, 2 × 3 × 3) = (5, 18)
    assert test_df.shape == (5, 18), f"Expected test shape (5, 18), got {test_df.shape}"

    # --- Data type validation ---
    # All values should be float-like
    assert oof_df.dtypes.apply(lambda dt: np.issubdtype(dt, np.floating)).all(), \
        "All OOF columns must be floating dtype"
    assert test_df.dtypes.apply(lambda dt: np.issubdtype(dt, np.floating)).all(), \
        "All test columns must be floating dtype"

    # --- Regression outputs are unbounded ---
    # Just verify they are not artificially clipped to [0,1]
    # (e.g., some values should be < 0 or > 1 given standard normal inputs)
    oof_values = oof_df.values.ravel()
    test_values = test_df.values.ravel()
    # Since y ≈ N(0, sqrt(3)) ± noise, predictions should span negative and positive
    assert np.any(oof_values < 0) or np.any(oof_values > 1), \
        "Regression predictions should not be confined to [0,1]"
    assert np.any(test_values < 0) or np.any(test_values > 1), \
        "Test regression predictions should not be confined to [0,1]"

    # --- Trained artifacts count ---
    expected_artifacts = len(["lgb", "cb"]) * 3 * 3  # models × folds × seeds
    assert len(trained_artifacts) == expected_artifacts, \
        f"Expected {expected_artifacts} trained artifacts, got {len(trained_artifacts)}"

    # --- Processor behavior: numeric-only, so should be identity ---
    first_proc, first_model = trained_artifacts[0]
    # Even though process_categorical=True, there are no object/categorical cols
    # So transform should return identical DataFrame
    X_trans = first_proc.transform(X)
    pd.testing.assert_frame_equal(X_trans, X, check_dtype=True)

    # Also check that cat_cols_ is empty
    assert hasattr(first_proc, 'cat_cols_'), "Processor should have cat_cols_ attribute"
    assert first_proc.cat_cols_ == [], "No categorical columns expected"