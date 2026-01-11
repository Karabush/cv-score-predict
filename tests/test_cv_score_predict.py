def test_classification_full_pipeline():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from cv_score_predict import cv_score_predict

    # Data with categoricals and missing
    X = pd.DataFrame({
        "num": [1.0, 2.5, 3.1, 4.8, 5.2, 6.0, 7.3, 8.9],
        "cat": ["X", "Y", "X", "Z", "Y", "X", "Z", "W"]
    })
    y = [0, 1, 0, 1, 1, 0, 1, 0]
    X_test = pd.DataFrame({"num": [9.1, 10.2], "cat": ["Y", "V"]})  # V is unseen

    processor = ColumnTransformer([
        ("num", StandardScaler(), ["num"]),
        ("cat", "passthrough", ["cat"])
    ]).set_output(transform="pandas")

    oof, test_pred, trained_models, fitted_processor = cv_score_predict(
        X=X,
        y=y,
        X_test=X_test,
        pred_type="classification",
        processor=processor,
        process_categorical=True,
        models=["lgb", "xgb"],
        random_state=[42, 99],
        n_splits=2,
        return_trained=True,
        verbose=0
    )

    # Assertions
    assert len(oof) == len(X)
    assert len(test_pred) == len(X_test)
    assert test_pred.min() >= 0 and test_pred.max() <= 1  # probabilities
    assert len(trained_models) == 2 * 2 * 2  # 2 models × 2 folds × 2 seeds
    assert fitted_processor is not None

    # Check that unseen category 'V' was encoded as -1
    X_enc = fitted_processor.transform(X_test)
    assert X_enc[["cat"]].iloc[1, 0] == -1  # 'V' → -1