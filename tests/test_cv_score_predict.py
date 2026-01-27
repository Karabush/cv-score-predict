def test_classification_with_categorical_encoding_and_preprocessor():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from cv_score_predict import cv_score_predict

    # Create data with numerical and categorical features
    X = pd.DataFrame({
        "num": [1.0, 2.5, 3.1, 4.8, 5.2, 6.0, 7.3, 8.9],
        "cat": ["X", "Y", "X", "Z", "Y", "X", "Z", "W"]
    })
    y = [0, 1, 0, 1, 1, 0, 1, 0]
    X_test = pd.DataFrame({"num": [9.1, 10.2], "cat": ["Y", "V"]})  # 'V' is unseen

    # Define a column-wise preprocessor that returns a DataFrame
    processor = ColumnTransformer(
        [("num_", StandardScaler(), ["num"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    # Run CV with multiple seeds and models
    oof_df, test_df, trained_artifacts = cv_score_predict(
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
        verbose=0,
        return_raw_test_preds=True,  
    )

    # Shape checks
    assert oof_df.shape == (8, 4)  # 2 models × 2 seeds ✓
    assert test_df.shape == (2, 8)  # 2 models × 2 folds × 2 seeds ✓

    # No NaNs in OOF
    assert not oof_df.isna().any().any(), "OOF must be fully populated"

    # Probabilities in [0,1]
    assert np.all((oof_df >= 0) & (oof_df <= 1)), "OOF predictions must be valid probabilities"
    assert np.all((test_df >= 0) & (test_df <= 1)), "Test predictions must be valid probabilities"

    # Trained artifacts count: 2 models × 2 folds × 2 seeds = 8
    assert len(trained_artifacts) == 8

    # Check encoding of unseen category
    fold_processor, _ = trained_artifacts[0]
    X_test_transformed = fold_processor.transform(X_test)
    cat_encoded = X_test_transformed["cat"]
    assert cat_encoded.iloc[1] == -1, "Unseen category 'V' must encode to -1"
    assert isinstance(cat_encoded.dtype, pd.CategoricalDtype), "Encoded column must be categorical dtype"
    assert not np.allclose(X_test_transformed["num"], X_test["num"]), "Numerical column must be scaled"