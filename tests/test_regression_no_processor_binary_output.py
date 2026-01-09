def test_regression_no_processor_binary_output():
    import pandas as pd
    import numpy as np
    from cv_score_predict import cv_score_predict

    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    y = X.sum(axis=1) + np.random.randn(20) * 0.1
    X_test = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])

    oof, test_pred, _, _ = cv_score_predict(
        X=X,
        y=y,
        X_test=X_test,
        pred_type="regression",
        processor=None,
        models=["lgb", "cb"],
        random_state=[1, 2, 3],  # 3 seeds
        n_splits=3,
        predict_proba=False,  # irrelevant for regression, but should not break
        verbose=0
    )

    assert len(oof) == 20
    assert len(test_pred) == 5
    assert isinstance(oof[0], float)
    assert isinstance(test_pred[0], float)