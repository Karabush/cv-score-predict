# cv-score-predict

A robust utility for **cross-validated ensemble prediction** that performs per‑fold early stopping and uses the early‑stopped models themselves for prediction.
Each fold trains LightGBM, XGBoost, or CatBoost with early stopping on its validation split; the resulting early‑stopped estimators generate both OOF predictions and averaged test predictions. The function also supports custom preprocessing pipelines, safe categorical encoding, repeated CV over multiple seeds, and optional return of trained models and the fitted encoder.

Designed for **kagglers, ML engineers, and data scientists** who need reliable, leakage-free CV with minimal boilerplate.

---

## ✨ Key Features

- **Multi-model ensembling**: Train and average predictions from LGBM, XGBoost and CatBoost in a single CV run.
- **Native categorical support**: Automatically encodes string/categorical columns with `OrdinalEncoder` and configures models (e.g., `cat_features` for CatBoost, `enable_categorical` for XGBoost).
- **Safe preprocessing**: Integrates any scikit-learn-compatible `processor` pipeline (e.g., `ColumnTransformer`, `Pipeline`) — fitted **per fold** to prevent data leakage.
- **Repeated CV**: Average results over multiple random seeds for stable metrics.
- **Early stopping**: Enabled by default for all models using fold-wise validation.
- **Full artifact return**: Get OOF predictions, test predictions, trained models, and fitted `OrdinalEncoder` for later use.

---

## 📥 Parameters

| Parameter | Type | Default | Description |
|----------|------|--------|-------------|
| `X` | `pd.DataFrame` | — | Training features. |
| `y` | `Union[pd.Series, np.ndarray]` | — | Target values. |
| `X_test` | `Optional[pd.DataFrame]` | `None` | Test set for final prediction. If `None`, no test predictions are returned. |
| `pred_type` | `str` | — | Either `'classification'` or `'regression'` (**required**). |
| `processor` | `Optional[object]` | `None` | Preprocessing pipeline with `fit_transform` and `transform` methods. Must return a `pd.DataFrame` (use `set_output(transform='pandas')`). If `None`, features are passed through unchanged. |
| `process_categorical` | `bool` | `True` | If `True`, object/category columns are encoded with `OrdinalEncoder` (using `-1` for missing/unseen) and converted to pandas `category` dtype for model compatibility. |
| `models` | `Union[List[str], str]` | `('lgb', 'xgb', 'cb')` | Models to ensemble. Supported: `'lgb'` (LightGBM), `'xgb'` (XGBoost), `'cb'` (CatBoost). |
| `params_dict` | `Optional[Dict[str, dict]]` | `None` | Model-specific hyperparameters. Keys: model names; values: param dicts. |
| `scoring_dict` | `Optional[Dict[str, Callable]]` | `None` | Metrics for evaluation. Keys: metric names; values: scoring functions (e.g., `roc_auc_score`). Defaults: `{'roc_auc': roc_auc_score}` (classification), `{'rmse': rmse_fn}` (regression). |
| `decision_threshold` | `float` | `0.5` | Threshold to convert probabilities to class labels (classification only). |
| `n_splits` | `int` | `5` | Number of cross-validation folds. |
| `random_state` | `Union[int, List[int]]` | `42` | Seed(s) for reproducibility. If a list, CV is repeated for each seed and results are averaged. |
| `early_stopping_rounds` | `int` | `50` | Early stopping rounds for boosting models (if not overridden in `params_dict`). |
| `verbose` | `int` | `2` | Logging level: `2` = full per-fold details, `1` = final summary, `0` = silent. |
| `return_trained` | `bool` | `False` | If `True`, returns list of trained model instances (one per model × fold × seed). |
| `return_oe` | `bool` | `False` | If `True` and `process_categorical=True`, returns the fitted `OrdinalEncoder`. |
| `predict_proba` | `bool` | `True` | For classification: if `True`, return probabilities; if `False`, return binary labels (using `decision_threshold`). Ignored for regression. |
---

## 🚀 Installation

```bash
pip install cv-score-predict
```

Requirements:

* Python ≥ 3.8
* Dependencies:
numpy, pandas, scikit-learn ≥1.4, lightgbm, xgboost, catboost

---

## 📌 Basic Usage
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from cv_score_predict import cv_score_predict

# Simulate data
X = pd.DataFrame({
    "num": [1, 2, 3, 4, 5, 6, 7, 8],
    "cat": ["A", "B", "A", "C", "B", "A", "C", "D"]
})
y = [0, 1, 0, 1, 1, 0, 1, 0]
X_test = pd.DataFrame({"num": [9, 10], "cat": ["B", "E"]})

# Optional processor (applied per fold!)
processor = make_column_transformer(
    (StandardScaler(), ["num"]),
    remainder="passthrough"
)

# Run CV with 3 seeds → results averaged over seeds & folds
oof_pred, test_pred, _, _ = cv_score_predict(
    X=X,
    y=y,
    X_test=X_test,
    pred_type="classification",
    processor=processor,
    models=["lgb", "xgb"],
    process_categorical=True,
    random_state=[42, 123, 999],
    n_splits=3,
    verbose=2,
)
```

Output will show scores per seed, then final averaged metrics.

---

## 🔧 Advanced Usage: Reuse Artifacts for New Data
```python
# Step 1: Run CV and return artifacts
oof, _, trained_models, oe = cv_score_predict(
    X,
    y,
    X_test=None,  # we'll predict manually
    pred_type="classification",
    processor=processor,
    models=["lgb", "cb"],
    process_categorical=True,
    random_state=[42, 123],
    n_splits=5,
    return_trained=True,
    return_oe=True,
)

# Step 2: For deployment: refit processor on FULL TRAINING data
# First: encode categoricals using returned oe
cat_cols = ["cat"]
X_full = X.copy()
X_full[cat_cols] = oe.transform(X_full[cat_cols]).astype('category')
X_new = pd.DataFrame({"num": [7, 8], "cat": [None, "A"]})

# Fit processor on full encoded data
processor = make_column_transformer(
    (StandardScaler(), ["num"]),
    remainder="passthrough"
)
processor.fit(X_full)

# Apply to new data
X_new_proc = X_new.copy()
X_new_proc[cat_cols] = oe.transform(X_new_proc[cat_cols]).astype('category')
X_new_proc = processor.transform(X_new_proc)

# Predict with all trained models and average
preds = [model.predict_proba(X_new_proc)[:, 1] for model in trained_models]
final_pred = np.mean(preds, axis=0)
```
## 📝 Notes
Categorical columns are encoded with OrdinalEncoder(dtype=np.int32) and converted to category dtype for model compatibility.
Always use set_output(transform="pandas") in sklearn pipelines to preserve dtypes.
The processor used in CV is refit on each fold to prevent data leakage, so there is no single global version. For deployment, refit your preprocessing pipeline on the full training set (as shown in the advanced example).

## 📄 License
This project is licensed under the MIT License.
See the LICENSE file for details.