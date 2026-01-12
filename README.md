# cv-score-predict

A robust utility for **cross-validated ensemble prediction** that performs per‑fold early stopping and exposes raw model outputs for advanced stacking, diagnostics, or custom ensembling.
Each fold trains LightGBM, XGBoost, or CatBoost with early stopping on its validation split; the resulting estimators generate raw out-of-fold (OOF) and test predictions from every model, fold, and seed. The function supports custom preprocessing pipelines, dynamic per-fold categorical encoding, repeated CV over multiple seeds, and when requested — returns trained models along with their corresponding fold-specific preprocessors.

Designed for **kagglers, ML engineers, and data scientists** who need reliable, leakage-free CV with minimal boilerplate.

---

## ✨ Key Features

- **Per‑fold early stopping**: Each fold trains with early stopping on its validation split and uses the early‑stopped estimator for OOF and test predictions.
- **Raw prediction matrices**: Returns two DataFrames:
    - `oof_preds_df`: raw OOF predictions — one column per (model, seed). Predictions from all folds for a given (model, seed) are stitched together into a single column.
    - `test_preds_df`: raw test predictions — one column per (model, seed, fold), i.e., one prediction per fitted model.
    - → Perfect for averaging, stacking, model blending, or error analysis.
- **Multi-model support**: Train LightGBM (`'lgb'`), XGBoost (`'xgb'`), and CatBoost (`'cb'`) in the same CV loop.
- **Safe fold-wise preprocessing**: Accepts any scikit-learn–compatible processor with `fit_transform`/`transform`. Fitted independently per fold to prevent data leakage.
- **Dynamic categorical handling**: When `process_categorical=True`, the function:
    - Detects object/category columns after the base processor runs,
    - Fits an `OrdinalEncoder` per fold (using `-1` for missing/unseen categories),
    - Converts encoded columns to pandas `'category'` dtype so boosting libraries auto-detect them,
    - Automatically sets model-specific flags: `enable_categorical=True` for XGBoost, `cat_features=col_names` for CatBoost. LightGBM requires no extra flag thanks to pandas categorical dtype.
- **Repeated CV over seeds**: Accepts a single seed or a list of seeds; CV is repeated for each seed, and all raw predictions are preserved.
- **Flexible scoring and thresholding**: 
    - Custom `scoring_dict` supported (e.g., accuracy, log loss, RMSE).
    - Defaults: ROC AUC for classification, RMSE for regression.
    - For classification, return probabilities (`predict_proba=True`) or binary labels (`predict_proba=False`)  using `decision_threshold.
- **Artifact return**: When `return_trained=True`, returns a list of tuples (`fold_processor, model`) — one per model × fold × seed — where `fold_processor` is the preprocessor fitted on that fold’s training data,
- **Transparent, diagnostic-rich logging**: 
    With `verbose=2` (default), the function prints:
    - Per-fold scores for every model,
    - Stacked (mean of model predictions) score per fold,
    - Per-seed mean scores (by model and stacked),
    - Final cross-seed summary of mean CV performance.
    - → Enables instant diagnosis of model instability, fold bias, or seed sensitivity — no extra code needed.

---

## 📥 Parameters

| Parameter | Type | Default | Description |
|----------|------|--------|-------------|
| `X` | `pd.DataFrame` | — | Training features. |
| `y` | `Union[pd.Series, np.ndarray]` | — | Target values. |
| `X_test` | `Optional[pd.DataFrame]` | `None` | Test set for final prediction. If `None`, no test predictions are returned. |
| `pred_type` | `str` | — | Either `'classification'` or `'regression'` (**required**). |
| `processor` | `Optional[object]` | `None` | Preprocessing pipeline with `fit_transform` and `transform` methods. Must return a `pd.DataFrame` (use `set_output(transform='pandas')`). If `None`, features are passed through unchanged. |
| `process_categorical` | `bool` | `True` | If True, object/category columns in the processor’s output are encoded per fold with OrdinalEncoder (using -1 for missing/unseen) and converted to pandas 'category' dtype. |
| `models` | `Union[List[str], str]` | `('lgb', 'xgb', 'cb')` | Models to ensemble. Supported: `'lgb'` (LightGBM), `'xgb'` (XGBoost), `'cb'` (CatBoost). |
| `params_dict` | `Optional[Dict[str, dict]]` | `None` | Model-specific hyperparameters. Keys: model names; values: param dicts. |
| `scoring_dict` | `Optional[Dict[str, Callable]]` | `None` | Metrics for evaluation. Keys: metric names; values: scoring functions (e.g., `roc_auc_score`). Defaults: `{'roc_auc': roc_auc_score}` (classification), `{'rmse': rmse_fn}` (regression). |
| `decision_threshold` | `float` | `0.5` | Threshold to convert probabilities to class labels (classification only). |
| `n_splits` | `int` | `5` | Number of cross-validation folds. |
| `random_state` | `Union[int, List[int]]` | `42` | Seed(s) for reproducibility. If a list, CV is repeated for each seed and results are averaged. |
| `early_stopping_rounds` | `int` | `50` | Early stopping rounds for boosting models (if not overridden in `params_dict`). |
| `verbose` | `int` | `2` | Logging level: `2` = full per-fold details, `1` = final summary, `0` = silent. |
| `return_trained` | `bool` | `False` | If True, returns a list of (fold_processor, model) tuples (one per model × fold × seed). |
| `predict_proba` | `bool` | `True` | For classification: if `True`, return probabilities; if `False`, return binary labels (using `decision_threshold`). Ignored for regression. |
---

## 🚀 Installation

```bash
pip install cv-score-predict
```

Requirements:

* Python ≥ 3.8
* Dependencies:
`numpy`, `pandas`, `scikit-learn ≥1.4`, `lightgbm`, `xgboost`, `catboost`

---

## 📌 Basic Usage
```python
iimport pandas as pd
from cv_score_predict import cv_score_predict

# Simulate data
X = pd.DataFrame({
    "num": [1, 2, 3, 4, 5, 6, 7, 8],
    "cat": ["A", "B", "A", "C", "B", "A", "C", "D"]
})
y = [0, 1, 0, 1, 1, 0, 1, 0]
X_test = pd.DataFrame({"num": [9, 10], "cat": ["B", "E"]})

# Run CV with 2 seeds → get raw prediction matrices
oof_preds_df, test_preds_df, _ = cv_score_predict(
    X=X,
    y=y,
    X_test=X_test,
    pred_type="classification",
    process_categorical=True,
    models=["lgb", "xgb"],
    random_state=[42, 123],
    n_splits=2,
    verbose=2,
)

# Analyze OOF predictions
print("OOF predictions shape:", oof_preds_df.shape)   # e.g., (8, 4) → 2 models × 2 seeds
print("Test predictions shape:", test_preds_df.shape) # e.g., (2, 8) → 2 models × 2 seeds × 2 folds
print(oof_preds_df.columns.tolist())
# ['lgb_seed_42', 'xgb_seed_42', 'lgb_seed_123', 'xgb_seed_123']

# Average across all models/seeds for a final OOF prediction
final_oof = oof_preds_df.mean(axis=1)
```

💡 Note: OOF predictions are already stitched across folds per (model, seed), so each OOF column is complete. Test predictions remain per-fold to preserve variance estimation.

---

## 🔧 Advanced Usage: Reuse Artifacts for New Data
```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from cv_score_predict import cv_score_predict

# Define a processor that returns a DataFrame
base_processor = make_column_transformer(
    (StandardScaler(), ["num"]),
    remainder="passthrough"
).set_output(transform='pandas')

scoring_dict = {
    "roc_auc": roc_auc_score,
    "accuracy": accuracy_score,
    "log_loss": log_loss,
}
params_dict = {
    "lgb": {"learning_rate": 0.1, "num_leaves": 100},
    "xgb": {"learning_rate": 0.1, "max_depth": 10},
    "cb": {"learning_rate": 0.1, "depth": 8},
}
# Run CV and return artifacts
oof_preds_df, _, trained_pipelines = cv_score_predict(
    X, y,
    X_test=None,
    pred_type="classification",
    processor=base_processor,
    process_categorical=True,
    models=["lgb", "xgb", "cb"],
    params_dict=params_dict,
    scoring_dict=scoring_dict,
    random_state=[42, 123],
    n_splits=5,
    return_trained=True,
)
# Create new data
X_new = pd.DataFrame({"num": [7, 8], "cat": [None, "A"]})

# Transform and predict using each trained pipeline
all_new_preds = []
for fold_processor, model in trained_pipelines:
    X_new_proc = fold_processor.transform(X_new)
    pred = model.predict_proba(X_new_proc)[:, 1]
    all_new_preds.append(pred)

# Ensemble by averaging
final_new_pred = np.mean(all_new_preds, axis=0)
```
This gives you a leakage-free stacking pipeline with proper early stopping and categorical handling.

---

## 📝 Notes
* Column naming: 
    - OOF: `{model}_seed_{seed}`
    - Test: `{model}_seed_{seed}_fold_{fold}`
* Always use `.set_output(transform="pandas")` in sklearn pipelines to preserve column names and dtypes.
* Categorical detection happens after your base processor runs—so even if your pipeline creates or modifies categorical columns, they’ll be handled correctly when `process_categorical=True`.

---

## 📄 License
This project is licensed under the MIT License.
See the LICENSE file for details.