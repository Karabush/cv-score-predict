from typing import (
    Union, List, Dict, Tuple, Optional, Callable, Any, Literal,
)
import numpy as np 
import pandas as pd 
import lightgbm as lgb 
import xgboost as xgb 
import catboost as cb 

from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, KFold 
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class _CatWrapper(BaseEstimator, TransformerMixin):
    """
    Applies base_processor, then OrdinalEncodes any object/category columns in its output.
    """
    def __init__(self, base_processor):
        self.base_processor = base_processor

    def fit(self, X, y=None):
        X_proc = self.base_processor.fit_transform(X, y)

        # If base_processor doesn't return a DataFrame, we won't attempt to detect or encode categories.
        if not isinstance(X_proc, pd.DataFrame):
            self._returns_df = False
            self.cat_cols_ = []
            self.oe_ = None

            return self

        self._returns_df = True

        # Collect categorical/object columns (avoid select_dtypes to reduce intermediate objects)
        self.cat_cols_ = []
        for col, dtype in X_proc.dtypes.items():
            if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
                self.cat_cols_.append(col)

        if self.cat_cols_:
            self.oe_ = OrdinalEncoder(
                dtype=np.int32,
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                encoded_missing_value=-1,
                min_frequency=0.01,
            ).set_output(transform='pandas')
            self.oe_.fit(X_proc[self.cat_cols_])
        else:
            self.oe_ = None

        return self

    def transform(self, X):
        X_proc = self.base_processor.transform(X)

        # If processor returned non-DataFrame at fit time, or if no categorical columns 
        # were detected at fit time - we do not attempt to encode anything and just return 
        # the processor output unchanged.
        if not isinstance(X_proc, pd.DataFrame) or not self.cat_cols_:
            return X_proc

        # Encode categorical and convert to pandas categorical dtype
        X_proc[self.cat_cols_] = self.oe_.transform(X_proc[self.cat_cols_]).astype('category')

        return X_proc

class _IdentityProcessor(BaseEstimator, TransformerMixin):
    """Processor fallback (identity)"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.copy()

# Type aliases 
ModelKey = Literal['lgb', 'xgb', 'cb']
PredictionType = Literal['classification', 'regression']

def cv_score_predict(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    X_test: Optional[pd.DataFrame] = None,
    pred_type: PredictionType = None,  # 'classification' or 'regression' (mandatory)
    processor: Optional[Union[BaseEstimator, TransformerMixin]] = None,
    process_categorical: bool = True,
    models: Union[List[ModelKey], ModelKey] = ('lgb', 'xgb', 'cb'),
    params_dict: Optional[Dict[str, dict]] = None,
    scoring_dict: Optional[Dict[str, Callable]] = None,
    decision_threshold: float = 0.5,
    n_splits: int = 5,
    random_state: Union[int, List[int]] = 42,
    early_stopping_rounds: int = 50,
    verbose: int = 2,
    return_trained: bool = False,
    predict_proba: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[Any]], Optional[Any]]:
    """
    Cross-validate supported estimators (optionally repeated over multiple seeds),
    collect out‑of‑fold (OOF) predictions for scoring, and produce averaged
    test‑set predictions for final use. Accepts a scikit‑learn style processor
    pipeline to apply on each fold.

    Important behavior
    ------------------
    Estimators are trained with early stopping on each fold's validation set. 
    If custom model parameters are provided, number of iterations should be reasonable high
    to allow space for early stopping. Final test predictions (when `X_test` is provided) 
    are produced by the early‑stopped estimators from each fold and averaged.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series or np.ndarray
        Target values.
    X_test : pd.DataFrame or None, optional
        Final test set to predict. If None, no test predictions are produced.
    pred_type : str
        Either 'classification' or 'regression'.
    processor : object or None, optional
        Preprocessing pipeline with `fit_transform` and `transform` methods.
    process_categorical : bool, default True
        If True, object/category columns are encoded using an OrdinalEncoder
        fitted on the training DataFrame only (no leakage), then converted to
        pandas `category` dtype so libraries that auto‑detect categories work
        correctly. If False, the user is responsible for categorical handling
        (for example, inside `processor`).
    models : list or str, default ('lgb', 'xgb', 'cb')
        Model keys to train. Supported values: 'lgb', 'xgb', 'cb'.
    params_dict : dict or None, optional
        Mapping `model_name -> dict` of model parameters. 
        If None, n_estimators=10000 is used to allow space for early stopping.
    scoring_dict : dict or None, optional
        Mapping `metric_name -> callable(y_true, y_pred_or_proba)`. If None,
        defaults are provided (classification: ROC AUC; regression: RMSE).
    decision_threshold : float, default 0.5
        Threshold to convert probabilities to class labels for threshold‑based metrics.
    n_splits : int, default 5
        Number of CV folds.
    random_state : int or list of ints, default 42
        Single seed or list of seeds to repeat CV. Results are averaged across seeds.
    early_stopping_rounds : int, default 50
        Default early stopping rounds used when model params do not override it.
    verbose : int, default 2
        2 prints detailed per‑fold/per‑model scores,
        1 prints only final averaged scores,
        0 prints nothing.
    return_trained : bool, default False
        If True, return the list of trained estimator instances (one per model
        per fold per seed) and the final fitted preprocessing pipeline. 
        If False (default), trained estimators are not accumulated and the final 
        preprocessing pipeline is not fitted and None is returned in that positions.
    predict_proba : bool, default True
        For classification: if True return probabilities; if False return binary
        labels using `decision_threshold`. Ignored for regression.

    Returns
    -------
    oof_preds_total : np.ndarray
        Averaged out‑of‑fold predictions (probabilities for classification when
        `predict_proba=True`, otherwise binary labels for classification;
        predictions for regression).
    test_preds_total : np.ndarray or None
        Averaged test‑set predictions across folds and seeds, or None if
        `X_test` is None.
    trained_models_or_none : list or None
        If `return_trained=True`, the list of trained model instances (order preserved).
        Otherwise None.
    final_processor_or_none : Pipeline or None
        Final fitted preprocessing pipeline (processor + OrdinalEncoder if used), fitted on full X, 
        returned only if return_trained=True. Otherwise None.
    """
    # Input Validation
    if pred_type not in ('classification', 'regression'):
        raise ValueError("pred_type must be 'classification' or 'regression'")

    if models is None:
        raise ValueError("`models` cannot be None.")
    
    if isinstance(models, str):
        models = [models]
    allowed = {'lgb', 'xgb', 'cb'}
    for m in models:
        if m not in allowed:
            raise ValueError(f"Unsupported model '{m}'. Allowed: {allowed}")

    if isinstance(random_state, int):
        random_states = [random_state]
    else:
        random_states = list(random_state)

    if X_test is not None and len(X_test) == 0:
        raise ValueError("`X_test` must not be empty if provided.")
    
    if len(X) != len(y):
        raise ValueError("`X` and `y` must have the same number of samples.")
    
    # Ensure y as pd.Series for consistent indexing with iloc
    y = y if isinstance(y, pd.Series) else pd.Series(y)

    # Default scoring
    if scoring_dict is None:
        if pred_type == 'classification':
            scoring_dict = {'roc_auc': roc_auc_score}
        else:
            scoring_dict = {'rmse': lambda y_true, y_pred: float(np.sqrt(mean_squared_error(y_true, y_pred)))}

    # Default parameters
    if params_dict is None:
        params_dict = {m: {} for m in models}
    else:
        for m in models:
            params_dict.setdefault(m, {})

    if processor is None:
        base_processor = _IdentityProcessor()
    elif not hasattr(processor, 'fit_transform') or not hasattr(processor, 'transform'):
        raise TypeError("`processor` must have `fit_transform` and `transform` methods.")
    else:
        base_processor = processor

    # Prepare result containers
    n_samples = len(X)
    test_n = len(X_test) if X_test is not None else 0
    
    oof_preds_total = np.zeros(n_samples, dtype=np.float64)
    test_preds_total = np.zeros(test_n, dtype=np.float64) if X_test is not None else None
    trained_models: List[Any] = [] if return_trained else None

    # CV results storage
    cv_results = {
        'stacked': {name: [] for name in scoring_dict.keys()},
        'per_model': {m: {name: [] for name in scoring_dict.keys()} for m in models},
    }
    # Helper for controlled printing
    def _print(msg, level=2):
        if verbose >= level:
            print(msg)

    # Main loop across random states
    for seed in random_states:
        _print(f'\n=== Random State {seed} ===', level=2)
        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            if pred_type == 'classification'
            else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        )
        oof_preds = np.zeros(n_samples, dtype=float)
        test_preds = np.zeros(test_n, dtype=float) if X_test is not None else None

        # Per-seed storage for reporting 
        seed_model_scores = {m: {name: [] for name in scoring_dict.keys()} for m in models}
        seed_stack_scores = {metric_name: [] for metric_name in scoring_dict.keys()}
        
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            _print(f'\nFold {fold + 1}/{n_splits}', level=2)

            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Choose preprocessor based on process_categorical
            if process_categorical:
                fold_base = deepcopy(base_processor)
                fold_processor = _CatWrapper(fold_base)
            else:
                fold_processor = deepcopy(base_processor)

            # Apply processor if provided (fit on fold train only)
            X_train = fold_processor.fit_transform(X_train, y_train)
            X_val = fold_processor.transform(X_val)
            X_test_proc = fold_processor.transform(X_test) if X_test is not None else None
            
            # Get categorical columns for model params (only if using wrapper)
            cat_cols = getattr(fold_processor, 'cat_cols_', [])

            # Update model params for categorical handling
            local_params_dict = {}
            for m in models:
                p = params_dict[m].copy()
                if cat_cols and process_categorical:
                    # lgb handles categories automatically via pandas categorical dtype
                    if m == 'xgb':
                        p['enable_categorical'] = True
                    elif m == 'cb':
                        p['cat_features'] = cat_cols
                local_params_dict[m] = p

            fold_val_preds_list = []
            fold_test_preds_list = []

            for model_name in models:
                p = local_params_dict[model_name]

                # Train model
                if model_name == 'lgb':
                    ModelClass = lgb.LGBMClassifier if pred_type == 'classification' else lgb.LGBMRegressor
                    # Set a high default to allow early stopping to determine optimal rounds
                    if not p:
                        p.setdefault('n_estimators', 10000)

                    p.setdefault('verbosity', -1)
                    model = ModelClass(**p)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                    )
                elif model_name in ['xgb', 'cb']:
                    p.setdefault('early_stopping_rounds', early_stopping_rounds)
                   
                    if model_name == 'xgb': 
                        ModelClass = xgb.XGBClassifier if pred_type == 'classification' else xgb.XGBRegressor
                        if not p:
                            p.setdefault('n_estimators', 10000) 
                    else:
                        ModelClass = cb.CatBoostClassifier if pred_type == 'classification' else cb.CatBoostRegressor
                        if not p:
                            p.setdefault('iterations', 10000)

                    model = ModelClass(**p) 
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                # Append trained model instance only if requested
                if return_trained:
                    trained_models.append(model)

                # Predictions              
                if pred_type == 'classification':
                    # Prefer predict_proba; if user requested binary output at top-level,
                    # we still compute probabilities here and convert later if needed
                    val_preds = model.predict_proba(X_val)[:, 1]
                    test_fold_preds = model.predict_proba(X_test_proc)[:, 1] if X_test_proc is not None else None
                    
                else:  # regression
                    val_preds = model.predict(X_val)
                    test_fold_preds = model.predict(X_test_proc) if X_test_proc is not None else None

                fold_val_preds_list.append(val_preds)

                if X_test is not None:
                    fold_test_preds_list.append(test_fold_preds)

                # Score individual model on this fold
                if pred_type == 'classification':
                    val_binary = (val_preds >= decision_threshold).astype(int)

                    for metric_name, scoring_fn in scoring_dict.items():
                        name_l = metric_name.lower()

                        if any(k in name_l for k in ('roc', 'auc', 'logloss', 'log_loss')):
                            score = scoring_fn(y_val, val_preds)
                        else:
                            score = scoring_fn(y_val, val_binary)

                        cv_results['per_model'][model_name][metric_name].append(score)
                        seed_model_scores[model_name][metric_name].append(score)
                        _print(f'  {model_name.upper()} {metric_name}: {score:.5f}', level=2)
                else:
                    for metric_name, scoring_fn in scoring_dict.items():
                        score = scoring_fn(y_val, val_preds)
                        cv_results['per_model'][model_name][metric_name].append(score)
                        seed_model_scores[model_name][metric_name].append(score)
                        _print(f'  {model_name.upper()} {metric_name}: {score:.5f}', level=2)

            # Aggregate fold predictions (average across models)
            fold_val_preds = np.mean(np.vstack(fold_val_preds_list), axis=0)
            fold_test_preds = np.mean(np.vstack(fold_test_preds_list), axis=0) if X_test is not None else None

            # Fill OOF and accumulate test preds
            oof_preds[val_idx] = fold_val_preds
            
            if X_test is not None:
                test_preds += fold_test_preds / n_splits

            # Score stacked prediction
            if pred_type == 'classification':
                fold_val_binary = (fold_val_preds >= decision_threshold).astype(int)
                for metric_name, scoring_fn in scoring_dict.items():
                    name_l = metric_name.lower()

                    if any(k in name_l for k in ('roc', 'auc', 'logloss', 'log_loss')):
                        stacked_score = scoring_fn(y_val, fold_val_preds)
                    else:
                        stacked_score = scoring_fn(y_val, fold_val_binary)

                    cv_results['stacked'][metric_name].append(stacked_score)
                    seed_stack_scores[metric_name].append(stacked_score)
                    _print(f'  Stacked {metric_name}: {stacked_score:.5f}', level=2)
            else:
                for metric_name, scoring_fn in scoring_dict.items():
                    stacked_score = scoring_fn(y_val, fold_val_preds)
                    cv_results['stacked'][metric_name].append(stacked_score)
                    seed_stack_scores[metric_name].append(stacked_score)
                    _print(f'  Stacked {metric_name}: {stacked_score:.5f}', level=2)

        # --- End of folds for this seed ---
        
        # Accumulate across seeds
        oof_preds_total += oof_preds / len(random_states)

        if X_test is not None:
            test_preds_total += test_preds / len(random_states)
        
        if verbose >= 2:
            # Print per-model means (seed)
            _print(f'\nSeed {seed} mean scores:', level=2)
            for model_name in models:
                for metric_name, vals in seed_model_scores[model_name].items():
                    mean_val = float(np.mean(vals)) if vals else float('nan')
                    _print(f'  {model_name.upper()} {metric_name}: {mean_val:.5f}', level=2)

            # Print stacked mean scores (seed)
            for metric_name, score in {k: float(np.mean(v)) for k, v in seed_stack_scores.items()}.items():
                _print(f'  Stacked {metric_name}: {score:.5f}', level=2)

    # Final summary print
    if verbose >= 1:
        print('\n' + '=' * 30)
        print('=== CV Results Summary ===\n')
        print('Mean CV Scores per Model:')
        for model_name in models:
            print(f'\n--- {model_name.upper()} ---')
            for metric_name, scores in cv_results['per_model'][model_name].items():
                print(f'  {metric_name}: {np.mean(scores):.5f}')
  
        print('\nMean Stacked CV Scores:')
        for metric_name, scores in cv_results['stacked'].items():
            print(f'  {metric_name}: {np.mean(scores):.5f}')
        
    # Post-process outputs for classification when predict_proba flag is False
    if pred_type == 'classification' and not predict_proba:
        # Convert averaged probabilities to binary labels using decision_threshold
        oof_preds_total = (oof_preds_total >= decision_threshold).astype(int)
        if test_preds_total is not None:
            test_preds_total = (test_preds_total >= decision_threshold).astype(int)
    
    # Final preprocessing pipeline fitted on full data
    final_processor = None
    if return_trained:
        if process_categorical:
            final_processor = _CatWrapper(base_processor)
            final_processor.fit(X, y)
        else:
            final_processor = base_processor
            final_processor.fit(X, y)

    # Prepare return values
    test_preds_return = test_preds_total if X_test is not None else None
    trained_models = trained_models if return_trained else None
    final_processor = final_processor if return_trained else None

    return oof_preds_total, test_preds_return, trained_models, final_processor