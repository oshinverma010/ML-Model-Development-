"""Pipelines, baselines and hyperparameter grids (scikit-learn only, random_state = 42 everywhere).

Every data-dependent step (correlation filter, winsorising limits, imputer, scaler, model) lives inside a
sklearn Pipeline, so it is fitted on the training fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required to import IterativeImputer)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import validate_data

SEED = 42
MODELS = ["lr", "rf", "hgb"]
MODEL_LABELS = {"lr": "Elastic-net logistic regression", "rf": "Random forest", "hgb": "Gradient boosting (HGB)"}
COMPLEXITY = {"lr": 0, "rf": 1, "hgb": 2}                       # simpler first (plan section 7)
IMPUTERS = {"lr": ["median", "ffill_median", "iterative"],
            "rf": ["median", "ffill_median", "iterative"],
            "hgb": ["native", "median", "ffill_median", "iterative"]}
IMPUTER_COMPLEXITY = {"native": 0, "median": 1, "ffill_median": 2, "iterative": 3}
WEIGHT_COMPLEXITY = {None: 0, "balanced": 1}


# --------------------------------------------------------------------------- custom transformers
class CorrelationFilter(TransformerMixin, BaseEstimator):
    """Greedy near-duplicate filter. Features are visited in column order; a feature is dropped when its
    |Spearman rho| with an already kept feature exceeds `threshold`. Fitted on the training fold only."""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=True)
        rho = pd.DataFrame(X).corr(method="spearman", min_periods=20).abs().fillna(0.0).to_numpy()
        keep: list[int] = []
        for j in range(X.shape[1]):
            if all(rho[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_ = np.array(keep)
        self.dropped_ = np.array([j for j in range(X.shape[1]) if j not in set(keep)])
        return self

    def transform(self, X):
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return X[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)
        return names[self.keep_]


class Winsorizer(TransformerMixin, BaseEstimator):
    """Clip each column to its [lower, upper] percentiles learned on the training fold (NaN-safe)."""

    def __init__(self, lower: float = 1.0, upper: float = 99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=True)
        lo = np.full(X.shape[1], -np.inf)
        hi = np.full(X.shape[1], np.inf)
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[~np.isnan(col)]
            if len(col):
                lo[j], hi[j] = np.percentile(col, [self.lower, self.upper])
        self.lo_, self.hi_ = lo, hi
        return self

    def transform(self, X):
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return np.clip(X, self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)


# --------------------------------------------------------------------------- pipeline factory
def _estimator(model: str, params: dict, class_weight):
    if model == "lr":
        p = {"C": 1.0, "l1_ratio": 0.5, **params}
        return LogisticRegression(solver="saga", max_iter=5000, random_state=SEED, class_weight=class_weight, **p)
    if model == "rf":
        p = {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 10, "max_features": "sqrt", **params}
        return RandomForestClassifier(random_state=SEED, n_jobs=1, class_weight=class_weight, **p)
    if model == "hgb":
        p = {"learning_rate": 0.05, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 50, **params}
        # early_stopping=False: the internal validation split would be random, ignoring time order
        return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, class_weight=class_weight, **p)
    raise ValueError(model)


def make_pipeline(model: str, imputer: str = "median", class_weight=None, params: dict | None = None,
                  corr_filter: bool = True, l1_select: bool = False) -> Pipeline:
    """Pipeline: correlation filter -> winsorise -> impute -> (scale for LR) -> [L1 select] -> classifier.

    `imputer` in {"median", "ffill_median", "iterative", "native"}; "ffill_median" differs from "median" only in the
    input data (the dataset is built with ffill=True), the pipeline is identical. "native" is allowed for HGB only.
    """
    params = params or {}
    steps: list[tuple[str, object]] = []
    if corr_filter:
        steps.append(("corr_filter", CorrelationFilter(0.9)))
    steps.append(("winsorise", Winsorizer(1.0, 99.0)))
    if imputer in ("median", "ffill_median"):
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True)))
    elif imputer == "iterative":
        steps.append(("impute", IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=SEED,
                                                 sample_posterior=False, add_indicator=True)))
    elif imputer == "native":
        if model != "hgb":
            raise ValueError("native NaN handling is only available for HGB")
    else:
        raise ValueError(imputer)
    if model == "lr":
        steps.append(("scale", StandardScaler()))
    if l1_select:
        steps.append(("l1_select", SelectFromModel(
            LogisticRegression(solver="saga", l1_ratio=1.0, C=0.1, max_iter=5000, random_state=SEED))))
    steps.append(("clf", _estimator(model, params, class_weight)))
    return Pipeline(steps)


def pipeline_feature_names(pipe: Pipeline, feature_cols: list[str]) -> list[str]:
    """Names of the columns reaching the classifier (after filter, indicators, selection)."""
    return list(pipe[:-1].get_feature_names_out(np.asarray(feature_cols, dtype=object)))


def dataset_variant(imputer: str) -> bool:
    """True if this imputation option needs the forward-filled dataset."""
    return imputer == "ffill_median"


# --------------------------------------------------------------------------- search spaces (plan section 6)
GRIDS = {
    "lr": {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
           "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
    "rf": {"clf__n_estimators": [300, 500], "clf__max_depth": [3, 5, 8, 12, None],
           "clf__min_samples_leaf": [5, 10, 20, 40], "clf__max_features": ["sqrt", 0.3, 0.5]},
    "hgb": {"clf__learning_rate": [0.01, 0.03, 0.05, 0.1], "clf__max_depth": [2, 3, 4, 6],
            "clf__max_leaf_nodes": [8, 15, 31], "clf__min_samples_leaf": [20, 50, 100],
            "clf__l2_regularization": [0, 0.1, 1, 10], "clf__max_iter": [100, 200, 400]},
}
N_ITER = 30


def params_from_search(best: dict) -> dict:
    """Strip the 'clf__' prefix of RandomizedSearchCV parameters."""
    return {k.split("__", 1)[1]: v for k, v in best.items()}


# --------------------------------------------------------------------------- baselines
def persistence_pred(X: pd.DataFrame) -> np.ndarray:
    """Persistence rule: 1 if CO2pc fell and GDPpc rose over the previous 3 years; 0 if either is missing."""
    ok = X["co2_chg3"].notna() & X["gdp_cagr3"].notna()
    return ((X["co2_chg3"] < 0) & (X["gdp_cagr3"] > 0) & ok).astype(float).to_numpy()


def prior_pred(y_train, n: int) -> np.ndarray:
    return np.full(n, float(np.mean(y_train)))
