"""Evaluation metrics, threshold rule and country-clustered bootstrap (shared by CV and test code)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

SEED = 42
N_BOOT = 2000


def safe_auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def safe_ap(y, p) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins (weighted by bin size)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        rows.append({"bin": b, "p_low": b / n_bins, "p_high": (b + 1) / n_bins, "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()) if m.any() else np.nan,
                     "observed_rate": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def score_metrics(y, p) -> dict:
    """Threshold-free metrics."""
    y, p = np.asarray(y), np.asarray(p, float)
    return {"pr_auc": safe_ap(y, p), "roc_auc": safe_auc(y, p), "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
            "ece": ece(y, np.clip(p, 0, 1))}


def threshold_metrics(y, p, thr: float) -> dict:
    """Metrics for hard predictions p >= thr."""
    y = np.asarray(y)
    yhat = (np.asarray(p, float) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "accuracy": float((tp + tn) / len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)) if len(np.unique(y)) > 1 else float("nan"),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yhat)) if yhat.std() > 0 and y.std() > 0 else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def all_metrics(y, p, thr: float) -> dict:
    return {**score_metrics(y, p), **threshold_metrics(y, p, thr)}


def best_f1_threshold(y, p) -> float:
    """Threshold maximising F1 on the given (pooled out-of-fold) predictions. Ties -> the higher threshold."""
    y, p = np.asarray(y), np.asarray(p, float)
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    best = np.flatnonzero(f1 >= f1.max() - 1e-12)[-1]
    return float(thr[best])


# --------------------------------------------------------------------------- clustered bootstrap
class ClusterBootstrap:
    """Resamples whole countries with replacement (rows of a country are strongly dependent)."""

    def __init__(self, groups, n_boot: int = N_BOOT, seed: int = SEED):
        groups = np.asarray(groups)
        self.codes, uniq = pd.factorize(groups)
        self.n_groups = len(uniq)
        order = np.argsort(self.codes, kind="stable")
        self.rows_by_group = np.split(order, np.cumsum(np.bincount(self.codes))[:-1])
        self.n_boot, self.seed = n_boot, seed

    def draws(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_boot):
            pick = rng.integers(0, self.n_groups, self.n_groups)
            yield np.concatenate([self.rows_by_group[g] for g in pick])

    def ci(self, fn, alpha: float = 0.05) -> tuple[float, float, np.ndarray]:
        """Percentile CI of fn(row_index_array) -> float or dict. Returns (lo, hi, draws) for scalar fn."""
        vals = np.array([fn(ix) for ix in self.draws()], dtype=float)
        lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi), vals

    def ci_dict(self, fn, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        """fn returns dict of metric -> value; returns metric -> (lo, hi)."""
        recs = [fn(ix) for ix in self.draws()]
        df = pd.DataFrame(recs).astype(float)
        out = {}
        for c in df.columns:
            v = df[c].to_numpy()
            if np.isnan(v).all():
                out[c] = (float("nan"), float("nan"))
            else:
                out[c] = tuple(float(x) for x in np.nanpercentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        return out
