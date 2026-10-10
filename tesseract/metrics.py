"""Ranking scores for the fix-risk predictor. Numpy only, so the viewer can backtest without torch."""

from __future__ import annotations

import numpy as np


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based ranks; tied scores share the mean of their ranks."""
    order = np.argsort(scores, kind="mergesort")
    ordinal = np.empty(scores.size, dtype=np.float64)
    ordinal[order] = np.arange(1, scores.size + 1)
    _values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    totals = np.bincount(inverse, weights=ordinal)
    return totals[inverse] / counts[inverse]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve. NaN unless both classes are present."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    hits = labels[np.argsort(-scores, kind="mergesort")]
    precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float(precision[hits].sum() / n_pos)


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    if labels.size == 0:
        return float("nan")
    k = max(1, min(int(k), scores.size))
    return float(labels[np.argsort(-scores, kind="mergesort")[:k]].mean())


def risk_scores(scores: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    valid = np.asarray(valid, dtype=bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()[valid]
    labels = np.asarray(labels, dtype=bool).ravel()[valid]
    return {
        "auroc": auroc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "precision_at_10pct": precision_at_k(scores, labels, round(0.1 * labels.size)),
        "base_rate": float(labels.mean()) if labels.size else float("nan"),
        "tokens": int(labels.size),
    }
