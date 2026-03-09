"""
Metrics Calculator - Evaluation metrics for failure detection baselines

Purpose: Compute binary classification metrics, early-detection metrics at
varying prefix lengths, and cross-site generalization metrics.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import logging
import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.baselines.base import BaseBaseline
from src.preprocessing.k_prefix import PrefixEntry

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5


def find_optimal_threshold(
    y_true: list[int],
    y_scores: list[float],
    num_thresholds: int = 200,
) -> float:
    """Sweep thresholds on a validation set and return the one maximizing macro-F1.

    Uses ``average='macro'`` (mean of per-class F1) so that the trivial
    all-failure predictor — which achieves perfect failure-F1 but zero
    success-F1 — is penalized under heavy class imbalance.

    Args:
        y_true: Ground-truth binary labels (1=failure, 0=success).
        y_scores: Predicted failure scores in [0, 1].
        num_thresholds: Number of evenly-spaced candidates to evaluate.

    Returns:
        Threshold in [0, 1] that maximizes macro-F1.  Falls back to 0.5
        when inputs are degenerate (empty, single-class, or all-zero F1).
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_scores_arr = np.asarray(y_scores, dtype=np.float64)

    if len(y_true_arr) == 0 or len(np.unique(y_true_arr)) < 2:
        return DEFAULT_THRESHOLD

    best_f1 = -1.0
    best_t = DEFAULT_THRESHOLD
    for t in np.linspace(0.0, 1.0, num_thresholds):
        y_pred = (y_scores_arr >= t).astype(np.int32)
        f1 = float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    if best_f1 == 0.0:
        return DEFAULT_THRESHOLD

    return best_t


class MetricsCalculator:
    """Compute all thesis evaluation metrics."""

    def compute_binary_metrics(
        self,
        y_true: list[int],
        y_scores: list[float],
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Compute standard binary classification metrics.

        Args:
            y_true: Ground-truth labels (1=failure, 0=success).
            y_scores: Predicted failure probabilities.
            threshold: Decision boundary for converting scores to labels.

        Returns:
            Dict with precision, recall, f1, accuracy, auc_roc, auc_pr.
            AUC values are NaN when only one class is present.
        """
        y_true_arr = np.asarray(y_true, dtype=np.int32)
        y_scores_arr = np.asarray(y_scores, dtype=np.float64)
        y_pred = (y_scores_arr >= threshold).astype(np.int32)

        metrics: dict[str, float] = {
            "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
            "f1_success": float(
                f1_score(y_true_arr, y_pred, pos_label=0, zero_division=0)
            ),
            "accuracy": float(accuracy_score(y_true_arr, y_pred)),
            "auc_roc": self._safe_roc_auc(y_true_arr, y_scores_arr),
            "auc_pr": self._safe_avg_precision(y_true_arr, y_scores_arr),
        }
        return metrics

    def compute_at_k(
        self,
        baseline: BaseBaseline,
        entries: list[PrefixEntry],
        k_values: list[int] | None = None,
        thresholds: dict[int, float] | None = None,
    ) -> dict[int, dict[str, float]]:
        """Compute metrics at each prefix length k.

        Args:
            baseline: A fitted baseline that supports predict_at_k.
            entries: Test entries with symbols and outcome labels.
            k_values: Prefix lengths to evaluate. Defaults to [3, 5, 8, 10].
            thresholds: Per-k decision thresholds (e.g. tuned on a val set).
                Missing k values fall back to ``DEFAULT_THRESHOLD``.

        Returns:
            Mapping from k to metrics dict.
        """
        if k_values is None:
            k_values = [3, 5, 8, 10]
        if thresholds is None:
            thresholds = {}

        y_true = BaseBaseline._labels_from_entries(entries).tolist()
        results: dict[int, dict[str, float]] = {}

        for k in k_values:
            scores = [baseline.predict_at_k(e.symbols, k) for e in entries]
            t = thresholds.get(k, DEFAULT_THRESHOLD)
            results[k] = self.compute_binary_metrics(y_true, scores, threshold=t)
            logger.debug(
                "Baseline %s @ k=%d (thr=%.3f) — F1=%.3f, AUC-PR=%.3f",
                baseline.name,
                k,
                t,
                results[k]["f1"],
                results[k]["auc_pr"],
            )

        return results

    def compute_cross_site(
        self,
        baseline: BaseBaseline,
        train_entries: list[PrefixEntry],
        holdout_entries: list[PrefixEntry],
        in_dist_auc_pr: float = float("nan"),
        threshold: float = DEFAULT_THRESHOLD,
    ) -> dict[str, float]:
        """Evaluate cross-site generalization on holdout websites.

        Instantiates a fresh baseline of the same type, fits on
        train_entries, and predicts on holdout_entries.

        Args:
            baseline: Baseline instance (used only to determine the class).
            train_entries: Training data for a fresh fit.
            holdout_entries: Entries from held-out websites.
            in_dist_auc_pr: In-distribution AUC-PR from test-set evaluation,
                used to compute the generalization delta.
            threshold: Decision boundary for binary predictions (tuned on val).

        Returns:
            Holdout metrics dict with an additional ``auc_delta`` key
            (in_dist_auc_pr minus holdout AUC-PR).
        """
        if not holdout_entries:
            logger.warning("No holdout entries — returning NaN metrics")
            return self._nan_metrics()

        fresh: BaseBaseline = type(baseline)()
        fresh.fit(train_entries)

        y_true = BaseBaseline._labels_from_entries(holdout_entries).tolist()
        scores = [fresh.predict(e.symbols) for e in holdout_entries]
        metrics = self.compute_binary_metrics(y_true, scores, threshold=threshold)

        holdout_auc_pr = metrics["auc_pr"]
        metrics["auc_delta"] = (
            float("nan")
            if math.isnan(in_dist_auc_pr) or math.isnan(holdout_auc_pr)
            else in_dist_auc_pr - holdout_auc_pr
        )

        logger.info(
            "Cross-site %s — AUC-PR=%.3f, delta=%.3f",
            baseline.name,
            holdout_auc_pr,
            metrics["auc_delta"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """ROC-AUC that returns NaN when only one class is present."""
        try:
            return float(roc_auc_score(y_true, y_scores))
        except ValueError:
            return float("nan")

    @staticmethod
    def _safe_avg_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Average precision that returns NaN when only one class is present."""
        try:
            return float(average_precision_score(y_true, y_scores))
        except ValueError:
            return float("nan")

    @staticmethod
    def _nan_metrics() -> dict[str, float]:
        """Return a metrics dict filled with NaN values."""
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "f1_success": float("nan"),
            "accuracy": float("nan"),
            "auc_roc": float("nan"),
            "auc_pr": float("nan"),
            "auc_delta": float("nan"),
        }
