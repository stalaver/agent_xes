"""
Experiment Runner - Orchestrate full baseline comparison experiments

Purpose: Split data, fit baselines, compute per-k and cross-site metrics,
and produce summary tables in human-readable and LaTeX formats.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from src.baselines.base import BaseBaseline
from src.evaluation.data_split import DataSplitter
from src.evaluation.metrics import MetricsCalculator
from src.preprocessing.k_prefix import PrefixDataset, PrefixEntry

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResults:
    """Container for a full comparison experiment's results.

    Attributes:
        per_baseline: Mapping from baseline name to its result dict.
            Each value has keys ``at_k`` (dict[int, metrics]) and
            ``cross_site`` (metrics dict or None).
        split_info: Counts of train/val/test/holdout entries.
        k_values: Prefix lengths that were evaluated.
    """

    per_baseline: dict[str, dict] = field(default_factory=dict)
    split_info: dict[str, int] = field(default_factory=dict)
    k_values: list[int] = field(default_factory=list)


class ExperimentRunner:
    """Orchestrates full baseline comparison."""

    def __init__(
        self,
        baselines: list[BaseBaseline],
        dataset: PrefixDataset,
        holdout_sites: list[str] | None = None,
        k_values: list[int] | None = None,
        seed: int = 42,
    ) -> None:
        """Configure an experiment.

        Args:
            baselines: Baseline instances to compare.
            dataset: Prefix dataset containing all entries.
            holdout_sites: Websites reserved for cross-site evaluation.
            k_values: Prefix lengths to evaluate. Defaults to [3, 5, 8, 10].
            seed: Random seed for data splitting.
        """
        self._baselines = baselines
        self._dataset = dataset
        self._holdout_sites = holdout_sites
        self._k_values = k_values if k_values is not None else [3, 5, 8, 10]
        self._seed = seed
        self._splitter = DataSplitter()
        self._metrics = MetricsCalculator()

    def run(self) -> ExperimentResults:
        """Execute the full experiment pipeline.

        1. Split data into train/val/test + holdout.
        2. Fit each baseline on the training set.
        3. Compute at-k metrics on the test set.
        4. Compute cross-site metrics on the holdout set.

        Returns:
            ExperimentResults with per-baseline, per-k results.
        """
        split = self._splitter.split(
            entries=self._dataset.entries,
            holdout_sites=self._holdout_sites,
            seed=self._seed,
        )

        results = ExperimentResults(
            split_info=split.summary(),
            k_values=list(self._k_values),
        )

        train_labels = BaseBaseline._labels_from_entries(split.train)
        n_classes = len(np.unique(train_labels))

        for baseline in self._baselines:
            logger.info("Evaluating baseline: %s", baseline.name)

            if n_classes < 2:
                logger.warning(
                    "Training split has only %d class(es) — skipping %s "
                    "(results will be NaN)",
                    n_classes,
                    baseline.name,
                )
                nan_at_k = {k: self._metrics._nan_metrics() for k in self._k_values}
                results.per_baseline[baseline.name] = {
                    "at_k": nan_at_k,
                    "cross_site": self._metrics._nan_metrics() if split.holdout else None,
                }
                continue

            baseline.fit(split.train)

            at_k = self._metrics.compute_at_k(
                baseline, split.test, self._k_values
            )

            cross_site: dict | None = None
            if split.holdout:
                max_k = max(self._k_values)
                in_dist_auc_pr = at_k.get(max_k, {}).get("auc_pr", float("nan"))
                cross_site = self._metrics.compute_cross_site(
                    baseline, split.train, split.holdout, in_dist_auc_pr
                )

            results.per_baseline[baseline.name] = {
                "at_k": at_k,
                "cross_site": cross_site,
            }

        return results

    def summary_table(self, results: ExperimentResults) -> str:
        """Produce human-readable and LaTeX summary tables.

        Args:
            results: Output from :meth:`run`.

        Returns:
            String containing an ASCII table followed by a LaTeX tabular block.
        """
        lines: list[str] = []
        lines.append(self._ascii_table(results))
        lines.append("")
        lines.append(self._latex_table(results))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Table formatters
    # ------------------------------------------------------------------

    def _ascii_table(self, results: ExperimentResults) -> str:
        """Build a human-readable comparison table.

        Rows are baselines; columns are F1@k values plus cross-site AUC-PR
        and AUC delta.
        """
        k_cols = [f"F1@{k}" for k in results.k_values]
        header_parts = ["Baseline".ljust(20)] + [c.rjust(8) for c in k_cols]

        has_cross = any(
            r.get("cross_site") is not None
            for r in results.per_baseline.values()
        )
        if has_cross:
            header_parts += ["AUC-PR-X".rjust(10), "AUC-Δ".rjust(8)]

        header = " | ".join(header_parts)
        sep = "-" * len(header)

        rows: list[str] = [header, sep]

        for name, data in results.per_baseline.items():
            parts = [name.ljust(20)]
            for k in results.k_values:
                f1 = data["at_k"].get(k, {}).get("f1", float("nan"))
                parts.append(self._fmt(f1).rjust(8))
            if has_cross:
                cs = data.get("cross_site") or {}
                parts.append(self._fmt(cs.get("auc_pr", float("nan"))).rjust(10))
                parts.append(self._fmt(cs.get("auc_delta", float("nan"))).rjust(8))
            rows.append(" | ".join(parts))

        return "\n".join(rows)

    def _latex_table(self, results: ExperimentResults) -> str:
        """Build a LaTeX tabular block."""
        k_cols = [f"F1@{k}" for k in results.k_values]
        has_cross = any(
            r.get("cross_site") is not None
            for r in results.per_baseline.values()
        )

        n_cols = 1 + len(k_cols) + (2 if has_cross else 0)
        col_spec = "l" + "r" * (n_cols - 1)

        lines: list[str] = [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
        ]

        header_cells = ["Baseline"] + k_cols
        if has_cross:
            header_cells += ["AUC-PR (X-site)", r"$\Delta$ AUC"]
        lines.append(" & ".join(header_cells) + r" \\")
        lines.append(r"\midrule")

        for name, data in results.per_baseline.items():
            cells = [name.replace("_", r"\_")]
            for k in results.k_values:
                f1 = data["at_k"].get(k, {}).get("f1", float("nan"))
                cells.append(self._fmt(f1))
            if has_cross:
                cs = data.get("cross_site") or {}
                cells.append(self._fmt(cs.get("auc_pr", float("nan"))))
                cells.append(self._fmt(cs.get("auc_delta", float("nan"))))
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Baseline comparison: F1 at varying prefix lengths and cross-site generalization.}")
        lines.append(r"\label{tab:baseline_comparison}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    @staticmethod
    def _fmt(value: float, decimals: int = 3) -> str:
        """Format a float for display, showing 'NaN' for missing values."""
        if math.isnan(value):
            return "NaN"
        return f"{value:.{decimals}f}"
