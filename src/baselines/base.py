"""
Base Baseline - Abstract interface for all baseline failure detectors

Purpose: Define the shared interface that all baselines (and the main approach)
implement so they can be evaluated uniformly in the comparison framework.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from src.data_collection.trace_schema import TaskOutcome
from src.preprocessing.k_prefix import PrefixEntry

logger = logging.getLogger(__name__)


class BaseBaseline(ABC):
    """Abstract base class for failure-detection baselines.

    All baselines accept symbolized prefix sequences and produce a failure
    probability between 0.0 and 1.0.  Subclasses must implement ``fit``,
    ``predict``, and ``predict_at_k``.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, entries: list[PrefixEntry]) -> None:
        """Train on labeled prefix entries.

        Args:
            entries: Training data with symbols and outcome labels.
        """

    @abstractmethod
    def predict(self, symbols: list[str]) -> float:
        """Return failure probability for a symbol sequence.

        Args:
            symbols: Symbolized prefix sequence.

        Returns:
            Failure probability in [0.0, 1.0].
        """

    @abstractmethod
    def predict_at_k(self, symbols: list[str], k: int) -> float:
        """Return failure probability using only the first *k* symbols.

        Args:
            symbols: Full symbolized prefix sequence.
            k: Number of leading symbols to use.

        Returns:
            Failure probability in [0.0, 1.0].
        """

    def predict_batch(self, entries: list[PrefixEntry]) -> list[float]:
        """Predict failure probability for a batch of entries.

        Default implementation calls :meth:`predict` in a loop.
        Subclasses may override for vectorized efficiency.

        Args:
            entries: List of prefix entries to score.

        Returns:
            List of failure probabilities, one per entry.
        """
        return [self.predict(e.symbols) for e in entries]

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _labels_from_entries(entries: list[PrefixEntry]) -> np.ndarray:
        """Convert entry outcomes to binary labels (failure=1, else=0).

        Args:
            entries: Prefix entries with ``outcome`` field.

        Returns:
            1-D int array of binary labels.
        """
        return np.array(
            [1 if e.outcome == TaskOutcome.FAILURE else 0 for e in entries],
            dtype=np.int32,
        )
