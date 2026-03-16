"""
Step Count Baseline - Prefix length as a naive failure signal

Purpose: Sanity-check baseline that scores traces purely by how many
steps they contain relative to a maximum.  If this matches BIDE's
performance, the mined patterns are not adding value beyond length.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import logging

from src.baselines.base import BaseBaseline
from src.preprocessing.k_prefix import PrefixEntry

logger = logging.getLogger(__name__)


class StepCountBaseline(BaseBaseline):
    """Baseline that predicts failure probability as prefix length / max_steps."""

    name: str = "step_count"

    def __init__(self, max_steps: int = 10) -> None:
        self._max_steps = max_steps

    def fit(self, entries: list[PrefixEntry]) -> None:
        """No-op; step count requires no training.

        Args:
            entries: Training prefix entries (unused).
        """
        logger.info("StepCountBaseline.fit() called (no-op), max_steps=%d", self._max_steps)

    def predict(self, symbols: list[str]) -> float:
        """Return failure probability based on sequence length.

        Args:
            symbols: Symbolized prefix sequence.

        Returns:
            min(len(symbols) / max_steps, 1.0).
        """
        return min(len(symbols) / self._max_steps, 1.0)

    def predict_at_k(self, symbols: list[str], k: int) -> float:
        """Return failure probability using prefix length k.

        Args:
            symbols: Full symbolized prefix sequence (unused).
            k: Number of leading symbols to consider.

        Returns:
            min(k / max_steps, 1.0).
        """
        return min(k / self._max_steps, 1.0)
