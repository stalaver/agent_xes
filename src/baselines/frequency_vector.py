"""
Frequency Vector Baseline - Symbol count vectors with Logistic Regression

Purpose: Count occurrences of each unique symbol in a k-prefix and classify
using logistic regression.  Serves as a bag-of-symbols baseline that ignores
sequential ordering.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.baselines.base import BaseBaseline
from src.preprocessing.k_prefix import PrefixEntry

logger = logging.getLogger(__name__)


class FrequencyVectorBaseline(BaseBaseline):
    """Bag-of-symbols baseline using symbol frequency counts.

    Builds a vocabulary from training data, converts each prefix into a
    count vector, and fits a logistic regression classifier.
    """

    name: str = "frequency_vector"

    def __init__(self) -> None:
        self.vocab_: dict[str, int] | None = None
        self.model_: LogisticRegression | None = None

    def fit(self, entries: list[PrefixEntry]) -> None:
        """Build vocabulary and fit logistic regression.

        Args:
            entries: Training prefix entries with symbols and outcomes.
        """
        symbols_corpus = [e.symbols for e in entries]
        labels = self._labels_from_entries(entries)

        unique_symbols = sorted({s for seq in symbols_corpus for s in seq})
        self.vocab_ = {sym: idx for idx, sym in enumerate(unique_symbols)}
        logger.info(
            "FrequencyVector vocabulary built: %d unique symbols",
            len(self.vocab_),
        )

        X = np.array([self._vectorize(seq) for seq in symbols_corpus])

        self.model_ = LogisticRegression(max_iter=1000, solver="lbfgs")
        self.model_.fit(X, labels)
        logger.info("FrequencyVector logistic regression fitted on %d samples", len(labels))

    def predict(self, symbols: list[str]) -> float:
        """Return failure probability for a symbol sequence.

        Args:
            symbols: Symbolized prefix sequence.

        Returns:
            Failure probability in [0.0, 1.0].

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.model_ is None or self.vocab_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        vec = self._vectorize(symbols).reshape(1, -1)
        return float(self.model_.predict_proba(vec)[0, 1])

    def predict_at_k(self, symbols: list[str], k: int) -> float:
        """Return failure probability using only the first *k* symbols.

        Args:
            symbols: Full symbolized prefix sequence.
            k: Number of leading symbols to use.

        Returns:
            Failure probability in [0.0, 1.0].
        """
        return self.predict(symbols[:k])

    def _vectorize(self, symbols: list[str]) -> np.ndarray:
        """Convert a symbol sequence into a count vector.

        Unknown symbols (not seen during fit) are silently ignored.

        Args:
            symbols: Symbolized prefix sequence.

        Returns:
            1-D array of symbol counts with length equal to vocabulary size.
        """
        assert self.vocab_ is not None
        vec = np.zeros(len(self.vocab_), dtype=np.float64)
        for sym in symbols:
            idx = self.vocab_.get(sym)
            if idx is not None:
                vec[idx] += 1.0
        return vec
