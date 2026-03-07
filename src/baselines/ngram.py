"""
N-gram Baseline - N-gram binary features with Random Forest

Purpose: Extract n-gram features (bigrams, trigrams, 4-grams) from symbolized
prefixes and classify using a random forest.  Captures local sequential
patterns without full sequential pattern mining.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.baselines.base import BaseBaseline
from src.preprocessing.k_prefix import PrefixEntry

logger = logging.getLogger(__name__)


class NGramBaseline(BaseBaseline):
    """N-gram baseline using binary n-gram presence features.

    Extracts all n-grams for configurable window sizes, builds a vocabulary
    of observed n-gram tuples, and fits a random forest classifier on
    binary feature vectors.
    """

    name: str = "ngram"

    def __init__(self, ns: tuple[int, ...] = (2, 3, 4)) -> None:
        """
        Args:
            ns: Tuple of n-gram sizes to extract (e.g. ``(2, 3, 4)``).
        """
        self.ns = ns
        self.vocab_: dict[tuple[str, ...], int] | None = None
        self.model_: RandomForestClassifier | None = None

    def fit(self, entries: list[PrefixEntry]) -> None:
        """Build n-gram vocabulary and fit random forest.

        Args:
            entries: Training prefix entries with symbols and outcomes.
        """
        labels = self._labels_from_entries(entries)

        all_ngrams_per_entry = [self._extract_ngrams(e.symbols) for e in entries]

        unique_ngrams = sorted({ng for ngs in all_ngrams_per_entry for ng in ngs})
        self.vocab_ = {ng: idx for idx, ng in enumerate(unique_ngrams)}
        logger.info(
            "NGram vocabulary built: %d unique n-grams (ns=%s)",
            len(self.vocab_),
            self.ns,
        )

        X = np.array(
            [self._vectorize_from_ngrams(ngs) for ngs in all_ngrams_per_entry]
        )

        self.model_ = RandomForestClassifier(
            n_estimators=100, random_state=42,
        )
        self.model_.fit(X, labels)
        logger.info("NGram random forest fitted on %d samples", len(labels))

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

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_ngrams(self, symbols: list[str]) -> set[tuple[str, ...]]:
        """Extract all n-grams of configured sizes from a symbol sequence.

        Args:
            symbols: Symbolized prefix sequence.

        Returns:
            Set of n-gram tuples present in the sequence.
        """
        ngrams: set[tuple[str, ...]] = set()
        for n in self.ns:
            for i in range(len(symbols) - n + 1):
                ngrams.add(tuple(symbols[i : i + n]))
        return ngrams

    def _vectorize(self, symbols: list[str]) -> np.ndarray:
        """Convert a symbol sequence into a binary n-gram feature vector.

        Args:
            symbols: Symbolized prefix sequence.

        Returns:
            1-D binary array with length equal to n-gram vocabulary size.
        """
        return self._vectorize_from_ngrams(self._extract_ngrams(symbols))

    def _vectorize_from_ngrams(
        self, ngrams: set[tuple[str, ...]]
    ) -> np.ndarray:
        """Convert pre-extracted n-grams into a binary feature vector.

        Args:
            ngrams: Set of n-gram tuples present in a sequence.

        Returns:
            1-D binary array with length equal to n-gram vocabulary size.
        """
        assert self.vocab_ is not None
        vec = np.zeros(len(self.vocab_), dtype=np.float64)
        for ng in ngrams:
            idx = self.vocab_.get(ng)
            if idx is not None:
                vec[idx] = 1.0
        return vec
