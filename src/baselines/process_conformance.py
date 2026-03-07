"""
Process Conformance Baseline - Transition-model deviation scoring

Purpose: Discover a "normal" process model from success-only traces by
building a first-order transition probability matrix, then measure how much
an unseen sequence deviates from it.  Serves as a lightweight stand-in for
Alpha Miner conformance checking without a PM4Py dependency.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.baselines.base import BaseBaseline
from src.data_collection.trace_schema import TaskOutcome
from src.preprocessing.k_prefix import PrefixEntry

logger = logging.getLogger(__name__)


class ProcessConformanceBaseline(BaseBaseline):
    """Process conformance baseline using a first-order transition model.

    Learns transition probabilities from success traces and scores unseen
    sequences by average deviation: failure_score = 1 - mean(P(b|a)) over
    consecutive symbol pairs.
    """

    name: str = "process_conformance"

    def __init__(self) -> None:
        self.transitions_: dict[str, dict[str, float]] | None = None

    # ------------------------------------------------------------------
    # BaseBaseline interface
    # ------------------------------------------------------------------

    def fit(self, entries: list[PrefixEntry]) -> None:
        """Build transition probability matrix from success-only sequences.

        Args:
            entries: Training prefix entries with symbols and outcomes.
        """
        success_seqs = [
            e.symbols for e in entries
            if e.outcome == TaskOutcome.SUCCESS and len(e.symbols) >= 2
        ]

        if not success_seqs:
            logger.warning(
                "ProcessConformance: no success sequences with length >= 2; "
                "transition model will be empty"
            )
            self.transitions_ = {}
            return

        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for seq in success_seqs:
            for a, b in zip(seq, seq[1:]):
                counts[a][b] += 1

        self.transitions_ = {}
        for src, dests in counts.items():
            total = sum(dests.values())
            self.transitions_[src] = {
                dst: cnt / total for dst, cnt in dests.items()
            }

        n_transitions = sum(len(d) for d in self.transitions_.values())
        logger.info(
            "ProcessConformance: built transition model from %d success "
            "sequences (%d states, %d transitions)",
            len(success_seqs),
            len(self.transitions_),
            n_transitions,
        )

    def predict(self, symbols: list[str]) -> float:
        """Return failure probability based on transition deviation.

        Args:
            symbols: Symbolized sequence.

        Returns:
            Failure probability in [0.0, 1.0].

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.transitions_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if len(symbols) < 2:
            return 0.5

        probs = [
            self.transitions_.get(a, {}).get(b, 0.0)
            for a, b in zip(symbols, symbols[1:])
        ]
        avg_prob = sum(probs) / len(probs)
        return 1.0 - avg_prob

    def predict_at_k(self, symbols: list[str], k: int) -> float:
        """Return failure probability using only the first *k* symbols.

        Args:
            symbols: Full symbolized sequence.
            k: Number of leading symbols to use.

        Returns:
            Failure probability in [0.0, 1.0].
        """
        return self.predict(symbols[:k])
