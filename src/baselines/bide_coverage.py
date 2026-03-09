"""
BIDE Coverage Baseline - Closed sequential pattern mining with coverage-based ranking

Purpose: Wrap the main BIDE+ offline pipeline (mine -> rank -> match) as a
baseline that implements the BaseBaseline interface for uniform comparison
with the other six baselines.  Mines closed sequential patterns from failure
traces, ranks them by precision * log2(1 + site_coverage), and scores unseen
sequences by the fraction of total ranked-pattern weight that matches.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from src.baselines.base import BaseBaseline
from src.data_collection.trace_schema import TaskOutcome
from src.mining.pattern_ranker import PatternRanker, RankerConfig, ScoredPattern
from src.mining.spmf_wrapper import SPMFConfig, SPMFWrapper
from src.preprocessing.k_prefix import PrefixDataset, PrefixEntry

logger = logging.getLogger(__name__)


class BIDECoverageBaseline(BaseBaseline):
    """BIDE+ baseline with coverage-based pattern ranking.

    Mines closed sequential patterns from failure traces using BIDE+,
    ranks them by precision * log2(1 + site_coverage), and scores
    unseen sequences by the fraction of total ranked-pattern weight
    that matches as subsequences.
    """

    name: str = "bide_coverage"

    def __init__(
        self,
        min_support: float = 0.3,
        top_k_patterns: int = 50,
        abstraction_level: int = 1,
        spmf_jar_path: str = "lib/spmf.jar",
        min_precision: float = 0.5,
        min_sites: int = 1,
    ) -> None:
        """
        Args:
            min_support: Minimum relative support for BIDE+ mining.
            top_k_patterns: Number of top-ranked patterns to retain.
            abstraction_level: Symbolization level metadata (0, 1, or 2).
                Stored for PrefixDataset construction; actual symbolization
                happens upstream before ``fit()`` is called.
            spmf_jar_path: Path to the SPMF jar file.
            min_precision: Minimum precision threshold for pattern filtering.
            min_sites: Minimum distinct websites for cross-site validity.
        """
        self.min_support = min_support
        self.top_k_patterns = top_k_patterns
        self.abstraction_level = abstraction_level
        self.spmf_jar_path = Path(spmf_jar_path)
        self.min_precision = min_precision
        self.min_sites = min_sites

        self.patterns_: list[ScoredPattern] = []
        self._total_score: float = 0.0
        self._work_dir: tempfile.TemporaryDirectory[str] | None = None

    # ------------------------------------------------------------------
    # BaseBaseline interface
    # ------------------------------------------------------------------

    def fit(self, entries: list[PrefixEntry]) -> None:
        """Mine BIDE+ patterns and rank by coverage-weighted precision.

        Pipeline: extract failure sequences -> BIDE+ mining ->
        coverage-based ranking against all entries -> keep top-k patterns.

        Args:
            entries: Training prefix entries with symbols and outcomes.
        """
        self._cleanup_work_dir()
        self._work_dir = tempfile.TemporaryDirectory(prefix="bide_cov_")

        failure_seqs = [
            e.symbols for e in entries
            if e.outcome.is_failure and e.symbols
        ]

        if len(failure_seqs) < 2:
            logger.warning(
                "BIDECoverage: fewer than 2 failure sequences (%d); "
                "no patterns will be mined",
                len(failure_seqs),
            )
            self.patterns_ = []
            self._total_score = 0.0
            return

        spmf_config = SPMFConfig(
            spmf_jar_path=self.spmf_jar_path,
            min_support=self.min_support,
        )
        wrapper = SPMFWrapper(spmf_config)
        raw_patterns, _vocab = wrapper.mine_patterns(
            sequences=failure_seqs,
            work_dir=Path(self._work_dir.name),
            min_support=self.min_support,
        )

        dataset = PrefixDataset(
            entries=list(entries),
            k=max((len(e.symbols) for e in entries), default=0),
            abstraction_level=self.abstraction_level,
        )
        ranker_config = RankerConfig(
            min_precision=self.min_precision,
            min_sites=self.min_sites,
        )
        ranker = PatternRanker(ranker_config)
        ranked = ranker.rank_patterns(raw_patterns, dataset)

        self.patterns_ = ranked[: self.top_k_patterns]
        self._total_score = sum(p.score for p in self.patterns_) or 1.0

        logger.info(
            "BIDECoverage: kept %d/%d ranked patterns from %d raw "
            "(min_support=%.3f, top_k=%d)",
            len(self.patterns_),
            len(ranked),
            len(raw_patterns),
            self.min_support,
            self.top_k_patterns,
        )

    def predict(self, symbols: list[str]) -> float:
        """Score a symbol sequence against ranked failure patterns.

        Returns the fraction of total pattern score that matches as
        subsequences of the input.

        Args:
            symbols: Symbolized prefix sequence.

        Returns:
            Failure probability in [0.0, 1.0].
        """
        if not self.patterns_:
            return 0.0

        matched_score = sum(
            p.score for p in self.patterns_
            if self.is_subsequence(p.symbols, symbols)
        )
        return min(matched_score / self._total_score, 1.0)

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
    # Subsequence matching (extractable for Phase 3A PatternMatcher)
    # ------------------------------------------------------------------

    @staticmethod
    def is_subsequence(pattern: list[str], sequence: list[str]) -> bool:
        """Check whether *pattern* is an ordered subsequence of *sequence*.

        A pattern ``[A, B, C]`` matches ``[X, A, Y, B, Z, C, W]`` because
        A, B, C appear in that order (not necessarily contiguous).

        This method is intentionally a public static method so it can be
        extracted into the Phase 3A ``PatternMatcher`` without changes.

        Args:
            pattern: Candidate pattern symbols.
            sequence: Full symbol sequence to search within.

        Returns:
            True if every symbol in *pattern* appears in *sequence* in order.
        """
        if not pattern:
            return True
        if len(pattern) > len(sequence):
            return False

        pat_idx = 0
        for symbol in sequence:
            if symbol == pattern[pat_idx]:
                pat_idx += 1
                if pat_idx == len(pattern):
                    return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_work_dir(self) -> None:
        """Remove the previous temporary working directory if it exists."""
        if self._work_dir is not None:
            try:
                self._work_dir.cleanup()
            except OSError:
                logger.debug("BIDECoverage: could not clean up previous work dir")
            self._work_dir = None
