"""
TaSPM Baseline - Targeted Sequential Pattern Mining on failure traces

Purpose: Mine closed sequential patterns from failure-only traces using BIDE+
(approximating TaSPM) and score new sequences by the fraction of failure
patterns they contain.  Unlike the main approach, this baseline operates on
full sequences (no k-prefix extraction) and skips coverage-based ranking.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from src.baselines.base import BaseBaseline
from src.data_collection.trace_schema import TaskOutcome
from src.mining.spmf_wrapper import RawPattern, SPMFConfig, SPMFWrapper
from src.preprocessing.k_prefix import PrefixEntry

logger = logging.getLogger(__name__)


class TaSPMBaseline(BaseBaseline):
    """TaSPM-approximation baseline via BIDE+ on failure-only traces.

    Mines closed sequential patterns exclusively from failure sequences,
    then scores unseen sequences by the fraction of mined patterns that
    appear as subsequences.
    """

    name: str = "taspm"

    def __init__(
        self,
        min_support: float = 0.05,
        spmf_jar: str = "lib/spmf.jar",
    ) -> None:
        """
        Args:
            min_support: Minimum relative support for BIDE+ mining.
            spmf_jar: Path to the SPMF jar file.
        """
        self.min_support = min_support
        self.spmf_jar = Path(spmf_jar)
        self.patterns_: list[RawPattern] = []
        self._work_dir: tempfile.TemporaryDirectory[str] | None = None

    # ------------------------------------------------------------------
    # BaseBaseline interface
    # ------------------------------------------------------------------

    def fit(self, entries: list[PrefixEntry]) -> None:
        """Mine BIDE+ patterns from failure-only full sequences.

        Args:
            entries: Training prefix entries with symbols and outcomes.
        """
        self._cleanup_work_dir()
        self._work_dir = tempfile.TemporaryDirectory(prefix="taspm_")

        failure_seqs = [
            e.symbols for e in entries
            if e.outcome == TaskOutcome.FAILURE and e.symbols
        ]

        if len(failure_seqs) < 2:
            logger.warning(
                "TaSPM: fewer than 2 failure sequences (%d); "
                "no patterns will be mined",
                len(failure_seqs),
            )
            self.patterns_ = []
            return

        config = SPMFConfig(
            spmf_jar_path=self.spmf_jar,
            min_support=self.min_support,
        )
        wrapper = SPMFWrapper(config)
        patterns, _vocab = wrapper.mine_patterns(
            sequences=failure_seqs,
            work_dir=Path(self._work_dir.name),
            min_support=self.min_support,
        )
        self.patterns_ = patterns
        logger.info(
            "TaSPM: mined %d patterns from %d failure sequences "
            "(min_support=%.3f)",
            len(self.patterns_),
            len(failure_seqs),
            self.min_support,
        )

    def predict(self, symbols: list[str]) -> float:
        """Return failure probability as the fraction of matched patterns.

        Args:
            symbols: Symbolized sequence.

        Returns:
            Failure probability in [0.0, 1.0].
        """
        if not self.patterns_:
            return 0.0

        matched = sum(
            1 for p in self.patterns_
            if _is_subsequence(p.symbols, symbols)
        )
        return matched / len(self.patterns_)

    def predict_at_k(self, symbols: list[str], k: int) -> float:
        """Return failure probability using only the first *k* symbols.

        Args:
            symbols: Full symbolized sequence.
            k: Number of leading symbols to use.

        Returns:
            Failure probability in [0.0, 1.0].
        """
        return self.predict(symbols[:k])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_work_dir(self) -> None:
        """Remove the previous temporary working directory if it exists."""
        if self._work_dir is not None:
            try:
                self._work_dir.cleanup()
            except OSError:
                logger.debug("TaSPM: could not clean up previous work dir")
            self._work_dir = None


# ----------------------------------------------------------------------
# Module-level utility
# ----------------------------------------------------------------------

def _is_subsequence(pattern: list[str], sequence: list[str]) -> bool:
    """Check whether *pattern* is an ordered subsequence of *sequence*.

    Args:
        pattern: Candidate pattern symbols.
        sequence: Full symbol sequence to search within.

    Returns:
        True if every symbol in *pattern* appears in *sequence* in order.
    """
    it = iter(sequence)
    return all(symbol in it for symbol in pattern)
