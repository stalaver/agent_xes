"""
Pattern Matcher - Subsequence matching of failure patterns against live traces

Purpose: Match ranked failure-signature patterns from the offline library
against a symbolized trace prefix, recording match positions and computing
an aggregate failure score for the online detection phase.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

from src.mining.pattern_ranker import ScoredPattern
from src.mining.signature_library import SignatureLibrary

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """A single pattern that matched against a symbol sequence.

    Attributes:
        pattern: The scored pattern from the library.
        positions: Indices in the input sequence where each pattern symbol matched.
        score: The pattern's ranking score (precision * log2(1 + coverage)).
    """

    pattern: ScoredPattern
    positions: list[int]
    score: float


class PatternMatcher:
    """Match library patterns against symbolized trace prefixes.

    Iterates over ranked failure patterns, checks each for subsequence
    membership in the input, and computes an aggregate failure score
    as the fraction of total pattern weight that matches.
    """

    def __init__(
        self,
        library: Union[SignatureLibrary, list[ScoredPattern]],
        max_patterns: Optional[int] = None,
    ) -> None:
        """Initialize with a pattern library.

        Args:
            library: A SignatureLibrary or a plain list of ScoredPattern objects.
            max_patterns: If set, only use the top N patterns by score.
        """
        if isinstance(library, SignatureLibrary):
            patterns = library.patterns
        else:
            patterns = list(library)

        if max_patterns is not None:
            patterns = patterns[:max_patterns]

        self._patterns = patterns
        self._total_score = sum(p.score for p in self._patterns) or 1.0

        logger.info(
            "PatternMatcher initialized with %d patterns (total_score=%.4f)",
            len(self._patterns),
            self._total_score,
        )

    @property
    def patterns(self) -> list[ScoredPattern]:
        """The patterns used for matching."""
        return self._patterns

    @property
    def total_score(self) -> float:
        """Sum of all pattern scores (denominator for match_score)."""
        return self._total_score

    def match(self, symbols: list[str]) -> list[PatternMatch]:
        """Find all library patterns that are subsequences of the input.

        Args:
            symbols: Symbolized trace prefix to match against.

        Returns:
            List of PatternMatch objects sorted by score descending.
        """
        matches: list[PatternMatch] = []
        for pattern in self._patterns:
            positions = self.find_match_positions(pattern.symbols, symbols)
            if positions is not None:
                matches.append(PatternMatch(
                    pattern=pattern,
                    positions=positions,
                    score=pattern.score,
                ))
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def match_score(self, symbols: list[str]) -> float:
        """Compute an aggregate failure score for the input.

        Returns the fraction of total pattern score that matches as
        subsequences, capped at 1.0.

        Args:
            symbols: Symbolized trace prefix.

        Returns:
            Failure probability in [0.0, 1.0].
        """
        if not self._patterns:
            return 0.0

        matched_score = sum(
            p.score for p in self._patterns
            if self.is_subsequence(p.symbols, symbols)
        )
        return min(matched_score / self._total_score, 1.0)

    @staticmethod
    def is_subsequence(pattern: list[str], sequence: list[str]) -> bool:
        """Check whether *pattern* is an ordered subsequence of *sequence*.

        A pattern ``[A, B, C]`` matches ``[X, A, Y, B, Z, C, W]`` because
        A, B, C appear in that order (not necessarily contiguous).

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

    @staticmethod
    def find_match_positions(
        pattern: list[str], sequence: list[str],
    ) -> Optional[list[int]]:
        """Find the indices in *sequence* where each *pattern* symbol matches.

        Works like ``is_subsequence`` but records the position of each
        matched element for interpretability output.

        Args:
            pattern: Ordered pattern symbols.
            sequence: Full symbol sequence to search within.

        Returns:
            List of sequence indices (one per pattern symbol), or None if
            the pattern is not a subsequence.
        """
        if not pattern:
            return []
        if len(pattern) > len(sequence):
            return None

        positions: list[int] = []
        pat_idx = 0
        for seq_idx, symbol in enumerate(sequence):
            if symbol == pattern[pat_idx]:
                positions.append(seq_idx)
                pat_idx += 1
                if pat_idx == len(pattern):
                    return positions
        return None
