"""
Pattern Ranker - Coverage-based ranking of mined sequential patterns

Purpose: Rank BIDE-mined patterns by their failure-prediction quality and
cross-site generalization. Filters patterns by precision and website coverage
thresholds to produce a reliable failure signature library.

Ranking Formula:
    score = precision * log2(1 + coverage)
where:
    precision = |failed traces matching pattern| / |all traces matching pattern|
    coverage  = number of distinct websites where pattern appears

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.data_collection.trace_schema import FailureType, TaskOutcome
from src.mining.spmf_wrapper import RawPattern
from src.preprocessing.k_prefix import PrefixDataset, PrefixEntry

logger = logging.getLogger(__name__)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class ScoredPattern:
    """
    A sequential pattern with computed quality metrics.

    Attributes:
        symbols: Ordered symbol sequence of the pattern.
        support: Absolute support count from SPMF.
        precision: Fraction of matching traces that are failures.
        coverage: Number of distinct websites where pattern appears.
        score: Composite ranking score (precision * log2(1 + coverage)).
        failure_type: Most common failure type among matching failed traces.
        matching_trace_ids: IDs of traces whose prefix contains this pattern.
    """

    symbols: list[str]
    support: int
    precision: float = 0.0
    coverage: int = 0
    score: float = 0.0
    failure_type: Optional[FailureType] = None
    matching_trace_ids: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.symbols)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "symbols": self.symbols,
            "support": self.support,
            "precision": round(self.precision, 4),
            "coverage": self.coverage,
            "score": round(self.score, 4),
            "failure_type": self.failure_type.value if self.failure_type else None,
            "matching_trace_ids": self.matching_trace_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScoredPattern":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with ScoredPattern fields.

        Returns:
            Reconstructed ScoredPattern.
        """
        return cls(
            symbols=data["symbols"],
            support=data["support"],
            precision=data.get("precision", 0.0),
            coverage=data.get("coverage", 0),
            score=data.get("score", 0.0),
            failure_type=(FailureType(data["failure_type"])
                          if data.get("failure_type") else None),
            matching_trace_ids=data.get("matching_trace_ids", []),
        )


@dataclass
class RankerConfig:
    """
    Configuration for pattern ranking and filtering.

    Attributes:
        min_precision: Minimum precision threshold (patterns below are dropped).
        min_sites: Minimum number of distinct websites for cross-site validity.
        min_support: Minimum absolute support count.
    """

    min_precision: float = 0.5
    min_sites: int = 2
    min_support: int = 2

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "min_precision": self.min_precision,
            "min_sites": self.min_sites,
            "min_support": self.min_support,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RankerConfig":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with RankerConfig fields.

        Returns:
            Reconstructed RankerConfig.
        """
        return cls(
            min_precision=data.get("min_precision", 0.5),
            min_sites=data.get("min_sites", 2),
            min_support=data.get("min_support", 2),
        )


# =============================================================================
# PatternRanker
# =============================================================================

class PatternRanker:
    """
    Rank and filter mined sequential patterns by failure-prediction quality.
    """

    def __init__(self, config: Optional[RankerConfig] = None):
        """Initialize with ranking configuration.

        Args:
            config: RankerConfig with thresholds. Defaults to standard values.
        """
        self.config = config or RankerConfig()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def rank_patterns(
        self,
        raw_patterns: list[RawPattern],
        dataset: PrefixDataset,
    ) -> list[ScoredPattern]:
        """Compute metrics, filter, and rank a list of mined patterns.

        Args:
            raw_patterns: Patterns from SPMF (symbols + support).
            dataset: PrefixDataset with symbolized sequences and metadata.

        Returns:
            List of ScoredPattern, sorted by score descending, filtered
            by precision, coverage, and support thresholds.
        """
        logger.info(
            "Ranking %d raw patterns against %d traces",
            len(raw_patterns),
            len(dataset),
        )

        scored: list[ScoredPattern] = []
        for raw in raw_patterns:
            sp = self._compute_metrics(raw, dataset)
            scored.append(sp)

        filtered = self._filter(scored)
        filtered.sort(key=lambda p: p.score, reverse=True)

        logger.info(
            "Ranking complete: %d/%d patterns passed filters "
            "(min_precision=%.2f, min_sites=%d, min_support=%d)",
            len(filtered),
            len(scored),
            self.config.min_precision,
            self.config.min_sites,
            self.config.min_support,
        )
        return filtered

    # -----------------------------------------------------------------
    # Metric computation
    # -----------------------------------------------------------------

    def _compute_metrics(
        self,
        raw: RawPattern,
        dataset: PrefixDataset,
    ) -> ScoredPattern:
        """Compute precision, coverage, score, and failure type for a pattern.

        Args:
            raw: The raw mined pattern.
            dataset: PrefixDataset to match against.

        Returns:
            ScoredPattern with all metrics populated.
        """
        matching_entries: list[PrefixEntry] = []

        for entry in dataset.entries:
            if _is_subsequence(raw.symbols, entry.symbols):
                matching_entries.append(entry)

        matching_ids = [e.trace_id for e in matching_entries]
        total_matches = len(matching_entries)

        if total_matches == 0:
            return ScoredPattern(
                symbols=raw.symbols,
                support=raw.support,
                precision=0.0,
                coverage=0,
                score=0.0,
                failure_type=None,
                matching_trace_ids=matching_ids,
            )

        failed_entries = [
            e for e in matching_entries if e.outcome == TaskOutcome.FAILURE
        ]
        precision = len(failed_entries) / total_matches

        websites = {e.website for e in matching_entries}
        coverage = len(websites)

        score = precision * math.log2(1 + coverage)

        failure_type = self._assign_failure_type(failed_entries)

        return ScoredPattern(
            symbols=raw.symbols,
            support=raw.support,
            precision=precision,
            coverage=coverage,
            score=score,
            failure_type=failure_type,
            matching_trace_ids=matching_ids,
        )

    def _assign_failure_type(
        self, failed_entries: list[PrefixEntry]
    ) -> Optional[FailureType]:
        """Assign failure type by majority vote among matching failed traces.

        Args:
            failed_entries: PrefixEntry objects with outcome == FAILURE.

        Returns:
            Most common FailureType, or None if no failures have a type.
        """
        type_counts: Counter[FailureType] = Counter()
        for entry in failed_entries:
            if entry.failure_type is not None:
                type_counts[entry.failure_type] += 1

        if not type_counts:
            return None

        return type_counts.most_common(1)[0][0]

    # -----------------------------------------------------------------
    # Filtering
    # -----------------------------------------------------------------

    def _filter(self, patterns: list[ScoredPattern]) -> list[ScoredPattern]:
        """Apply precision, coverage, and support filters.

        Args:
            patterns: All scored patterns.

        Returns:
            Patterns that pass all thresholds.
        """
        return [
            p for p in patterns
            if (p.precision >= self.config.min_precision
                and p.coverage >= self.config.min_sites
                and p.support >= self.config.min_support)
        ]


# =============================================================================
# Helpers
# =============================================================================

def _is_subsequence(pattern: list[str], sequence: list[str]) -> bool:
    """Check if pattern is a subsequence of sequence.

    A pattern [A, B, C] matches sequence [X, A, Y, B, Z, C, W] because
    A, B, C appear in that order (not necessarily contiguous).

    Args:
        pattern: The pattern to search for.
        sequence: The sequence to search in.

    Returns:
        True if pattern is a subsequence of sequence.
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
