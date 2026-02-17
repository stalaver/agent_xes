"""
Signature Library - Persistent storage and retrieval of failure patterns

Purpose: Store ranked failure-signature patterns and provide efficient
lookup for the online detection phase. Supports JSON persistence,
subsequence matching against live traces, and retrieval by failure type.

Persistence Format:
    data/patterns/signature_library_{level}_{k}.json

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.data_collection.trace_schema import FailureType
from src.mining.pattern_ranker import ScoredPattern, _is_subsequence

logger = logging.getLogger(__name__)


class SignatureLibrary:
    """
    Persistent collection of ranked failure-signature patterns.

    Wraps a list of ScoredPattern objects, provides matching against
    live symbol sequences, and supports JSON serialization.
    """

    def __init__(self, patterns: Optional[list[ScoredPattern]] = None):
        """Initialize the library with an optional list of patterns.

        Args:
            patterns: Pre-ranked ScoredPattern list (sorted by score desc).
        """
        self._patterns: list[ScoredPattern] = patterns or []

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def patterns(self) -> list[ScoredPattern]:
        """All patterns in the library, ordered by score descending."""
        return self._patterns

    def __len__(self) -> int:
        return len(self._patterns)

    def __iter__(self):
        return iter(self._patterns)

    # -----------------------------------------------------------------
    # Matching
    # -----------------------------------------------------------------

    def match(self, sequence: list[str]) -> list[ScoredPattern]:
        """Find all patterns that match a given symbol sequence.

        A pattern matches if it is a subsequence of the input sequence.
        Results are returned sorted by score descending (library order).

        Args:
            sequence: Symbolized trace prefix to match against.

        Returns:
            List of matching ScoredPattern objects, sorted by score.
        """
        return [
            p for p in self._patterns
            if _is_subsequence(p.symbols, sequence)
        ]

    def top_n_matches(
        self, sequence: list[str], n: int
    ) -> list[ScoredPattern]:
        """Find the top N matching patterns by score.

        Args:
            sequence: Symbolized trace prefix.
            n: Maximum number of matches to return.

        Returns:
            Up to N matching patterns, sorted by score descending.
        """
        matches = self.match(sequence)
        return matches[:n]

    def has_match(self, sequence: list[str]) -> bool:
        """Check whether any pattern in the library matches the sequence.

        Short-circuits on the first match for efficiency.

        Args:
            sequence: Symbolized trace prefix.

        Returns:
            True if at least one pattern matches.
        """
        for p in self._patterns:
            if _is_subsequence(p.symbols, sequence):
                return True
        return False

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------

    def get_by_failure_type(
        self, failure_type: FailureType
    ) -> list[ScoredPattern]:
        """Retrieve patterns associated with a specific failure type.

        Args:
            failure_type: The FailureType to filter by.

        Returns:
            List of patterns with the given failure type, sorted by score.
        """
        return [
            p for p in self._patterns
            if p.failure_type == failure_type
        ]

    def get_by_min_score(self, min_score: float) -> list[ScoredPattern]:
        """Retrieve patterns above a minimum score threshold.

        Args:
            min_score: Minimum score to include.

        Returns:
            List of patterns with score >= min_score.
        """
        return [p for p in self._patterns if p.score >= min_score]

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> dict:
        """Compute summary statistics about the library.

        Returns:
            Dictionary with pattern count, score/precision/coverage
            distributions, and failure type breakdown.
        """
        if not self._patterns:
            return {"total": 0}

        scores = [p.score for p in self._patterns]
        precisions = [p.precision for p in self._patterns]
        coverages = [p.coverage for p in self._patterns]
        lengths = [len(p) for p in self._patterns]

        failure_type_counts: dict[str, int] = {}
        for p in self._patterns:
            key = p.failure_type.value if p.failure_type else "none"
            failure_type_counts[key] = failure_type_counts.get(key, 0) + 1

        return {
            "total": len(self._patterns),
            "score": {
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "mean": round(sum(scores) / len(scores), 4),
            },
            "precision": {
                "min": round(min(precisions), 4),
                "max": round(max(precisions), 4),
                "mean": round(sum(precisions) / len(precisions), 4),
            },
            "coverage": {
                "min": min(coverages),
                "max": max(coverages),
                "mean": round(sum(coverages) / len(coverages), 2),
            },
            "pattern_length": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": round(sum(lengths) / len(lengths), 2),
            },
            "failure_types": failure_type_counts,
        }

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the signature library to a JSON file.

        Args:
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "num_patterns": len(self._patterns),
            "patterns": [p.to_dict() for p in self._patterns],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Saved SignatureLibrary (%d patterns) to %s",
            len(self),
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SignatureLibrary":
        """Load a signature library from a JSON file.

        Args:
            path: Input file path.

        Returns:
            Loaded SignatureLibrary.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        patterns = [
            ScoredPattern.from_dict(p) for p in data.get("patterns", [])
        ]
        library = cls(patterns=patterns)
        logger.info(
            "Loaded SignatureLibrary (%d patterns) from %s",
            len(library),
            path,
        )
        return library

    def to_dict(self) -> dict:
        """Serialize the full library to dictionary."""
        return {
            "version": 1,
            "num_patterns": len(self._patterns),
            "patterns": [p.to_dict() for p in self._patterns],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SignatureLibrary":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with patterns list.

        Returns:
            Reconstructed SignatureLibrary.
        """
        patterns = [
            ScoredPattern.from_dict(p) for p in data.get("patterns", [])
        ]
        return cls(patterns=patterns)
