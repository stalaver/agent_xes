"""
K-Prefix Extractor - Extract and symbolize trace prefixes for pattern mining

Purpose: Extract the first K steps from agent traces and produce symbolized
sequences ready for SPMF input. Provides batch processing and a PrefixDataset
container that pairs sequences with trace metadata for downstream ranking.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.data_collection.trace_schema import (
    AgentTrace,
    TraceStep,
    TaskOutcome,
    FailureType,
)
from src.preprocessing.symbolizer import TraceSymbolizer

logger = logging.getLogger(__name__)


# =============================================================================
# PrefixEntry
# =============================================================================

@dataclass
class PrefixEntry:
    """
    A single symbolized prefix with its trace metadata.

    Attributes:
        trace_id: Unique identifier for the source trace.
        website: Website the trace was collected from.
        outcome: Task outcome (success, failure, timeout, error).
        failure_type: Failure category if outcome is failure.
        symbols: Symbolized sequence for the prefix.
    """

    trace_id: str
    website: str
    outcome: TaskOutcome
    failure_type: Optional[FailureType]
    symbols: list[str]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "trace_id": self.trace_id,
            "website": self.website,
            "outcome": self.outcome.value,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "symbols": self.symbols,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrefixEntry":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with PrefixEntry fields.

        Returns:
            Reconstructed PrefixEntry.
        """
        return cls(
            trace_id=data["trace_id"],
            website=data["website"],
            outcome=TaskOutcome(data["outcome"]),
            failure_type=(FailureType(data["failure_type"])
                          if data.get("failure_type") else None),
            symbols=data["symbols"],
        )


# =============================================================================
# PrefixDataset
# =============================================================================

@dataclass
class PrefixDataset:
    """
    Collection of symbolized prefixes with metadata for pattern mining.

    Holds all prefix entries extracted from a batch of traces, along with
    the extraction parameters used to produce them.

    Attributes:
        entries: List of PrefixEntry objects.
        k: Number of prefix steps used.
        abstraction_level: Symbolization level (0, 1, or 2).
    """

    entries: list[PrefixEntry] = field(default_factory=list)
    k: int = 10
    abstraction_level: int = 1

    @property
    def sequences(self) -> list[list[str]]:
        """All symbol sequences (convenience accessor for SPMF input)."""
        return [entry.symbols for entry in self.entries]

    @property
    def websites(self) -> set[str]:
        """Distinct websites in the dataset."""
        return {entry.website for entry in self.entries}

    @property
    def trace_ids(self) -> list[str]:
        """All trace IDs in order."""
        return [entry.trace_id for entry in self.entries]

    def get_failed_entries(self) -> list[PrefixEntry]:
        """Return entries whose outcome is FAILURE."""
        return [
            e for e in self.entries
            if e.outcome == TaskOutcome.FAILURE
        ]

    def get_successful_entries(self) -> list[PrefixEntry]:
        """Return entries whose outcome is SUCCESS."""
        return [
            e for e in self.entries
            if e.outcome == TaskOutcome.SUCCESS
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def summary(self) -> dict:
        """Return summary statistics about the dataset.

        Returns:
            Dictionary with counts by outcome, website, and failure type.
        """
        outcome_counts: dict[str, int] = {}
        website_counts: dict[str, int] = {}
        failure_counts: dict[str, int] = {}

        for entry in self.entries:
            ov = entry.outcome.value
            outcome_counts[ov] = outcome_counts.get(ov, 0) + 1

            website_counts[entry.website] = (
                website_counts.get(entry.website, 0) + 1
            )

            if entry.failure_type:
                fv = entry.failure_type.value
                failure_counts[fv] = failure_counts.get(fv, 0) + 1

        return {
            "total": len(self.entries),
            "k": self.k,
            "abstraction_level": self.abstraction_level,
            "outcomes": outcome_counts,
            "websites": website_counts,
            "failure_types": failure_counts,
        }

    def to_dict(self) -> dict:
        """Serialize the full dataset to dictionary."""
        return {
            "k": self.k,
            "abstraction_level": self.abstraction_level,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrefixDataset":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with PrefixDataset fields.

        Returns:
            Reconstructed PrefixDataset.
        """
        return cls(
            k=data["k"],
            abstraction_level=data["abstraction_level"],
            entries=[PrefixEntry.from_dict(e) for e in data.get("entries", [])],
        )

    def save(self, path: Path) -> None:
        """Save dataset to JSON file.

        Args:
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved PrefixDataset (%d entries) to %s", len(self), path)

    @classmethod
    def load(cls, path: Path) -> "PrefixDataset":
        """Load dataset from JSON file.

        Args:
            path: Input file path.

        Returns:
            Loaded PrefixDataset.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = cls.from_dict(data)
        logger.info(
            "Loaded PrefixDataset (%d entries) from %s", len(dataset), path
        )
        return dataset


# =============================================================================
# Extraction functions
# =============================================================================

def extract_k_prefix(trace: AgentTrace, k: int) -> list[TraceStep]:
    """Extract the first K steps from a trace.

    Thin wrapper around AgentTrace.get_k_prefix() for consistency.

    Args:
        trace: Agent execution trace.
        k: Number of prefix steps.

    Returns:
        List of the first K TraceStep objects (or fewer if trace is shorter).
    """
    return trace.get_k_prefix(k)


def extract_symbolized_prefix(
    trace: AgentTrace,
    k: int,
    symbolizer: TraceSymbolizer,
) -> list[str]:
    """Extract and symbolize the first K steps of a trace.

    Args:
        trace: Agent execution trace.
        k: Number of prefix steps.
        symbolizer: Configured TraceSymbolizer instance.

    Returns:
        List of symbol strings for the prefix steps.
    """
    return symbolizer.symbolize_prefix(trace, k)


def batch_extract_prefixes(
    traces: list[AgentTrace],
    k: int,
    symbolizer: TraceSymbolizer,
) -> PrefixDataset:
    """Extract symbolized prefixes from a batch of traces.

    Produces a PrefixDataset containing one PrefixEntry per trace,
    pairing the symbolized sequence with the trace metadata needed
    for downstream pattern ranking.

    Args:
        traces: List of agent execution traces.
        k: Number of prefix steps.
        symbolizer: Configured TraceSymbolizer instance.

    Returns:
        PrefixDataset ready for SPMF input preparation and pattern ranking.
    """
    entries: list[PrefixEntry] = []

    for trace in traces:
        symbols = symbolizer.symbolize_prefix(trace, k)
        entry = PrefixEntry(
            trace_id=trace.metadata.trace_id,
            website=trace.metadata.website,
            outcome=trace.metadata.outcome,
            failure_type=trace.metadata.failure_type,
            symbols=symbols,
        )
        entries.append(entry)

    dataset = PrefixDataset(
        entries=entries,
        k=k,
        abstraction_level=symbolizer.abstraction_level,
    )

    logger.info(
        "Extracted %d prefixes (k=%d, level=%d): %s",
        len(dataset),
        k,
        symbolizer.abstraction_level,
        dataset.summary(),
    )
    return dataset
