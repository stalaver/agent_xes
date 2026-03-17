"""
Failure Predictor - Online decision-making for early failure termination

Purpose: Consume pattern-match results from PatternMatcher and apply
threshold-based rules to decide whether a running agent trace should be
terminated, flagged for human review, or allowed to continue.

Decision thresholds:
    - TERMINATE: confidence >= terminate_threshold (default 0.85)
    - ALERT:     confidence >= alert_threshold     (default 0.70)
    - CONTINUE:  confidence < alert_threshold

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.data_collection.trace_schema import AgentTrace
from src.detection.pattern_matcher import PatternMatch, PatternMatcher
from src.preprocessing.symbolizer import TraceSymbolizer
from src.utils.symbol_descriptions import describe_symbol

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "terminate": 0.85,
    "alert": 0.70,
}


class Decision(Enum):
    """Online detection decision for a running agent."""

    TERMINATE = "terminate"
    ALERT = "alert"
    CONTINUE = "continue"


@dataclass
class PredictionResult:
    """Result of running the failure predictor on a symbolized prefix.

    Attributes:
        decision: Whether to terminate, alert, or continue.
        confidence: Aggregate match score from PatternMatcher (0.0-1.0).
        matching_patterns: All pattern matches found in the prefix.
        explanation: Human-readable summary of the top matching patterns.
    """

    decision: Decision
    confidence: float
    matching_patterns: list[PatternMatch] = field(default_factory=list)
    explanation: str = ""


class FailurePredictor:
    """Threshold-based failure predictor for online detection.

    Takes pattern-match scores and applies configurable thresholds
    to produce actionable decisions with human-readable explanations.
    """

    def __init__(
        self,
        matcher: PatternMatcher,
        thresholds: Optional[dict[str, float]] = None,
    ) -> None:
        """Initialize the predictor.

        Args:
            matcher: PatternMatcher loaded with a failure-pattern library.
            thresholds: Dict with ``terminate`` and ``alert`` keys mapping
                to float thresholds.  Defaults to 0.85 / 0.70.
        """
        self._matcher = matcher
        self._thresholds = thresholds or dict(DEFAULT_THRESHOLDS)

        logger.info(
            "FailurePredictor initialized (terminate=%.2f, alert=%.2f)",
            self._thresholds["terminate"],
            self._thresholds["alert"],
        )

    @property
    def matcher(self) -> PatternMatcher:
        """The underlying PatternMatcher."""
        return self._matcher

    @property
    def thresholds(self) -> dict[str, float]:
        """Current decision thresholds."""
        return self._thresholds

    def predict(self, symbols: list[str]) -> PredictionResult:
        """Score a symbol sequence and produce a decision.

        Args:
            symbols: Symbolized trace prefix.

        Returns:
            PredictionResult with decision, confidence, matches, and
            explanation.
        """
        matches = self._matcher.match(symbols)
        confidence = self._matcher.match_score(symbols)
        decision = self._apply_thresholds(confidence)
        explanation = self._build_explanation(decision, confidence, matches)

        logger.debug(
            "Prediction: decision=%s confidence=%.4f matches=%d",
            decision.value,
            confidence,
            len(matches),
        )

        return PredictionResult(
            decision=decision,
            confidence=confidence,
            matching_patterns=matches,
            explanation=explanation,
        )

    def predict_trace(
        self,
        trace: AgentTrace,
        k: int,
        symbolizer: TraceSymbolizer,
    ) -> PredictionResult:
        """Symbolize a trace prefix and predict.

        Convenience method that handles symbolization before calling
        ``predict()``.

        Args:
            trace: The agent execution trace.
            k: Number of prefix steps to symbolize.
            symbolizer: TraceSymbolizer configured at the desired level.

        Returns:
            PredictionResult for the first *k* steps.
        """
        symbols = symbolizer.symbolize_prefix(trace, k)
        return self.predict(symbols)

    def _apply_thresholds(self, confidence: float) -> Decision:
        """Map a confidence score to a decision via thresholds.

        Args:
            confidence: Aggregate failure score in [0.0, 1.0].

        Returns:
            Decision enum value.
        """
        if confidence >= self._thresholds["terminate"]:
            return Decision.TERMINATE
        if confidence >= self._thresholds["alert"]:
            return Decision.ALERT
        return Decision.CONTINUE

    @staticmethod
    def _build_explanation(
        decision: Decision,
        confidence: float,
        matches: list[PatternMatch],
        top_n: int = 3,
    ) -> str:
        """Build a human-readable explanation of the prediction.

        Args:
            decision: The decision that was made.
            confidence: Aggregate failure score.
            matches: All pattern matches.
            top_n: Number of top matches to include in the explanation.

        Returns:
            Multi-line explanation string.
        """
        lines: list[str] = [
            f"Decision: {decision.value.upper()} (confidence={confidence:.4f})"
        ]

        if not matches:
            lines.append("No failure patterns matched this prefix.")
            return "\n".join(lines)

        lines.append(f"{len(matches)} pattern(s) matched. Top {min(top_n, len(matches))}:")

        for i, m in enumerate(matches[:top_n], start=1):
            descriptions = [describe_symbol(s) for s in m.pattern.symbols]
            pattern_desc = " -> ".join(descriptions)
            pos_str = ", ".join(str(p) for p in m.positions)
            lines.append(
                f"  {i}. [{pattern_desc}] "
                f"(score={m.score:.4f}, positions=[{pos_str}])"
            )

        return "\n".join(lines)
