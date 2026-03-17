"""
Token Calculator - Estimate computational savings from early termination

Purpose: Quantify how many tokens (and therefore cost/time) would be saved
if a failure predictor terminates an agent trace early instead of letting
it run to completion.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.data_collection.trace_schema import AgentTrace
from src.detection.failure_predictor import Decision, FailurePredictor
from src.preprocessing.symbolizer import TraceSymbolizer

logger = logging.getLogger(__name__)


@dataclass
class SavingsEstimate:
    """Token savings estimate for a single trace.

    Attributes:
        trace_id: Unique identifier for the trace.
        actual_steps: Total steps in the original trace.
        terminated_at_step: Step at which the predictor would terminate.
        actual_tokens: Total tokens consumed in the original trace.
        saved_tokens: Tokens that would be saved by early termination.
        savings_percentage: Fraction of tokens saved (0.0-1.0).
    """

    trace_id: str
    actual_steps: int
    terminated_at_step: int
    actual_tokens: int
    saved_tokens: int
    savings_percentage: float


class TokenCalculator:
    """Estimate token savings from early failure termination.

    Uses per-step token counts from traces when available, falling
    back to a corpus-level average when individual step counts are zero.
    """

    def __init__(self, avg_tokens_per_step: float) -> None:
        """Initialize with an average tokens-per-step estimate.

        Args:
            avg_tokens_per_step: Mean tokens per step across the corpus.
        """
        self._avg_tokens_per_step = avg_tokens_per_step

        logger.info(
            "TokenCalculator initialized (avg_tokens_per_step=%.1f)",
            self._avg_tokens_per_step,
        )

    @property
    def avg_tokens_per_step(self) -> float:
        """Corpus-level average tokens per step."""
        return self._avg_tokens_per_step

    @classmethod
    def from_corpus(cls, traces: list[AgentTrace]) -> "TokenCalculator":
        """Compute avg_tokens_per_step from a corpus of traces.

        Args:
            traces: Agent traces with per-step token counts.

        Returns:
            TokenCalculator initialized with the computed average.
        """
        total_tokens = 0
        total_steps = 0
        for trace in traces:
            for step in trace.steps:
                step_tokens = step.prompt_tokens + step.completion_tokens
                total_tokens += step_tokens
                total_steps += 1

        avg = total_tokens / total_steps if total_steps > 0 else 0.0

        logger.info(
            "TokenCalculator.from_corpus: %d tokens across %d steps -> avg=%.1f",
            total_tokens,
            total_steps,
            avg,
        )
        return cls(avg_tokens_per_step=avg)

    def estimate_savings(
        self, trace: AgentTrace, terminate_at: int,
    ) -> SavingsEstimate:
        """Estimate savings for a single trace terminated at a given step.

        Uses actual per-step token counts when available. If the trace
        has no token data (all zeros), falls back to the corpus average.

        Args:
            trace: The full agent execution trace.
            terminate_at: Step index at which to terminate (1-based count
                of steps executed).

        Returns:
            SavingsEstimate with token counts and percentage.
        """
        actual_tokens = self._trace_tokens(trace)
        prefix_tokens = self._prefix_tokens(trace, terminate_at)

        saved = max(actual_tokens - prefix_tokens, 0)
        pct = saved / actual_tokens if actual_tokens > 0 else 0.0

        return SavingsEstimate(
            trace_id=trace.metadata.trace_id,
            actual_steps=len(trace.steps),
            terminated_at_step=terminate_at,
            actual_tokens=actual_tokens,
            saved_tokens=saved,
            savings_percentage=pct,
        )

    def batch_estimate(
        self,
        traces: list[AgentTrace],
        predictor: FailurePredictor,
        symbolizer: TraceSymbolizer,
        k: int,
    ) -> list[SavingsEstimate]:
        """Run the predictor on each trace and estimate savings.

        For traces where the predictor decides TERMINATE, savings are
        computed at step *k*. For ALERT and CONTINUE, savings are zero.

        Args:
            traces: All traces to evaluate.
            predictor: Fitted FailurePredictor.
            symbolizer: TraceSymbolizer at the desired abstraction level.
            k: Prefix length for prediction.

        Returns:
            One SavingsEstimate per trace.
        """
        estimates: list[SavingsEstimate] = []
        for trace in traces:
            result = predictor.predict_trace(trace, k, symbolizer)

            if result.decision == Decision.TERMINATE:
                est = self.estimate_savings(trace, terminate_at=k)
            else:
                actual_tokens = self._trace_tokens(trace)
                est = SavingsEstimate(
                    trace_id=trace.metadata.trace_id,
                    actual_steps=len(trace.steps),
                    terminated_at_step=len(trace.steps),
                    actual_tokens=actual_tokens,
                    saved_tokens=0,
                    savings_percentage=0.0,
                )
            estimates.append(est)

        logger.info(
            "batch_estimate: %d traces, %d terminated at k=%d",
            len(traces),
            sum(1 for e in estimates if e.saved_tokens > 0),
            k,
        )
        return estimates

    @staticmethod
    def summary(estimates: list[SavingsEstimate]) -> dict:
        """Compute aggregate savings statistics.

        Args:
            estimates: List of per-trace savings estimates.

        Returns:
            Dict with total_traces, traces_terminated, total_tokens_saved,
            overall_savings_percentage, and avg_savings_per_terminated.
        """
        if not estimates:
            return {
                "total_traces": 0,
                "traces_terminated": 0,
                "total_tokens_saved": 0,
                "total_tokens_original": 0,
                "overall_savings_percentage": 0.0,
                "avg_savings_per_terminated": 0.0,
            }

        terminated = [e for e in estimates if e.saved_tokens > 0]
        total_saved = sum(e.saved_tokens for e in estimates)
        total_original = sum(e.actual_tokens for e in estimates)
        overall_pct = total_saved / total_original if total_original > 0 else 0.0
        avg_per_terminated = (
            sum(e.savings_percentage for e in terminated) / len(terminated)
            if terminated else 0.0
        )

        return {
            "total_traces": len(estimates),
            "traces_terminated": len(terminated),
            "total_tokens_saved": total_saved,
            "total_tokens_original": total_original,
            "overall_savings_percentage": round(overall_pct, 4),
            "avg_savings_per_terminated": round(avg_per_terminated, 4),
        }

    def _trace_tokens(self, trace: AgentTrace) -> int:
        """Total tokens for a trace, using per-step data or the average.

        Args:
            trace: Agent trace to measure.

        Returns:
            Total token count.
        """
        actual = sum(
            s.prompt_tokens + s.completion_tokens for s in trace.steps
        )
        if actual > 0:
            return actual
        return int(self._avg_tokens_per_step * len(trace.steps))

    def _prefix_tokens(self, trace: AgentTrace, k: int) -> int:
        """Tokens consumed by the first *k* steps.

        Args:
            trace: Agent trace.
            k: Number of prefix steps.

        Returns:
            Token count for steps[:k].
        """
        prefix = trace.steps[:k]
        actual = sum(s.prompt_tokens + s.completion_tokens for s in prefix)
        if actual > 0:
            return actual
        return int(self._avg_tokens_per_step * len(prefix))
