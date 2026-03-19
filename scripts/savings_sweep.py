#!/usr/bin/env python3
"""
Savings Sweep - Threshold sweep for online detection savings analysis

Purpose: Sweep termination-confidence thresholds across the BIDE Coverage
score range (typically 0.0–0.4) for multiple K-prefix lengths.  Reports
precision, recall, success-kill rate, and token savings per (K, threshold)
pair to identify the optimal operating point for the thesis.

Usage:
    python scripts/savings_sweep.py --trace-dir data/medium_traces
    python scripts/savings_sweep.py --trace-dir data/medium_traces --exclude-errors --exclude-timeouts
    python scripts/savings_sweep.py --k-values 3,5,8,10 --output-dir data/savings_results

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.baselines.bide_coverage import BIDECoverageBaseline
from src.data_collection.trace_logger import TraceLogger
from src.data_collection.trace_schema import AgentTrace, TaskOutcome
from src.detection.pattern_matcher import PatternMatcher
from src.detection.token_calculator import TokenCalculator
from src.evaluation.data_split import DataSplitter
from src.preprocessing.k_prefix import batch_extract_prefixes
from src.preprocessing.symbolizer import TraceSymbolizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [
    0.01, 0.02, 0.03, 0.05, 0.08, 0.10,
    0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
]


# =========================================================================
# Data loading
# =========================================================================

def load_traces(trace_dir: Path) -> list[AgentTrace]:
    """Load traces from a directory via TraceLogger.

    Args:
        trace_dir: Directory containing trace JSON files.

    Returns:
        List of AgentTrace objects.
    """
    tl = TraceLogger(base_dir=str(trace_dir.parent))
    return list(tl.iter_traces(directory=trace_dir))


# =========================================================================
# Sweep logic
# =========================================================================

def sweep_threshold(
    test_traces: list[AgentTrace],
    matcher: PatternMatcher,
    symbolizer: TraceSymbolizer,
    calculator: TokenCalculator,
    k: int,
    threshold: float,
) -> dict:
    """Evaluate a single (K, threshold) operating point.

    Args:
        test_traces: Test-split traces with ground-truth outcomes.
        matcher: PatternMatcher initialised with the BIDE library.
        symbolizer: TraceSymbolizer at the desired abstraction level.
        calculator: TokenCalculator fitted on the corpus.
        k: Prefix length.
        threshold: Confidence value at or above which a trace is terminated.

    Returns:
        Dict of metrics for this operating point.
    """
    total_failures = sum(1 for t in test_traces if t.metadata.outcome.is_failure)
    total_successes = len(test_traces) - total_failures

    tp = 0
    fp = 0
    n_terminated = 0
    total_tokens_saved = 0
    total_tokens_original = 0
    terminated_savings_pcts: list[float] = []

    for trace in test_traces:
        symbols = symbolizer.symbolize_trace(trace)[:k]
        matches = matcher.match(symbols)
        confidence = sum(m.score for m in matches) / matcher.total_score

        est_full = calculator.estimate_savings(trace, terminate_at=len(trace.steps))
        total_tokens_original += est_full.actual_tokens

        if confidence >= threshold:
            n_terminated += 1
            is_failure = trace.metadata.outcome.is_failure
            if is_failure:
                tp += 1
            else:
                fp += 1

            est = calculator.estimate_savings(trace, terminate_at=k)
            total_tokens_saved += est.saved_tokens
            terminated_savings_pcts.append(est.savings_percentage)

    precision = tp / n_terminated if n_terminated > 0 else 0.0
    recall = tp / total_failures if total_failures > 0 else 0.0
    kill_rate = fp / total_successes if total_successes > 0 else 0.0
    overall_savings = (
        total_tokens_saved / total_tokens_original
        if total_tokens_original > 0 else 0.0
    )
    avg_savings = (
        sum(terminated_savings_pcts) / len(terminated_savings_pcts)
        if terminated_savings_pcts else 0.0
    )

    return {
        "k": k,
        "threshold": threshold,
        "n_terminated": n_terminated,
        "true_positives": tp,
        "false_positives": fp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "success_kill_rate": round(kill_rate, 4),
        "total_tokens_saved": total_tokens_saved,
        "overall_savings_pct": round(overall_savings, 4),
        "avg_savings_per_terminated": round(avg_savings, 4),
    }


# =========================================================================
# Display
# =========================================================================

_TABLE_HEADER = (
    f"{'Thresh':>7s}  {'Term':>5s}  {'TP':>4s}  {'FP':>4s}  "
    f"{'Prec':>6s}  {'Recall':>6s}  {'Kill%':>6s}  "
    f"{'TokSaved':>10s}  {'Sav%':>6s}  {'AvgSav':>6s}"
)
_TABLE_SEP = "-" * len(_TABLE_HEADER)


def print_table(k: int, rows: list[dict]) -> None:
    """Print a formatted results table for a given K value.

    Args:
        k: Prefix length.
        rows: List of metric dicts from sweep_threshold.
    """
    print(f"\n  K = {k}")
    print(f"  {_TABLE_SEP}")
    print(f"  {_TABLE_HEADER}")
    print(f"  {_TABLE_SEP}")

    for r in rows:
        print(
            f"  {r['threshold']:7.2f}  {r['n_terminated']:5d}  "
            f"{r['true_positives']:4d}  {r['false_positives']:4d}  "
            f"{r['precision']:6.3f}  {r['recall']:6.3f}  "
            f"{r['success_kill_rate']:6.3f}  "
            f"{r['total_tokens_saved']:10,d}  "
            f"{r['overall_savings_pct']:6.2%}  "
            f"{r['avg_savings_per_terminated']:6.2%}"
        )

    print(f"  {_TABLE_SEP}")


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Threshold sweep for online detection savings analysis",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="data/medium_traces",
        help="Directory containing trace JSON files (default: data/medium_traces)",
    )
    parser.add_argument(
        "--exclude-errors",
        action="store_true",
        default=False,
        help="Remove traces with outcome 'error' before analysis",
    )
    parser.add_argument(
        "--exclude-timeouts",
        action="store_true",
        default=False,
        help="Remove traces with outcome 'timeout' before analysis",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="3,5,8,10",
        help="Comma-separated prefix lengths to evaluate (default: 3,5,8,10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )
    parser.add_argument(
        "--spmf-jar",
        type=str,
        default="lib/spmf.jar",
        help="Path to SPMF jar (default: lib/spmf.jar)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/savings_results",
        help="Directory for output JSON (default: data/savings_results)",
    )
    parser.add_argument(
        "--abstraction-level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Symbolization level: 0=fine, 1=medium, 2=coarse (default: 1)",
    )
    return parser.parse_args()


# =========================================================================
# Main
# =========================================================================

def main() -> int:
    """Run the threshold sweep end-to-end.

    Returns:
        Exit code (0 success, 1 error).
    """
    args = parse_args()

    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    spmf_jar = Path(args.spmf_jar)
    k_values = [int(v.strip()) for v in args.k_values.split(",")]

    # ------------------------------------------------------------------
    # 1. Load traces
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SAVINGS SWEEP — Threshold Analysis")
    print("=" * 70)
    print()

    if not trace_dir.exists():
        print(f"ERROR: Trace directory not found: {trace_dir}")
        return 1

    print(f"Loading traces from {trace_dir}...")
    traces = load_traces(trace_dir)

    if not traces:
        print("ERROR: No traces loaded")
        return 1

    failures = sum(1 for t in traces if t.metadata.outcome.is_failure)
    successes = len(traces) - failures
    print(f"Loaded {len(traces)} traces ({failures} failures, {successes} successes)")
    print()

    if args.exclude_errors:
        before = len(traces)
        traces = [t for t in traces if t.metadata.outcome != TaskOutcome.ERROR]
        print(f"  Excluded {before - len(traces)} error traces ({before} -> {len(traces)})")

    if args.exclude_timeouts:
        before = len(traces)
        traces = [t for t in traces if t.metadata.outcome != TaskOutcome.TIMEOUT]
        print(f"  Excluded {before - len(traces)} timeout traces ({before} -> {len(traces)})")

    if args.exclude_errors or args.exclude_timeouts:
        failures = sum(1 for t in traces if t.metadata.outcome.is_failure)
        successes = len(traces) - failures
        print(f"  After filtering: {len(traces)} traces ({failures} failures, {successes} successes)")
        print()

    # ------------------------------------------------------------------
    # 2. Symbolize, split, fit
    # ------------------------------------------------------------------
    symbolizer = TraceSymbolizer(abstraction_level=args.abstraction_level)
    max_k = max(k_values)

    print("[1/3] Building prefix dataset and splitting...")
    dataset = batch_extract_prefixes(traces, k=max_k, symbolizer=symbolizer)
    splitter = DataSplitter()
    split = splitter.split(dataset.entries, seed=args.seed)
    print(f"  Split: {split.summary()}")
    print()

    print("[2/3] Fitting BIDE Coverage on training data...")
    bide = BIDECoverageBaseline(spmf_jar_path=str(spmf_jar))
    bide.fit(split.train)
    print(f"  Mined {len(bide.patterns_)} patterns")
    print()

    if not bide.patterns_:
        print("ERROR: No patterns mined — cannot run sweep")
        return 1

    matcher = PatternMatcher(bide.patterns_)
    calculator = TokenCalculator.from_corpus(traces)

    traces_by_id: dict[str, AgentTrace] = {
        t.metadata.trace_id: t for t in traces
    }
    test_traces = [
        traces_by_id[e.trace_id]
        for e in split.test
        if e.trace_id in traces_by_id
    ]
    test_failures = sum(1 for t in test_traces if t.metadata.outcome.is_failure)
    test_successes = len(test_traces) - test_failures
    print(f"  Test set: {len(test_traces)} traces ({test_failures} failures, {test_successes} successes)")
    print()

    # ------------------------------------------------------------------
    # 3. Sweep
    # ------------------------------------------------------------------
    print("[3/3] Sweeping thresholds...")
    all_results: dict[str, list[dict]] = {}

    for k in k_values:
        rows: list[dict] = []
        for threshold in DEFAULT_THRESHOLDS:
            row = sweep_threshold(
                test_traces, matcher, symbolizer, calculator, k, threshold,
            )
            rows.append(row)
        all_results[str(k)] = rows
        print_table(k, rows)

    print()

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "savings_sweep.json"

    output_payload = {
        "generated_at": datetime.now().isoformat(),
        "trace_dir": str(trace_dir),
        "exclude_errors": args.exclude_errors,
        "exclude_timeouts": args.exclude_timeouts,
        "abstraction_level": args.abstraction_level,
        "seed": args.seed,
        "k_values": k_values,
        "thresholds": DEFAULT_THRESHOLDS,
        "n_traces_total": len(traces),
        "n_test_traces": len(test_traces),
        "n_test_failures": test_failures,
        "n_test_successes": test_successes,
        "n_patterns": len(bide.patterns_),
        "matcher_total_score": round(matcher.total_score, 6),
        "results_by_k": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Results saved to {output_path}")
    print()
    print("=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
