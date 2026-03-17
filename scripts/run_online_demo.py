#!/usr/bin/env python3
"""
Online Detection Demo - Demonstrate Phase 3A failure prediction and savings

Purpose: End-to-end demonstration of the online detection pipeline.
Loads (or builds) a pattern library, runs the FailurePredictor on test
traces, prints per-trace decisions for interesting examples, and reports
token savings from early termination.

Usage:
    # With pre-built library:
    python scripts/run_online_demo.py --library-path data/pipeline_test/pattern_library.json

    # With synthetic traces (builds library on the fly):
    python scripts/run_online_demo.py --mode synthetic --n-traces 100

    # With real traces:
    python scripts/run_online_demo.py --mode real --trace-dir data/raw_traces

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_collection.trace_schema import AgentTrace, TaskOutcome
from src.detection.failure_predictor import Decision, FailurePredictor, PredictionResult
from src.detection.pattern_matcher import PatternMatcher
from src.detection.token_calculator import SavingsEstimate, TokenCalculator
from src.evaluation.data_split import DataSplitter
from src.mining.signature_library import SignatureLibrary
from src.preprocessing.k_prefix import PrefixEntry, batch_extract_prefixes
from src.preprocessing.symbolizer import TraceSymbolizer
from src.utils.symbol_descriptions import describe_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================================
# Data loading
# =========================================================================

def load_traces(args: argparse.Namespace) -> list[AgentTrace]:
    """Load traces based on CLI arguments.

    Args:
        args: Parsed CLI arguments with mode, trace_dir, n_traces.

    Returns:
        List of agent traces.
    """
    if args.mode == "synthetic":
        from scripts.run_offline_pipeline import generate_synthetic_traces
        print(f"Generating {args.n_traces} synthetic traces...")
        return generate_synthetic_traces(args.n_traces)

    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"ERROR: Trace directory not found: {trace_dir}")
        sys.exit(1)

    from src.data_collection.trace_logger import TraceLogger
    tl = TraceLogger(base_dir=str(trace_dir.parent))
    traces = list(tl.iter_traces(directory=trace_dir))
    if not traces:
        print(f"ERROR: No traces found in {trace_dir}")
        sys.exit(1)

    return traces


def build_library(
    train_entries: list[PrefixEntry],
    spmf_jar: Path,
) -> SignatureLibrary:
    """Build a pattern library from training entries via the BIDE pipeline.

    Args:
        train_entries: Training prefix entries.
        spmf_jar: Path to the SPMF jar.

    Returns:
        Built SignatureLibrary.
    """
    from src.baselines.bide_coverage import BIDECoverageBaseline

    print("  Fitting BIDE Coverage on training data...")
    bide = BIDECoverageBaseline(spmf_jar_path=str(spmf_jar))
    bide.fit(train_entries)
    print(f"  Library: {len(bide.patterns_)} patterns")
    return SignatureLibrary(patterns=bide.patterns_)


# =========================================================================
# Interesting example selection
# =========================================================================

def select_interesting_examples(
    traces: list[AgentTrace],
    predictor: FailurePredictor,
    symbolizer: TraceSymbolizer,
    k: int,
) -> list[tuple[AgentTrace, PredictionResult]]:
    """Pick up to 5 interesting examples: 2 TERMINATE, 2 CONTINUE, 1 ALERT.

    Args:
        traces: Test traces.
        predictor: Fitted FailurePredictor.
        symbolizer: TraceSymbolizer.
        k: Prefix length.

    Returns:
        List of (trace, prediction) tuples, up to 5.
    """
    buckets: dict[Decision, list[tuple[AgentTrace, PredictionResult]]] = {
        Decision.TERMINATE: [],
        Decision.ALERT: [],
        Decision.CONTINUE: [],
    }

    for trace in traces:
        result = predictor.predict_trace(trace, k, symbolizer)
        buckets[result.decision].append((trace, result))

    for decision in buckets:
        buckets[decision].sort(key=lambda x: x[1].confidence, reverse=True)

    selected: list[tuple[AgentTrace, PredictionResult]] = []
    for item in buckets[Decision.TERMINATE][:2]:
        selected.append(item)
    for item in buckets[Decision.CONTINUE][:2]:
        selected.append(item)
    for item in buckets[Decision.ALERT][:1]:
        selected.append(item)

    return selected


def print_example(
    trace: AgentTrace,
    result: PredictionResult,
    symbolizer: TraceSymbolizer,
    k: int,
) -> None:
    """Print a detailed view of a single prediction.

    Args:
        trace: The agent trace.
        result: The prediction result.
        symbolizer: TraceSymbolizer for displaying the prefix.
        k: Prefix length.
    """
    meta = trace.metadata
    symbols = symbolizer.symbolize_prefix(trace, k)

    print(f"  Trace ID:    {meta.trace_id}")
    print(f"  Task:        {meta.task_description}")
    print(f"  Outcome:     {meta.outcome.value}")
    print(f"  Total steps: {len(trace.steps)}")
    print(f"  Decision:    {result.decision.value.upper()}")
    print(f"  Confidence:  {result.confidence:.4f}")
    print(f"  Matches:     {len(result.matching_patterns)}")
    print()

    print(f"  Prefix (K={len(symbols)}):")
    for i, sym in enumerate(symbols):
        marker = ""
        matched_positions = set()
        for m in result.matching_patterns[:3]:
            matched_positions.update(m.positions)
        if i in matched_positions:
            marker = "  <-- matched"
        print(f"    [{i:2d}] {sym}{marker}")
    print()

    if result.matching_patterns:
        top = result.matching_patterns[0]
        descs = [describe_symbol(s) for s in top.pattern.symbols]
        print(f"  Top pattern: {' -> '.join(descs)}")
        print(f"    Score={top.score:.4f}, Precision={top.pattern.precision:.4f}")
    print()


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Online detection demo: predict failures and estimate savings",
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Trace source: synthetic or real (default: synthetic)",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=100,
        help="Number of synthetic traces (synthetic mode only, default: 100)",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="data/raw_traces",
        help="Trace directory (real mode, default: data/raw_traces)",
    )
    parser.add_argument(
        "--library-path",
        type=str,
        default=None,
        help="Path to pre-built pattern library JSON (skips fitting)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="K-prefix length (default: 10)",
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
        "--terminate-threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for TERMINATE (default: 0.85)",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.70,
        help="Confidence threshold for ALERT (default: 0.70)",
    )
    return parser.parse_args()


# =========================================================================
# Main
# =========================================================================

def main() -> int:
    """Run the online detection demo end-to-end.

    Returns:
        Exit code (0 success, 1 error).
    """
    args = parse_args()
    symbolizer = TraceSymbolizer(abstraction_level=1)

    # ------------------------------------------------------------------
    # 1. Load traces
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 3A ONLINE DETECTION DEMO")
    print("=" * 70)
    print()

    traces = load_traces(args)
    failures = sum(1 for t in traces if t.metadata.outcome.is_failure)
    successes = len(traces) - failures
    print(f"Loaded {len(traces)} traces ({failures} failures, {successes} successes)")
    print()

    # ------------------------------------------------------------------
    # 2. Split data
    # ------------------------------------------------------------------
    print("[1/4] Splitting data...")
    dataset = batch_extract_prefixes(traces, k=args.k, symbolizer=symbolizer)
    splitter = DataSplitter()
    split = splitter.split(dataset.entries, seed=args.seed)
    print(f"  Split: {split.summary()}")
    print()

    # Build trace lookup for later
    traces_by_id: dict[str, AgentTrace] = {
        t.metadata.trace_id: t for t in traces
    }
    test_traces = [
        traces_by_id[e.trace_id]
        for e in split.test
        if e.trace_id in traces_by_id
    ]

    # ------------------------------------------------------------------
    # 3. Load or build library
    # ------------------------------------------------------------------
    print("[2/4] Preparing pattern library...")
    if args.library_path:
        lib_path = Path(args.library_path)
        if not lib_path.exists():
            print(f"  ERROR: Library not found: {lib_path}")
            return 1
        library = SignatureLibrary.load(lib_path)
        print(f"  Loaded {len(library)} patterns from {lib_path}")
    else:
        library = build_library(split.train, spmf_jar=Path(args.spmf_jar))

    if len(library) == 0:
        print("  WARNING: Pattern library is empty — predictions will all be CONTINUE")
    print()

    # ------------------------------------------------------------------
    # 4. Run predictor on interesting examples
    # ------------------------------------------------------------------
    print("[3/4] Running failure predictor on test traces...")
    thresholds = {
        "terminate": args.terminate_threshold,
        "alert": args.alert_threshold,
    }
    matcher = PatternMatcher(library)
    predictor = FailurePredictor(matcher, thresholds=thresholds)

    examples = select_interesting_examples(
        test_traces, predictor, symbolizer, args.k,
    )

    if not examples:
        print("  WARNING: No test traces available for examples")
    else:
        print(f"  Selected {len(examples)} interesting example(s):")
        print()

        for i, (trace, result) in enumerate(examples, start=1):
            print(f"  --- Example {i}/{len(examples)} ({result.decision.value.upper()}) ---")
            print_example(trace, result, symbolizer, args.k)
            print("-" * 70)
            print()
    print()

    # ------------------------------------------------------------------
    # 5. Token savings
    # ------------------------------------------------------------------
    print("[4/4] Estimating token savings...")
    calculator = TokenCalculator.from_corpus(traces)
    estimates = calculator.batch_estimate(
        test_traces, predictor, symbolizer, args.k,
    )
    stats = TokenCalculator.summary(estimates)

    print()
    print("  SAVINGS SUMMARY")
    print("  " + "-" * 40)
    print(f"  Total test traces:         {stats['total_traces']}")
    print(f"  Traces terminated early:   {stats['traces_terminated']}")
    print(f"  Total tokens (original):   {stats['total_tokens_original']:,}")
    print(f"  Total tokens saved:        {stats['total_tokens_saved']:,}")
    print(f"  Overall savings:           {stats['overall_savings_percentage']:.2%}")
    print(f"  Avg savings/terminated:    {stats['avg_savings_per_terminated']:.2%}")
    print()

    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
