#!/usr/bin/env python3
"""
Per-Model Analysis - Compare failure-detection performance across LLM agents

Purpose: Load traces from multiple models, report per-model corpus statistics
(Analysis B), then train a shared BIDE pattern library on all models and
evaluate per-model test subsets (Analysis A).  Produces formatted comparison
tables and saves results as JSON.

Usage:
    python scripts/per_model_analysis.py --trace-dir data/medium_traces
    python scripts/per_model_analysis.py --trace-dir data/medium_traces --exclude-errors --exclude-timeouts
    python scripts/per_model_analysis.py --k-values 3,5,8,10 --output-dir data/per_model_results

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.baselines.bide_coverage import BIDECoverageBaseline
from src.data_collection.trace_logger import TraceLogger
from src.data_collection.trace_schema import AgentTrace, TaskOutcome
from src.evaluation.data_split import DataSplitter
from src.evaluation.metrics import MetricsCalculator, find_optimal_threshold
from src.preprocessing.k_prefix import PrefixEntry, batch_extract_prefixes
from src.preprocessing.symbolizer import TraceSymbolizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
# Analysis B — Per-model corpus stats
# =========================================================================

def compute_corpus_stats(
    traces: list[AgentTrace],
) -> dict[str, dict[str, object]]:
    """Compute per-model corpus statistics.

    Args:
        traces: All loaded traces.

    Returns:
        Mapping from model name to stats dict.
    """
    by_model: dict[str, list[AgentTrace]] = defaultdict(list)
    for t in traces:
        by_model[t.metadata.model].append(t)

    stats: dict[str, dict[str, object]] = {}
    for model in sorted(by_model):
        model_traces = by_model[model]
        n_success = sum(
            1 for t in model_traces
            if t.metadata.outcome == TaskOutcome.SUCCESS
        )
        n_failure = sum(
            1 for t in model_traces
            if t.metadata.outcome == TaskOutcome.FAILURE
        )
        n_timeout = sum(
            1 for t in model_traces
            if t.metadata.outcome == TaskOutcome.TIMEOUT
        )
        n_error = sum(
            1 for t in model_traces
            if t.metadata.outcome == TaskOutcome.ERROR
        )

        success_steps = [
            t.total_steps for t in model_traces
            if t.metadata.outcome == TaskOutcome.SUCCESS
        ]
        failure_steps = [
            t.total_steps for t in model_traces
            if t.metadata.outcome != TaskOutcome.SUCCESS
        ]

        stats[model] = {
            "n_traces": len(model_traces),
            "n_success": n_success,
            "n_failure": n_failure,
            "n_timeout": n_timeout,
            "n_error": n_error,
            "avg_success_steps": (
                round(sum(success_steps) / len(success_steps), 2)
                if success_steps else 0.0
            ),
            "avg_failure_steps": (
                round(sum(failure_steps) / len(failure_steps), 2)
                if failure_steps else 0.0
            ),
        }

    return stats


def print_corpus_stats(stats: dict[str, dict[str, object]]) -> None:
    """Print a formatted corpus-stats table.

    Args:
        stats: Output of compute_corpus_stats.
    """
    header = (
        f"{'Model':<20s} | {'N':>5s} | {'Succ':>5s} | {'Fail':>5s} | "
        f"{'Tout':>5s} | {'Err':>5s} | {'AvgS_s':>7s} | {'AvgS_f':>7s}"
    )
    sep = "-" * len(header)

    print("ANALYSIS B — Per-Model Corpus Statistics")
    print(sep)
    print(header)
    print(sep)

    for model in sorted(stats):
        s = stats[model]
        print(
            f"{model:<20s} | {s['n_traces']:5d} | {s['n_success']:5d} | "
            f"{s['n_failure']:5d} | {s['n_timeout']:5d} | {s['n_error']:5d} | "
            f"{s['avg_success_steps']:7.2f} | {s['avg_failure_steps']:7.2f}"
        )

    total_n = sum(s["n_traces"] for s in stats.values())
    total_succ = sum(s["n_success"] for s in stats.values())
    total_fail = sum(s["n_failure"] for s in stats.values())
    total_tout = sum(s["n_timeout"] for s in stats.values())
    total_err = sum(s["n_error"] for s in stats.values())
    print(sep)
    print(
        f"{'ALL':<20s} | {total_n:5d} | {total_succ:5d} | "
        f"{total_fail:5d} | {total_tout:5d} | {total_err:5d} |         |"
    )
    print(sep)


# =========================================================================
# Analysis A — Per-model evaluation helpers
# =========================================================================

def evaluate_subset(
    entries: list[PrefixEntry],
    bide: BIDECoverageBaseline,
    k: int,
    threshold: float,
    mc: MetricsCalculator,
) -> dict[str, float]:
    """Score a subset of entries and return binary metrics.

    Args:
        entries: PrefixEntry objects to evaluate.
        bide: Fitted BIDECoverageBaseline.
        k: Prefix length.
        threshold: Decision threshold (tuned on val set).
        mc: MetricsCalculator instance.

    Returns:
        Dict with precision, recall, f1, auc_pr, etc.
    """
    y_true = [1 if e.outcome.is_failure else 0 for e in entries]
    y_scores = [bide.predict_at_k(e.symbols, k) for e in entries]
    return mc.compute_binary_metrics(y_true, y_scores, threshold=threshold)


def print_analysis_a_table(
    k: int,
    model_metrics: dict[str, dict[str, float]],
    model_counts: dict[str, int],
) -> None:
    """Print a formatted comparison table for a single K value.

    Args:
        k: Prefix length.
        model_metrics: Mapping from model name (or "ALL") to metrics dict.
        model_counts: Mapping from model name (or "ALL") to test-set size.
    """
    header = (
        f"{'Model':<20s} | {'N_test':>6s} | {'F1':>6s} | "
        f"{'AUC-PR':>7s} | {'Prec':>6s} | {'Recall':>6s}"
    )
    sep = "-" * len(header)

    print(f"K={k}")
    print(sep)
    print(header)
    print(sep)

    display_order = sorted(
        k for k in model_metrics if k != "ALL (combined)"
    ) + (["ALL (combined)"] if "ALL (combined)" in model_metrics else [])

    for model in display_order:
        m = model_metrics[model]
        n = model_counts[model]

        def _fmt(v: float) -> str:
            return f"{v:.3f}" if not math.isnan(v) else "   NaN"

        print(
            f"{model:<20s} | {n:6d} | {_fmt(m['f1']):>6s} | "
            f"{_fmt(m['auc_pr']):>7s} | {_fmt(m['precision']):>6s} | "
            f"{_fmt(m['recall']):>6s}"
        )

    print(sep)


# =========================================================================
# JSON serialization
# =========================================================================

def _sanitize(obj: object) -> object:
    """Make a value JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Per-model failure-detection analysis",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="data/medium_traces",
        help="Directory containing trace JSON files (default: data/medium_traces)",
    )
    parser.add_argument(
        "--spmf-jar",
        type=str,
        default="lib/spmf.jar",
        help="Path to SPMF jar (default: lib/spmf.jar)",
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
        help="Random seed for data splitting (default: 42)",
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
        "--abstraction-level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Symbolization level: 0=fine, 1=medium, 2=coarse (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/per_model_results",
        help="Directory for output JSON (default: data/per_model_results)",
    )
    return parser.parse_args()


# =========================================================================
# Main
# =========================================================================

def main() -> int:
    """Run the per-model analysis end-to-end.

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
    print("PER-MODEL ANALYSIS")
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
    # 2. Build trace_id -> model mapping
    # ------------------------------------------------------------------
    model_by_trace_id: dict[str, str] = {
        t.metadata.trace_id: t.metadata.model for t in traces
    }

    # ------------------------------------------------------------------
    # 3. Analysis B — Per-model corpus stats
    # ------------------------------------------------------------------
    print()
    corpus_stats = compute_corpus_stats(traces)
    print_corpus_stats(corpus_stats)
    print()

    # ------------------------------------------------------------------
    # 4. Analysis A — Build dataset, split, fit BIDE
    # ------------------------------------------------------------------
    print("ANALYSIS A — Per-Model Evaluation (train all, test per-model)")
    print("=" * 70)
    print()

    symbolizer = TraceSymbolizer(abstraction_level=args.abstraction_level)
    max_k = max(k_values)

    print("[1/4] Building prefix dataset and splitting...")
    dataset = batch_extract_prefixes(traces, k=max_k, symbolizer=symbolizer)
    splitter = DataSplitter()
    split = splitter.split(dataset.entries, seed=args.seed)
    print(f"  Split: {split.summary()}")
    print()

    print("[2/4] Fitting BIDE Coverage on full training set...")
    bide = BIDECoverageBaseline(spmf_jar_path=str(spmf_jar))
    bide.fit(split.train)
    print(f"  Mined {len(bide.patterns_)} patterns")
    print()

    if not bide.patterns_:
        print("ERROR: No patterns mined — cannot evaluate")
        return 1

    # ------------------------------------------------------------------
    # 5. Tune thresholds on full validation set
    # ------------------------------------------------------------------
    print("[3/4] Tuning thresholds on validation set...")
    thresholds: dict[int, float] = {}
    for k in k_values:
        y_true_val = [1 if e.outcome.is_failure else 0 for e in split.val]
        y_scores_val = [bide.predict_at_k(e.symbols, k) for e in split.val]
        thresholds[k] = find_optimal_threshold(y_true_val, y_scores_val)

    print(f"  Thresholds: {{{', '.join(f'{k}: {t:.3f}' for k, t in sorted(thresholds.items()))}}}")
    print()

    # ------------------------------------------------------------------
    # 6. Partition test entries by model and evaluate
    # ------------------------------------------------------------------
    print("[4/4] Evaluating per-model test subsets...")
    print()

    model_test_entries: dict[str, list[PrefixEntry]] = defaultdict(list)
    for entry in split.test:
        model = model_by_trace_id.get(entry.trace_id, "unknown")
        model_test_entries[model].append(entry)

    model_names = sorted(model_test_entries.keys())
    for model in model_names:
        n = len(model_test_entries[model])
        n_fail = sum(1 for e in model_test_entries[model] if e.outcome.is_failure)
        print(f"  {model}: {n} test entries ({n_fail} failures)")
    print()

    mc = MetricsCalculator()
    analysis_a_results: dict[str, dict[str, dict[str, float]]] = {}

    for k in k_values:
        model_metrics: dict[str, dict[str, float]] = {}
        model_counts: dict[str, int] = {}

        for model in model_names:
            entries = model_test_entries[model]
            model_metrics[model] = evaluate_subset(entries, bide, k, thresholds[k], mc)
            model_counts[model] = len(entries)

        model_metrics["ALL (combined)"] = evaluate_subset(
            split.test, bide, k, thresholds[k], mc,
        )
        model_counts["ALL (combined)"] = len(split.test)

        analysis_a_results[str(k)] = model_metrics
        print_analysis_a_table(k, model_metrics, model_counts)
        print()

    # ------------------------------------------------------------------
    # 7. Save results as JSON
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "per_model_analysis.json"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "trace_dir": str(trace_dir),
            "spmf_jar": str(spmf_jar),
            "k_values": k_values,
            "seed": args.seed,
            "exclude_errors": args.exclude_errors,
            "exclude_timeouts": args.exclude_timeouts,
            "abstraction_level": args.abstraction_level,
        },
        "corpus_stats": _sanitize(corpus_stats),
        "analysis_a": {
            "split_info": split.summary(),
            "n_patterns": len(bide.patterns_),
            "thresholds": {str(k): round(t, 4) for k, t in thresholds.items()},
            "per_k": _sanitize(analysis_a_results),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Results saved to {output_path}")
    print()
    print("=" * 70)
    print("PER-MODEL ANALYSIS COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
