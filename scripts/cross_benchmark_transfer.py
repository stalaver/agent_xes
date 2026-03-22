#!/usr/bin/env python3
"""
Cross-Benchmark Transfer Analysis

Tests whether failure patterns mined from MiniWoB traces (via BIDE+)
transfer to WebArena traces.  Loads an existing SignatureLibrary, applies
it to WebArena and MiniWoB traces, and produces a detailed coverage-score
report with observation-distribution statistics.

Usage:
    python scripts/cross_benchmark_transfer.py \
        --pattern-library data/pattern_library_medium/pattern_library.json \
        --webarena-traces data/raw_traces/webarena/llama-3.2-3b/ \
        --miniwob-traces data/medium_traces/ \
        --exclude-errors

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_collection.trace_logger import TraceLogger
from src.data_collection.trace_schema import AgentTrace, TaskOutcome
from src.mining.pattern_ranker import _is_subsequence
from src.mining.signature_library import SignatureLibrary
from src.preprocessing.symbolizer import TraceSymbolizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

K_VALUES = [3, 5, 8, 10]


# =========================================================================
# Trace loading (mirrors run_experiment.py pattern)
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


def filter_traces(
    traces: list[AgentTrace],
    exclude_errors: bool,
    exclude_timeouts: bool,
) -> list[AgentTrace]:
    """Remove error and/or timeout traces.

    Args:
        traces: Raw trace list.
        exclude_errors: Drop traces with outcome ERROR.
        exclude_timeouts: Drop traces with outcome TIMEOUT.

    Returns:
        Filtered trace list.
    """
    if exclude_errors:
        before = len(traces)
        traces = [t for t in traces if t.metadata.outcome != TaskOutcome.ERROR]
        print(f"   Excluded {before - len(traces)} error traces ({before} -> {len(traces)})")

    if exclude_timeouts:
        before = len(traces)
        traces = [t for t in traces if t.metadata.outcome != TaskOutcome.TIMEOUT]
        print(f"   Excluded {before - len(traces)} timeout traces ({before} -> {len(traces)})")

    return traces


# =========================================================================
# Coverage scoring (mirrors BIDECoverageBaseline.predict)
# =========================================================================

def compute_coverage_score(
    library: SignatureLibrary,
    symbols: list[str],
    total_score: float,
) -> float:
    """Fraction of library pattern weight that matches as subsequences.

    Args:
        library: Loaded SignatureLibrary.
        symbols: Symbolized trace prefix.
        total_score: Pre-computed sum of all pattern scores.

    Returns:
        Coverage score in [0.0, 1.0].
    """
    if total_score <= 0.0 or not symbols:
        return 0.0
    matched = library.match(symbols)
    matched_score = sum(p.score for p in matched)
    return min(matched_score / total_score, 1.0)


def score_group(scores: list[float]) -> dict:
    """Compute descriptive statistics for a list of scores.

    Args:
        scores: Coverage scores for a group of traces.

    Returns:
        Dict with count, mean, median, std.
    """
    if not scores:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "count": len(scores),
        "mean": round(statistics.mean(scores), 6),
        "median": round(statistics.median(scores), 6),
        "std": round(statistics.stdev(scores), 6) if len(scores) > 1 else 0.0,
    }


# =========================================================================
# Observation distribution
# =========================================================================

def compute_observation_distribution(
    traces: list[AgentTrace],
    symbolizer: TraceSymbolizer,
) -> dict:
    """Classify every symbolized step as SUCCESS / FAIL / UNKNOWN / R_STUCK.

    Args:
        traces: Traces to analyze.
        symbolizer: Shared symbolizer instance.

    Returns:
        Dict with total_steps, success_pct, fail_pct, unknown_pct, r_stuck_pct.
    """
    total = 0
    success_count = 0
    fail_count = 0
    unknown_count = 0
    r_stuck_count = 0

    for trace in traces:
        for step in trace.steps:
            sym = symbolizer.symbolize_step(step)
            total += 1
            if "_SUCCESS" in sym:
                success_count += 1
            elif "_FAIL" in sym:
                fail_count += 1
            if sym.startswith("UNKNOWN_"):
                unknown_count += 1
            if "R_STUCK" in sym:
                r_stuck_count += 1

    if total == 0:
        return {
            "total_steps": 0,
            "success_pct": 0.0,
            "fail_pct": 0.0,
            "unknown_pct": 0.0,
            "r_stuck_pct": 0.0,
        }

    return {
        "total_steps": total,
        "success_pct": round(100.0 * success_count / total, 2),
        "fail_pct": round(100.0 * fail_count / total, 2),
        "unknown_pct": round(100.0 * unknown_count / total, 2),
        "r_stuck_pct": round(100.0 * r_stuck_count / total, 2),
    }


# =========================================================================
# Transfer analysis core
# =========================================================================

def run_transfer_analysis(
    library: SignatureLibrary,
    webarena_traces: list[AgentTrace],
    miniwob_traces: list[AgentTrace],
    symbolizer: TraceSymbolizer,
    k_values: list[int],
) -> dict:
    """Run the full cross-benchmark transfer analysis.

    Args:
        library: Pre-trained MiniWoB pattern library.
        webarena_traces: WebArena traces to test against.
        miniwob_traces: MiniWoB traces for comparison baseline.
        symbolizer: Shared TraceSymbolizer instance.
        k_values: Prefix lengths to evaluate.

    Returns:
        Full results dictionary.
    """
    total_score = sum(p.score for p in library.patterns)

    # Symbolize all traces (full length for max K)
    max_k = max(k_values)

    wa_symbols: list[tuple[TaskOutcome, list[str]]] = []
    for t in webarena_traces:
        syms = symbolizer.symbolize_prefix(t, max_k)
        wa_symbols.append((t.metadata.outcome, syms))

    mw_symbols: list[tuple[TaskOutcome, list[str]]] = []
    for t in miniwob_traces:
        syms = symbolizer.symbolize_prefix(t, max_k)
        mw_symbols.append((t.metadata.outcome, syms))

    # Per-K analysis
    per_k: dict[str, dict] = {}
    for k in k_values:
        wa_failure_scores: list[float] = []
        wa_timeout_scores: list[float] = []
        wa_success_scores: list[float] = []

        for outcome, syms in wa_symbols:
            prefix = syms[:k]
            sc = compute_coverage_score(library, prefix, total_score)
            if outcome == TaskOutcome.FAILURE:
                wa_failure_scores.append(sc)
            elif outcome == TaskOutcome.TIMEOUT:
                wa_timeout_scores.append(sc)
            elif outcome == TaskOutcome.SUCCESS:
                wa_success_scores.append(sc)

        mw_failure_scores: list[float] = []
        mw_success_scores: list[float] = []

        for outcome, syms in mw_symbols:
            prefix = syms[:k]
            sc = compute_coverage_score(library, prefix, total_score)
            if outcome.is_failure:
                mw_failure_scores.append(sc)
            else:
                mw_success_scores.append(sc)

        nonzero = sum(1 for s in wa_failure_scores if s > 0)
        match_rate = (nonzero / len(wa_failure_scores) * 100.0) if wa_failure_scores else 0.0

        per_k[str(k)] = {
            "webarena_failure": score_group(wa_failure_scores),
            "webarena_timeout": score_group(wa_timeout_scores),
            "webarena_success": score_group(wa_success_scores),
            "miniwob_failure": score_group(mw_failure_scores),
            "miniwob_success": score_group(mw_success_scores),
            "pattern_match_rate": round(match_rate, 2),
        }

    # Pattern-level analysis at max K
    pattern_match_counts: Counter[int] = Counter()
    for _, syms in wa_symbols:
        prefix = syms[:max_k]
        for idx, pat in enumerate(library.patterns):
            if _is_subsequence(pat.symbols, prefix):
                pattern_match_counts[idx] += 1

    top_matched = []
    for idx, count in pattern_match_counts.most_common(10):
        pat = library.patterns[idx]
        top_matched.append({
            "symbols": pat.symbols,
            "match_count": count,
            "failure_type": pat.failure_type.value if pat.failure_type else None,
            "score": round(pat.score, 4),
            "precision": round(pat.precision, 4),
        })

    matched_indices = set(pattern_match_counts.keys())
    non_transferable = []
    for idx, pat in enumerate(library.patterns):
        if idx not in matched_indices:
            non_transferable.append({
                "symbols": pat.symbols,
                "failure_type": pat.failure_type.value if pat.failure_type else None,
                "score": round(pat.score, 4),
            })

    # Observation distribution
    obs_miniwob = compute_observation_distribution(miniwob_traces, symbolizer)
    obs_webarena = compute_observation_distribution(webarena_traces, symbolizer)

    return {
        "per_k": per_k,
        "top_matched_patterns": top_matched,
        "non_transferable_patterns": non_transferable,
        "observation_distribution": {
            "miniwob": obs_miniwob,
            "webarena": obs_webarena,
        },
    }


# =========================================================================
# Reporting
# =========================================================================

def print_report(results: dict, library: SignatureLibrary) -> None:
    """Print a formatted report to stdout.

    Args:
        results: Full results dictionary from run_transfer_analysis.
        library: The loaded SignatureLibrary (for summary stats).
    """
    print("=" * 70)
    print("CROSS-BENCHMARK TRANSFER ANALYSIS")
    print("=" * 70)
    print()

    lib_summary = library.summary()
    print(f"Pattern library: {lib_summary['total']} patterns")
    if lib_summary["total"] > 0:
        print(f"  Score range:     {lib_summary['score']['min']:.4f} – {lib_summary['score']['max']:.4f}")
        print(f"  Precision range: {lib_summary['precision']['min']:.4f} – {lib_summary['precision']['max']:.4f}")
        print(f"  Failure types:   {lib_summary['failure_types']}")
    print()

    per_k = results["per_k"]
    for k in sorted(per_k.keys(), key=int):
        data = per_k[k]
        print("-" * 70)
        print(f"K = {k}")
        print("-" * 70)

        for label, key in [
            ("WebArena failure ", "webarena_failure"),
            ("WebArena timeout ", "webarena_timeout"),
            ("WebArena success ", "webarena_success"),
            ("MiniWoB  failure ", "miniwob_failure"),
            ("MiniWoB  success ", "miniwob_success"),
        ]:
            g = data[key]
            caveat = ""
            if key == "webarena_success" and g["count"] <= 5:
                caveat = f"  (N={g['count']} — interpret with caution)"
            print(
                f"  {label}  N={g['count']:>4}  "
                f"mean={g['mean']:.6f}  median={g['median']:.6f}  "
                f"std={g['std']:.6f}{caveat}"
            )

        print(f"  Pattern match rate (WA failures with coverage > 0): "
              f"{data['pattern_match_rate']:.1f}%")
        print()

    print("=" * 70)
    print("TOP 10 MOST-MATCHED PATTERNS (at max K, against all WebArena traces)")
    print("=" * 70)

    for i, pat in enumerate(results["top_matched_patterns"], 1):
        seq = " -> ".join(pat["symbols"][:5])
        if len(pat["symbols"]) > 5:
            seq += " -> ..."
        ftype = pat["failure_type"] or "n/a"
        print(f"  {i:>2}. [{pat['match_count']} matches] {seq}")
        print(f"      type={ftype}  score={pat['score']:.4f}  prec={pat['precision']:.4f}")

    non_t = results["non_transferable_patterns"]
    print()
    print(f"Non-transferable patterns (in library but never matched WebArena): "
          f"{len(non_t)}/{lib_summary['total']}")
    if non_t:
        for pat in non_t[:10]:
            seq = " -> ".join(pat["symbols"][:5])
            if len(pat["symbols"]) > 5:
                seq += " -> ..."
            ftype = pat["failure_type"] or "n/a"
            print(f"    - {seq}  (type={ftype}, score={pat['score']:.4f})")
        if len(non_t) > 10:
            print(f"    ... and {len(non_t) - 10} more")

    print()
    print("=" * 70)
    print("OBSERVATION DISTRIBUTION (symbolized step outcomes)")
    print("=" * 70)
    print()
    print(f"  {'Benchmark':<12} {'Total Steps':>12} {'SUCCESS %':>10} "
          f"{'FAIL %':>8} {'UNKNOWN %':>10} {'R_STUCK %':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

    for bm_key, bm_label in [("miniwob", "MiniWoB"), ("webarena", "WebArena")]:
        obs = results["observation_distribution"][bm_key]
        print(
            f"  {bm_label:<12} {obs['total_steps']:>12,} {obs['success_pct']:>9.1f}% "
            f"{obs['fail_pct']:>7.1f}% {obs['unknown_pct']:>9.1f}% "
            f"{obs['r_stuck_pct']:>9.1f}%"
        )
    print()


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test cross-benchmark transfer of MiniWoB failure patterns to WebArena",
    )
    parser.add_argument(
        "--pattern-library",
        type=str,
        required=True,
        help="Path to SignatureLibrary JSON (trained on MiniWoB)",
    )
    parser.add_argument(
        "--webarena-traces",
        type=str,
        default="data/raw_traces/webarena/llama-3.2-3b/",
        help="Directory containing WebArena trace JSON files",
    )
    parser.add_argument(
        "--miniwob-traces",
        type=str,
        default="data/medium_traces/",
        help="Directory containing MiniWoB trace JSON files",
    )
    parser.add_argument(
        "--exclude-errors",
        action="store_true",
        default=False,
        help="Skip traces with outcome 'error' (typically 0-step traces)",
    )
    parser.add_argument(
        "--exclude-timeouts",
        action="store_true",
        default=False,
        help="Skip traces with outcome 'timeout'",
    )
    parser.add_argument(
        "--abstraction-level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Symbolization level (0=fine, 1=medium, 2=coarse)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/experiment_results/cross_benchmark_transfer.json",
        help="Path for JSON output file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    library_path = Path(args.pattern_library)
    webarena_dir = Path(args.webarena_traces)
    miniwob_dir = Path(args.miniwob_traces)
    output_path = Path(args.output)

    if not library_path.exists():
        print(f"ERROR: Pattern library not found: {library_path}")
        return 1
    if not webarena_dir.exists():
        print(f"ERROR: WebArena trace directory not found: {webarena_dir}")
        return 1
    if not miniwob_dir.exists():
        print(f"ERROR: MiniWoB trace directory not found: {miniwob_dir}")
        return 1

    # --- Load pattern library ---
    print(f"Loading pattern library from {library_path}...")
    library = SignatureLibrary.load(library_path)
    lib_summary = library.summary()
    print(f"   {lib_summary['total']} patterns loaded")
    print()

    # --- Load WebArena traces ---
    print(f"Loading WebArena traces from {webarena_dir}...")
    wa_traces = load_traces(webarena_dir)
    wa_outcomes = Counter(t.metadata.outcome.value for t in wa_traces)
    print(f"   {len(wa_traces)} traces: {dict(wa_outcomes)}")

    wa_traces = filter_traces(wa_traces, args.exclude_errors, args.exclude_timeouts)
    print()

    # --- Load MiniWoB traces ---
    print(f"Loading MiniWoB traces from {miniwob_dir}...")
    mw_traces = load_traces(miniwob_dir)
    mw_outcomes = Counter(t.metadata.outcome.value for t in mw_traces)
    print(f"   {len(mw_traces)} traces: {dict(mw_outcomes)}")

    mw_traces = filter_traces(mw_traces, args.exclude_errors, args.exclude_timeouts)
    print()

    if not wa_traces:
        print("ERROR: No WebArena traces after filtering")
        return 1

    # --- Run analysis ---
    symbolizer = TraceSymbolizer(abstraction_level=args.abstraction_level)
    print(f"Running transfer analysis (abstraction_level={args.abstraction_level})...")
    print()

    results = run_transfer_analysis(
        library=library,
        webarena_traces=wa_traces,
        miniwob_traces=mw_traces,
        symbolizer=symbolizer,
        k_values=K_VALUES,
    )

    # --- Add config metadata ---
    full_output = {
        "config": {
            "pattern_library": str(library_path),
            "webarena_traces": str(webarena_dir),
            "miniwob_traces": str(miniwob_dir),
            "abstraction_level": args.abstraction_level,
            "exclude_errors": args.exclude_errors,
            "exclude_timeouts": args.exclude_timeouts,
            "k_values": K_VALUES,
        },
        "library_summary": lib_summary,
        "webarena_trace_count": len(wa_traces),
        "miniwob_trace_count": len(mw_traces),
        **results,
    }

    # --- Report ---
    print_report(results, library)

    # --- Save JSON ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
