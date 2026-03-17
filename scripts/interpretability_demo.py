#!/usr/bin/env python3
"""
Interpretability Demo - Demonstrate BIDE Coverage pattern matching explanations

Purpose: Show concrete pattern matches on real traces to illustrate the
interpretability advantage of BIDE Coverage over opaque baselines like N-gram.
Loads traces, fits models, selects interesting test cases, and prints detailed
pattern match explanations with human-readable symbol descriptions.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.baselines.bide_coverage import BIDECoverageBaseline
from src.baselines.ngram import NGramBaseline
from src.data_collection.trace_logger import TraceLogger
from src.data_collection.trace_schema import AgentTrace, TaskOutcome
from src.evaluation.data_split import DataSplitter
from src.mining.pattern_ranker import ScoredPattern
from src.preprocessing.k_prefix import PrefixEntry, batch_extract_prefixes
from src.preprocessing.symbolizer import TraceSymbolizer
from src.utils.symbol_descriptions import describe_symbol, describe_pattern

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TRACE_LABELS: dict[str, str] = {
    "failure_high_score": "TRUE POSITIVE — Failed trace with HIGH failure score",
    "success_low_score": "TRUE NEGATIVE — Successful trace with LOW failure score",
    "success_high_score": "FALSE POSITIVE — Successful trace with HIGH failure score",
}


# =============================================================================
# Subsequence match with positions
# =============================================================================

def find_match_positions(
    pattern: list[str], sequence: list[str],
) -> list[int] | None:
    """Find the indices in *sequence* where each *pattern* symbol matches.

    Works like ``BIDECoverageBaseline.is_subsequence`` but records the
    position of each matched element.

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


# =============================================================================
# Data loading and preparation
# =============================================================================

def load_and_prepare(
    trace_dir: Path,
    k: int,
    seed: int,
) -> tuple[list[PrefixEntry], list[PrefixEntry], dict[str, AgentTrace]]:
    """Load traces, symbolize, split, and build a trace lookup dict.

    Args:
        trace_dir: Root directory containing trace JSON files.
        k: Prefix length for symbolization.
        seed: Random seed for the train/test split.

    Returns:
        Tuple of (train_entries, test_entries, traces_by_id).
    """
    tl = TraceLogger(base_dir=str(trace_dir.parent))
    traces = list(tl.iter_traces(directory=trace_dir))

    if not traces:
        print("ERROR: No traces loaded from", trace_dir)
        sys.exit(1)

    traces_by_id: dict[str, AgentTrace] = {
        t.metadata.trace_id: t for t in traces
    }

    failures = sum(1 for t in traces if t.metadata.outcome.is_failure)
    successes = len(traces) - failures
    websites = sorted({t.metadata.website for t in traces})
    models = sorted({t.metadata.model for t in traces})

    print(f"  Traces:   {len(traces)} ({failures} failures, {successes} successes)")
    print(f"  Websites: {websites}")
    print(f"  Models:   {models}")

    symbolizer = TraceSymbolizer(abstraction_level=1)
    dataset = batch_extract_prefixes(traces, k=k, symbolizer=symbolizer)

    splitter = DataSplitter()
    split = splitter.split(dataset.entries, seed=seed)

    print(f"  Split:    {split.summary()}")

    return split.train, split.test, traces_by_id


# =============================================================================
# Interesting trace selection
# =============================================================================

def find_interesting_traces(
    bide: BIDECoverageBaseline,
    test_entries: list[PrefixEntry],
) -> dict[str, tuple[PrefixEntry, float]]:
    """Pick 3 interesting test traces for the interpretability demo.

    Scores every test entry and selects:
    - failure_high_score: failure with the highest BIDE score (true positive)
    - success_low_score:  success with the lowest BIDE score  (true negative)
    - success_high_score: success with the highest BIDE score (false positive)

    Args:
        bide: Fitted BIDECoverageBaseline.
        test_entries: Test-set prefix entries.

    Returns:
        Dict mapping label to ``(PrefixEntry, score)``.
    """
    scored = [(e, bide.predict(e.symbols)) for e in test_entries]

    failures = [(e, s) for e, s in scored if e.outcome.is_failure]
    successes = [(e, s) for e, s in scored if e.outcome == TaskOutcome.SUCCESS]

    result: dict[str, tuple[PrefixEntry, float]] = {}

    if failures:
        result["failure_high_score"] = max(failures, key=lambda x: x[1])
    else:
        print("  WARNING: No failure traces in test set")

    if successes:
        result["success_low_score"] = min(successes, key=lambda x: x[1])
        result["success_high_score"] = max(successes, key=lambda x: x[1])
    else:
        print("  WARNING: No success traces in test set")

    return result


# =============================================================================
# Detailed trace output
# =============================================================================

def print_trace_detail(
    label: str,
    entry: PrefixEntry,
    score: float,
    bide: BIDECoverageBaseline,
    traces_by_id: dict[str, AgentTrace],
) -> None:
    """Print full interpretability output for a single trace.

    Shows the symbolic sequence, overall score, every matching pattern
    with positions, and a plain-language interpretation of the top match.

    Args:
        label: Category key (e.g. ``failure_high_score``).
        entry: The PrefixEntry being analysed.
        score: Overall failure score from ``bide.predict()``.
        bide: Fitted BIDE baseline (for pattern library access).
        traces_by_id: Lookup from trace_id to AgentTrace.
    """
    trace = traces_by_id.get(entry.trace_id)
    task_name = trace.metadata.task_description if trace else "unknown"
    model = trace.metadata.model if trace else "unknown"
    total_steps = trace.total_steps if trace else len(entry.symbols)

    heading = TRACE_LABELS.get(label, label)
    print(f"  --- {heading} ---")
    print(f"  Trace ID:    {entry.trace_id}")
    print(f"  Task:        {task_name}")
    print(f"  Model:       {model}")
    print(f"  Outcome:     {entry.outcome.value}")
    print(f"  Total steps: {total_steps}")
    print()

    print(f"  Symbolic sequence (K={len(entry.symbols)} prefix):")
    for i, sym in enumerate(entry.symbols):
        print(f"    [{i:2d}] {sym}")
    print()

    print(f"  Overall failure score: {score:.4f}")
    print()

    matching: list[tuple[int, ScoredPattern, list[int]]] = []
    for rank, pattern in enumerate(bide.patterns_, start=1):
        positions = find_match_positions(pattern.symbols, entry.symbols)
        if positions is not None:
            matching.append((rank, pattern, positions))

    if not matching:
        print("  No patterns matched this trace.")
        print()
        return

    print(f"  Matching patterns ({len(matching)} of {len(bide.patterns_)} in library):")
    print()
    for rank, pattern, positions in matching:
        pos_str = ", ".join(str(p) for p in positions)
        print(f"    Pattern #{rank}:")
        print(f"      Symbols:    {' -> '.join(pattern.symbols)}")
        print(f"      Precision:  {pattern.precision:.4f}")
        print(f"      Score:      {pattern.score:.4f}")
        print(f"      Length:     {len(pattern.symbols)}")
        print(f"      Matched at: [{pos_str}]")
        print()

    top_rank, top_pattern, top_positions = matching[0]
    print(f"  Plain-language interpretation of top match (pattern #{top_rank}):")
    print(f"    \"{describe_pattern(top_pattern.symbols)}\"")
    print()
    print("    Step-by-step breakdown:")
    for i, (sym, pos) in enumerate(zip(top_pattern.symbols, top_positions)):
        print(f"      {i + 1}. (trace position {pos}) {describe_symbol(sym)}")
    print()


# =============================================================================
# Pattern library summary
# =============================================================================

def print_pattern_library_summary(bide: BIDECoverageBaseline) -> None:
    """Print summary statistics about the fitted pattern library.

    Shows total count, score range, top-10 patterns, and length distribution.

    Args:
        bide: Fitted BIDECoverageBaseline with populated ``patterns_``.
    """
    patterns = bide.patterns_
    if not patterns:
        print("  Pattern library is empty — no patterns were mined.")
        return

    scores = [p.score for p in patterns]
    precisions = [p.precision for p in patterns]

    print(f"  Total patterns:  {len(patterns)}")
    print(f"  Score range:     [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  Precision range: [{min(precisions):.4f}, {max(precisions):.4f}]")
    print()

    top_n = min(10, len(patterns))
    print(f"  Top {top_n} patterns:")
    for i, p in enumerate(patterns[:top_n], start=1):
        sym_str = " -> ".join(p.symbols)
        print(f"    {i:2d}. [{sym_str}]")
        print(
            f"        precision={p.precision:.4f}  "
            f"coverage={p.coverage}  "
            f"score={p.score:.4f}  "
            f"support={p.support}"
        )
    print()

    lengths = Counter(len(p.symbols) for p in patterns)
    print("  Pattern length distribution:")
    for length in sorted(lengths):
        count = lengths[length]
        bar = "#" * min(count, 60)
        print(f"    length {length:2d}: {count:3d} {bar}")


# =============================================================================
# N-gram comparison
# =============================================================================

def print_ngram_comparison(
    ngram: NGramBaseline,
    bide: BIDECoverageBaseline,
    interesting: dict[str, tuple[PrefixEntry, float]],
) -> None:
    """Print N-gram vs BIDE scores for the same traces.

    Highlights that N-gram only produces an opaque probability while
    BIDE provides interpretable pattern matches.

    Args:
        ngram: Fitted NGramBaseline.
        bide: Fitted BIDECoverageBaseline.
        interesting: Dict of label -> (PrefixEntry, bide_score).
    """
    for label, (entry, bide_score) in interesting.items():
        ngram_score = ngram.predict(entry.symbols)
        matching_count = sum(
            1 for p in bide.patterns_
            if BIDECoverageBaseline.is_subsequence(p.symbols, entry.symbols)
        )

        heading = TRACE_LABELS.get(label, label)
        print(f"  {heading}")
        print(f"    Trace:          {entry.trace_id}")
        print(f"    Outcome:        {entry.outcome.value}")
        print(f"    N-gram score:   {ngram_score:.4f}  <- opaque probability, no explanation")
        print(f"    BIDE score:     {bide_score:.4f}  <- backed by {matching_count} interpretable pattern(s)")
        print()


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with trace_dir, k, seed, spmf_jar.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate BIDE Coverage interpretability by showing "
            "concrete pattern matches on real traces"
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="data/raw_traces",
        help="Directory containing trace JSON files (default: data/raw_traces)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="K-prefix length for symbolization (default: 10)",
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
        help="Path to SPMF jar file (default: lib/spmf.jar)",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    """Run the interpretability demo end-to-end.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    spmf_jar = Path(args.spmf_jar)

    if not trace_dir.exists():
        print(f"ERROR: Trace directory not found: {trace_dir}")
        return 1

    if not spmf_jar.exists():
        print(f"ERROR: SPMF jar not found: {spmf_jar}")
        return 1

    # ------------------------------------------------------------------
    # 1. Load, symbolize, split
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SECTION 1: DATA LOADING")
    print("=" * 70)
    train_entries, test_entries, traces_by_id = load_and_prepare(
        trace_dir, k=args.k, seed=args.seed,
    )
    print()

    # ------------------------------------------------------------------
    # 2. Fit BIDE Coverage
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SECTION 2: FITTING MODELS")
    print("=" * 70)

    print("  Fitting BIDE Coverage baseline...")
    bide = BIDECoverageBaseline(spmf_jar_path=str(spmf_jar))
    bide.fit(train_entries)
    print(f"  BIDE: {len(bide.patterns_)} patterns in library")

    print("  Fitting N-gram baseline...")
    ngram = NGramBaseline()
    ngram.fit(train_entries)
    print(f"  N-gram: {len(ngram.vocab_ or {})} n-gram features")
    print()

    # ------------------------------------------------------------------
    # 3. Pattern library summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SECTION 3: PATTERN LIBRARY SUMMARY")
    print("=" * 70)
    print_pattern_library_summary(bide)
    print()

    # ------------------------------------------------------------------
    # 4. Select interesting traces
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SECTION 4: INTERPRETABLE TRACE ANALYSIS")
    print("=" * 70)
    print()

    interesting = find_interesting_traces(bide, test_entries)

    if not interesting:
        print("  ERROR: Could not find any interesting traces.")
        return 1

    for label in ("failure_high_score", "success_low_score", "success_high_score"):
        if label not in interesting:
            continue
        entry, score = interesting[label]
        print_trace_detail(label, entry, score, bide, traces_by_id)
        print("-" * 70)
        print()

    # ------------------------------------------------------------------
    # 5. N-gram comparison
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SECTION 5: COMPARISON WITH N-GRAM BASELINE")
    print("=" * 70)
    print()
    print("  N-gram uses a Random Forest over binary n-gram features.")
    print("  It produces a single probability with no explanation of")
    print("  which patterns or features contributed to the decision.")
    print()

    print_ngram_comparison(ngram, bide, interesting)

    # ------------------------------------------------------------------
    # 6. Conclusion
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SECTION 6: INTERPRETABILITY SUMMARY")
    print("=" * 70)
    print()
    print("  BIDE Coverage advantages:")
    print("    - Each prediction is backed by specific sequential patterns")
    print("    - Patterns map directly to human-readable agent behaviors")
    print("    - Match positions show exactly where in the trace each")
    print("      pattern element was observed")
    print("    - Pattern precision quantifies how reliable each signal is")
    print("    - A practitioner can inspect WHY a trace was flagged,")
    print("      not just THAT it was flagged")
    print()
    print("  N-gram / Random Forest limitations:")
    print("    - Outputs an opaque probability score")
    print("    - No indication of which n-grams drove the decision")
    print("    - Feature importance is global, not per-prediction")
    print("    - Cannot point to specific trace positions or behaviors")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
