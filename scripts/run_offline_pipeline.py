#!/usr/bin/env python3
"""
Offline Training Pipeline - Generate a ranked pattern library from agent traces

Runs the complete offline pipeline:
    Raw Traces -> Symbolize -> K-Prefix -> BIDE Mine -> Rank -> Pattern Library

Supports synthetic traces for validation and real traces from WebArena/BrowserGym.

Usage:
    python scripts/run_offline_pipeline.py --mode synthetic --n-traces 100
    python scripts/run_offline_pipeline.py --mode real --trace-dir data/raw_traces/webarena

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

from src.data_collection.trace_schema import (
    AgentTrace,
    TraceStep,
    TraceMetadata,
    ActionRecord,
    ObservationRecord,
    ReasoningRecord,
    TaskOutcome,
    FailureType,
    ActionType,
    SelectorType,
    ElementState,
    generate_trace_id,
    get_current_timestamp,
)
from src.preprocessing.symbolizer import TraceSymbolizer
from src.preprocessing.k_prefix import batch_extract_prefixes, PrefixDataset
from src.mining.spmf_wrapper import SPMFWrapper, SPMFConfig, RawPattern
from src.mining.pattern_ranker import PatternRanker, RankerConfig, ScoredPattern
from src.mining.signature_library import SignatureLibrary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================================
# Synthetic trace generation
# =========================================================================

WEBSITES = [
    "shopping.webarena.dev",
    "reddit.webarena.dev",
    "gitlab.webarena.dev",
    "wikipedia.webarena.dev",
    "map.webarena.dev",
]

FAILURE_SCENARIOS = [
    {
        "outcome": TaskOutcome.FAILURE,
        "failure_type": FailureType.NAVIGATION,
        "description": "Agent clicks wrong elements repeatedly",
        "pattern": [
            ("click", SelectorType.ID, True, ElementState.VISIBLE, []),
            ("click", SelectorType.CLASS, False, ElementState.NOT_FOUND, ["stuck"]),
            ("click", SelectorType.CLASS, False, ElementState.NOT_FOUND, ["retry"]),
            ("navigate", SelectorType.UNKNOWN, True, ElementState.VISIBLE, ["backtrack"]),
            ("click", SelectorType.XPATH, False, ElementState.STALE, ["confused"]),
        ],
    },
    {
        "outcome": TaskOutcome.FAILURE,
        "failure_type": FailureType.VALIDATION,
        "description": "Agent types into wrong fields",
        "pattern": [
            ("click", SelectorType.ID, True, ElementState.VISIBLE, []),
            ("type", SelectorType.ID, True, ElementState.INTERACTABLE, []),
            ("click", SelectorType.ID, True, ElementState.VISIBLE, ["verify"]),
            ("type", SelectorType.CLASS, False, ElementState.NOT_FOUND, ["retry"]),
            ("type", SelectorType.CLASS, False, ElementState.NOT_FOUND, ["stuck"]),
        ],
    },
    {
        "outcome": TaskOutcome.FAILURE,
        "failure_type": FailureType.RECOVERY,
        "description": "Agent gets stuck in retry loop",
        "pattern": [
            ("click", SelectorType.ID, True, ElementState.VISIBLE, []),
            ("click", SelectorType.ID, False, ElementState.NOT_FOUND, ["retry"]),
            ("go_back", SelectorType.UNKNOWN, True, ElementState.UNKNOWN, ["backtrack"]),
            ("click", SelectorType.ID, False, ElementState.NOT_FOUND, ["retry"]),
            ("go_back", SelectorType.UNKNOWN, True, ElementState.UNKNOWN, ["backtrack"]),
        ],
    },
    {
        "outcome": TaskOutcome.FAILURE,
        "failure_type": FailureType.CONTEXT,
        "description": "Agent misunderstands task",
        "pattern": [
            ("navigate", SelectorType.UNKNOWN, True, ElementState.VISIBLE, ["explore"]),
            ("click", SelectorType.TEXT, True, ElementState.VISIBLE, ["explore"]),
            ("scroll", SelectorType.UNKNOWN, True, ElementState.UNKNOWN, ["explore"]),
            ("click", SelectorType.TEXT, True, ElementState.VISIBLE, ["confused"]),
            ("navigate", SelectorType.UNKNOWN, True, ElementState.VISIBLE, ["backtrack"]),
        ],
    },
    {
        "outcome": TaskOutcome.SUCCESS,
        "failure_type": None,
        "description": "Successful task completion",
        "pattern": [
            ("click", SelectorType.ID, True, ElementState.VISIBLE, []),
            ("type", SelectorType.ID, True, ElementState.INTERACTABLE, []),
            ("click", SelectorType.ID, True, ElementState.VISIBLE, []),
            ("click", SelectorType.ID, True, ElementState.VISIBLE, ["verify"]),
            ("stop", SelectorType.UNKNOWN, True, ElementState.UNKNOWN, []),
        ],
    },
]

_ACTION_TYPE_MAP = {
    "click": ActionType.CLICK,
    "type": ActionType.TYPE,
    "navigate": ActionType.NAVIGATE,
    "scroll": ActionType.SCROLL,
    "go_back": ActionType.GO_BACK,
    "stop": ActionType.STOP,
    "hover": ActionType.HOVER,
    "select": ActionType.SELECT,
}


def generate_synthetic_traces(n_traces: int = 100) -> list[AgentTrace]:
    """Generate synthetic traces with known failure patterns across websites.

    Distributes traces across websites and scenarios so that failure
    patterns appear on multiple sites (enabling cross-site filtering).

    Args:
        n_traces: Total number of traces to generate.

    Returns:
        List of AgentTrace objects.
    """
    traces: list[AgentTrace] = []

    n_scenarios = len(FAILURE_SCENARIOS)
    n_websites = len(WEBSITES)

    for i in range(n_traces):
        scenario = FAILURE_SCENARIOS[i % n_scenarios]
        website = WEBSITES[(i // n_scenarios) % n_websites]

        trace = AgentTrace(
            metadata=TraceMetadata(
                trace_id=generate_trace_id(),
                task_id=f"task_{i:03d}",
                task_description=scenario["description"],
                website=website,
                model="test-model",
                outcome=scenario["outcome"],
                failure_type=scenario["failure_type"],
                start_time=get_current_timestamp(),
                benchmark="synthetic",
            )
        )

        for step_num, (act, sel, found, state, kw) in enumerate(
            scenario["pattern"], start=1
        ):
            step = TraceStep(
                step_number=step_num,
                reasoning=ReasoningRecord(
                    raw_reasoning=f"Step {step_num} reasoning",
                    intent=f"intent_{act}",
                    keywords=kw,
                ),
                action=ActionRecord(
                    type=_ACTION_TYPE_MAP.get(act, ActionType.UNKNOWN),
                    selector=f"#el-{step_num}",
                    selector_type=sel,
                ),
                observation=ObservationRecord(
                    element_found=found,
                    element_state=state,
                    http_status=200 if found else 404,
                ),
                dom_hash=f"hash{step_num:02d}",
                timestamp=get_current_timestamp(),
                url=f"http://{website}/page{step_num}",
                prompt_tokens=500,
                completion_tokens=30,
            )
            trace.add_step(step)

        traces.append(trace)

    return traces


def load_real_traces(trace_dir: Path) -> list[AgentTrace]:
    """Load real traces from a directory via TraceLogger.

    Args:
        trace_dir: Directory containing trace JSON files.

    Returns:
        List of AgentTrace objects.
    """
    from src.data_collection.trace_logger import TraceLogger

    tl = TraceLogger(base_dir=str(trace_dir.parent))
    traces = list(tl.iter_traces(directory=trace_dir))
    return traces


# =========================================================================
# Pipeline
# =========================================================================

def run_offline_pipeline(
    traces: list[AgentTrace],
    k: int = 10,
    abstraction_level: int = 1,
    min_support: float = 0.05,
    min_precision: float = 0.5,
    min_sites: int = 2,
    spmf_jar: Path = Path("lib/spmf.jar"),
    output_dir: Path = Path("data/pipeline_test"),
) -> SignatureLibrary | None:
    """Run the complete offline training pipeline.

    Args:
        traces: Agent execution traces to process.
        k: Number of prefix steps.
        abstraction_level: Symbolization level (0, 1, or 2).
        min_support: BIDE minimum support threshold.
        min_precision: Minimum pattern precision for filtering.
        min_sites: Minimum distinct websites for cross-site validity.
        spmf_jar: Path to spmf.jar.
        output_dir: Directory for intermediate and output files.

    Returns:
        The built SignatureLibrary, or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OFFLINE TRAINING PIPELINE")
    print("=" * 60)
    print(f"Traces: {len(traces)}")
    print(f"K: {k}")
    print(f"Abstraction Level: {abstraction_level}")
    print(f"Min Support: {min_support}")
    print(f"Min Precision: {min_precision}")
    print(f"Min Sites: {min_sites}")
    print(f"SPMF Jar: {spmf_jar}")
    print(f"Output Dir: {output_dir}")
    print()

    # Step 1: Symbolize & extract k-prefixes
    print("[1/5] Symbolizing traces and extracting k-prefixes...")
    symbolizer = TraceSymbolizer(abstraction_level=abstraction_level)
    dataset = batch_extract_prefixes(traces, k=k, symbolizer=symbolizer)

    summary = dataset.summary()
    print(f"   Symbolized {len(dataset)} traces")
    print(f"   Websites: {summary['websites']}")
    print(f"   Outcomes: {summary['outcomes']}")
    if dataset.entries:
        print(f"   Sample: {dataset.entries[0].symbols[:5]}...")

    dataset.save(output_dir / "prefix_dataset.json")

    n_websites = len(dataset.websites)
    if n_websites < min_sites:
        print(f"   WARNING: Only {n_websites} website(s) in dataset, "
              f"lowering min_sites from {min_sites} to {n_websites}")
        logger.warning(
            "Only %d website(s) in dataset, lowering min_sites from %d to %d",
            n_websites, min_sites, n_websites,
        )
        min_sites = n_websites
    print()

    # Step 2: Prepare SPMF input & run BIDE
    print("[2/5] Running BIDE pattern mining...")
    try:
        spmf_config = SPMFConfig(
            spmf_jar_path=spmf_jar,
            min_support=min_support,
            max_pattern_length=5,
            timeout_seconds=300,
        )
        spmf = SPMFWrapper(config=spmf_config)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"   ERROR: {e}")
        print("   Make sure SPMF is installed:")
        print("     mkdir -p lib && curl -L -o lib/spmf.jar https://www.philippe-fournier-viger.com/spmf/SPMF.jar")
        return None

    try:
        patterns, vocab = spmf.mine_patterns(
            dataset.sequences,
            work_dir=output_dir,
            min_support=min_support,
        )
    except Exception as e:
        print(f"   ERROR during BIDE execution: {e}")
        return None

    print(f"   Found {len(patterns)} raw patterns")
    print(f"   Vocabulary size: {len(vocab)}")
    print()

    # Step 3: Rank and filter patterns
    print("[3/5] Ranking patterns...")
    ranker_config = RankerConfig(
        min_precision=min_precision,
        min_sites=min_sites,
        min_support=2,
    )
    ranker = PatternRanker(config=ranker_config)
    scored = ranker.rank_patterns(patterns, dataset)

    print(f"   After filtering: {len(scored)} patterns")
    print(f"   (precision >= {min_precision}, sites >= {min_sites})")
    print()

    # Step 4: Build and save signature library
    print("[4/5] Building signature library...")
    library = SignatureLibrary(patterns=scored)
    library_path = output_dir / "pattern_library.json"
    library.save(library_path)

    print(f"   Saved to {library_path}")
    if len(library) > 0:
        lib_summary = library.summary()
        print(f"   Score range: {lib_summary['score']['min']:.4f} - {lib_summary['score']['max']:.4f}")
        print(f"   Precision range: {lib_summary['precision']['min']:.4f} - {lib_summary['precision']['max']:.4f}")
    print()

    # Step 5: Quick sanity check - match library against the data
    print("[5/5] Sanity check - matching library against training data...")
    match_count = 0
    for entry in dataset.entries:
        if library.has_match(entry.symbols):
            match_count += 1

    print(f"   {match_count}/{len(dataset)} traces matched at least one pattern")
    print()

    # Summary
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total patterns: {len(scored)}")

    if scored:
        print("\nTop 5 patterns:")
        for i, p in enumerate(scored[:5]):
            arrow_seq = " -> ".join(p.symbols[:4])
            if len(p.symbols) > 4:
                arrow_seq += "..."
            ftype = p.failure_type.value if p.failure_type else "n/a"
            print(
                f"   {i+1}. {arrow_seq} "
                f"(prec={p.precision:.2f}, cov={p.coverage}, "
                f"score={p.score:.3f}, type={ftype})"
            )

    return library


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline training pipeline for failure pattern mining",
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Use synthetic traces for testing or real traces from disk",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=100,
        help="Number of synthetic traces to generate (synthetic mode only)",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Directory containing real traces (real mode only)",
    )
    parser.add_argument("--k", type=int, default=10, help="K-prefix length")
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Abstraction level (0=fine, 1=medium, 2=coarse)",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.05,
        help="BIDE minimum support threshold",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.5,
        help="Minimum pattern precision for filtering",
    )
    parser.add_argument(
        "--min-sites",
        type=int,
        default=2,
        help="Minimum distinct websites for cross-site filtering",
    )
    parser.add_argument(
        "--spmf-jar",
        type=str,
        default="lib/spmf.jar",
        help="Path to spmf.jar",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pipeline_test",
        help="Output directory",
    )
    parser.add_argument(
        "--exclude-errors",
        action="store_true",
        default=False,
        help="Remove traces with outcome 'error' before mining",
    )
    parser.add_argument(
        "--exclude-timeouts",
        action="store_true",
        default=False,
        help="Remove traces with outcome 'timeout' before mining",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "synthetic":
        print(f"Generating {args.n_traces} synthetic traces...")
        traces = generate_synthetic_traces(args.n_traces)
    else:
        if not args.trace_dir:
            print("ERROR: --trace-dir is required for real mode")
            return 1
        trace_dir = Path(args.trace_dir)
        if not trace_dir.exists():
            print(f"ERROR: Trace directory not found: {trace_dir}")
            return 1
        print(f"Loading traces from {trace_dir}...")
        traces = load_real_traces(trace_dir)

    if not traces:
        print("ERROR: No traces loaded")
        return 1

    failures = sum(1 for t in traces if t.metadata.outcome.is_failure)
    successes = len(traces) - failures
    print(f"Loaded {len(traces)} traces ({failures} failures, {successes} successes)")
    print()

    if args.exclude_errors:
        original_count = len(traces)
        traces = [t for t in traces if t.metadata.outcome != TaskOutcome.ERROR]
        removed = original_count - len(traces)
        print(f"   Excluded {removed} error traces ({original_count} -> {len(traces)})")
        print()

    if args.exclude_timeouts:
        original_count = len(traces)
        traces = [t for t in traces if t.metadata.outcome != TaskOutcome.TIMEOUT]
        removed = original_count - len(traces)
        print(f"   Excluded {removed} timeout traces ({original_count} -> {len(traces)})")
        print()

    library = run_offline_pipeline(
        traces,
        k=args.k,
        abstraction_level=args.level,
        min_support=args.min_support,
        min_precision=args.min_precision,
        min_sites=args.min_sites,
        spmf_jar=Path(args.spmf_jar),
        output_dir=Path(args.output_dir),
    )

    if library is not None:
        print("\n✓ Pipeline complete!")
        return 0
    else:
        print("\n✗ Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
