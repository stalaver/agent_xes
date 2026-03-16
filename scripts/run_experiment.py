#!/usr/bin/env python3
"""
Experiment Runner CLI - Full 7-method baseline comparison via ExperimentRunner

Loads real traces, builds a PrefixDataset, instantiates baselines from the
registry, runs ExperimentRunner, and saves results + summary tables.

Usage:
    python scripts/run_experiment.py --trace-dir data/raw_traces
    python scripts/run_experiment.py --trace-dir data/raw_traces --baselines frequency_vector,ngram
    python scripts/run_experiment.py --trace-dir data/raw_traces --k-values 3,5,8,10

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.baselines import BASELINES, BaseBaseline
from src.data_collection.trace_logger import TraceLogger
from src.data_collection.trace_schema import AgentTrace, TaskOutcome
from src.evaluation.experiment import ExperimentRunner, ExperimentResults
from src.preprocessing.k_prefix import batch_extract_prefixes, PrefixDataset
from src.preprocessing.symbolizer import TraceSymbolizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SPMF_BASELINES = {"bide_coverage", "taspm"}


# =========================================================================
# Trace loading
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
# Baseline instantiation
# =========================================================================

def instantiate_baselines(
    names: list[str],
    spmf_jar: Path,
    ngram_ns: tuple[int, ...] | None = None,
) -> list[BaseBaseline]:
    """Instantiate baseline objects from the registry.

    SPMF-dependent baselines (bide_coverage, taspm) are skipped with a
    warning if the SPMF jar is not found on disk.

    Args:
        names: Baseline names to instantiate (keys in BASELINES).
        spmf_jar: Path to spmf.jar for SPMF-dependent baselines.
        ngram_ns: Optional n-gram sizes to pass to NGramBaseline.

    Returns:
        List of instantiated BaseBaseline objects.
    """
    spmf_available = spmf_jar.exists()
    instances: list[BaseBaseline] = []

    for name in names:
        cls = BASELINES.get(name)
        if cls is None:
            print(f"   WARNING: Unknown baseline '{name}' — skipping")
            continue

        if name in SPMF_BASELINES and not spmf_available:
            print(f"   WARNING: SPMF jar not found at {spmf_jar} — skipping {name}")
            continue

        if name == "bide_coverage":
            instances.append(cls(spmf_jar_path=str(spmf_jar)))
        elif name == "taspm":
            instances.append(cls(spmf_jar=str(spmf_jar)))
        elif name == "ngram" and ngram_ns is not None:
            instances.append(cls(ns=ngram_ns))
        else:
            instances.append(cls())

        logger.info("Instantiated baseline: %s", name)

    return instances


# =========================================================================
# Result serialization
# =========================================================================

def _serialize_value(v: object) -> object:
    """Make a value JSON-safe (handle NaN, numpy types)."""
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, dict):
        return {str(k): _serialize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_serialize_value(item) for item in v]
    return v


def build_run_dir(base_output_dir: Path, tag: str = "") -> Path:
    """Create a timestamped subdirectory for this experiment run.

    Args:
        base_output_dir: Parent directory for all experiment runs.
        tag: Optional human-readable label appended to the timestamp.

    Returns:
        Path to the new run directory (not yet created on disk).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{timestamp}_{tag}" if tag else timestamp
    return base_output_dir / name


def save_config(args: argparse.Namespace, run_dir: Path) -> None:
    """Save the CLI arguments as config.json for reproducibility.

    Args:
        args: Parsed CLI arguments.
        run_dir: Run directory to write into (must already exist).
    """
    config_path = run_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"   Saved {config_path}")


def _update_latest_symlink(base_output_dir: Path, run_dir: Path) -> None:
    """Point a ``latest`` symlink at the most recent run directory.

    Args:
        base_output_dir: Parent directory containing all run dirs.
        run_dir: The run directory to link to.
    """
    link_path = base_output_dir / "latest"
    rel_target = run_dir.relative_to(base_output_dir)
    if link_path.is_symlink() or link_path.exists():
        os.remove(link_path)
    os.symlink(rel_target, link_path)
    print(f"   Updated {link_path} -> {rel_target}")


def save_results(
    results: ExperimentResults,
    summary_text: str,
    run_dir: Path,
    base_output_dir: Path,
) -> None:
    """Persist experiment results and summary table to disk.

    Args:
        results: ExperimentResults from the runner.
        summary_text: ASCII + LaTeX summary from summary_table().
        run_dir: Timestamped run directory for this experiment.
        base_output_dir: Parent directory (used for the ``latest`` symlink).
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "split_info": results.split_info,
        "k_values": results.k_values,
        "failure_rate": results.failure_rate,
        "per_baseline": _serialize_value(results.per_baseline),
    }
    results_path = run_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    table_path = run_dir / "summary_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"   Saved {results_path}")
    print(f"   Saved {table_path}")

    _update_latest_symlink(base_output_dir, run_dir)


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full baseline comparison experiment",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="data/raw_traces",
        help="Directory containing trace JSON files",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="K-prefix length for symbolization",
    )
    parser.add_argument(
        "--abstraction-level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Symbolization level (0=fine, 1=medium, 2=coarse)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="3,5,8,10",
        help="Comma-separated prefix lengths for F1@K evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiment_results",
        help="Directory for experiment output files",
    )
    parser.add_argument(
        "--spmf-jar",
        type=str,
        default="lib/spmf.jar",
        help="Path to spmf.jar",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default=None,
        help="Comma-separated baseline names to run (default: all)",
    )
    parser.add_argument(
        "--ngram-ns",
        type=str,
        default=None,
        help="Comma-separated n-gram sizes for NGramBaseline (e.g. '2,3,4,5,8,10')",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Short label for this run (e.g. 'fix_threshold', 'coarse_symbols')",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=False,
        help="Downsample majority class to match minority class size",
    )
    parser.add_argument(
        "--exclude-errors",
        action="store_true",
        default=False,
        help="Remove traces with outcome 'error' before building the dataset",
    )
    parser.add_argument(
        "--exclude-timeouts",
        action="store_true",
        default=False,
        help="Remove traces with outcome 'timeout' before building the dataset",
    )
    return parser.parse_args()


# =========================================================================
# Main
# =========================================================================

def main() -> int:
    args = parse_args()

    trace_dir = Path(args.trace_dir)
    base_output_dir = Path(args.output_dir)
    spmf_jar = Path(args.spmf_jar)
    k_values = [int(v.strip()) for v in args.k_values.split(",")]
    run_dir = build_run_dir(base_output_dir, tag=args.tag)

    if args.baselines:
        baseline_names = [b.strip() for b in args.baselines.split(",")]
    else:
        baseline_names = list(BASELINES.keys())

    # --- Load traces ---
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
    websites = {t.metadata.website for t in traces}

    print(f"Loaded {len(traces)} traces ({failures} failures, {successes} successes)")
    print(f"Websites: {websites}")
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

    # --- Build PrefixDataset ---
    print("[1/4] Symbolizing traces and extracting k-prefixes...")
    symbolizer = TraceSymbolizer(abstraction_level=args.abstraction_level)
    dataset = batch_extract_prefixes(traces, k=args.k, symbolizer=symbolizer)

    summary = dataset.summary()
    print(f"   Prefixes: {len(dataset)}")
    print(f"   Outcomes: {summary['outcomes']}")
    print()

    if args.balanced:
        fail_entries = dataset.get_failed_entries()
        succ_entries = dataset.get_successful_entries()
        orig_fail, orig_succ = len(fail_entries), len(succ_entries)
        min_size = min(orig_fail, orig_succ)
        rng = random.Random(args.seed)
        if orig_fail > min_size:
            fail_entries = rng.sample(fail_entries, min_size)
        if orig_succ > min_size:
            succ_entries = rng.sample(succ_entries, min_size)
        dataset.entries = fail_entries + succ_entries
        print(f"   Balanced: {orig_fail} failures, {orig_succ} successes -> "
              f"{len(fail_entries)} failures, {len(succ_entries)} successes")
        print()

    # --- Instantiate baselines ---
    ngram_ns = tuple(int(v.strip()) for v in args.ngram_ns.split(",")) if args.ngram_ns else None

    print("[2/4] Instantiating baselines...")
    baseline_instances = instantiate_baselines(baseline_names, spmf_jar, ngram_ns=ngram_ns)

    if not baseline_instances:
        print("ERROR: No baselines could be instantiated")
        return 1

    print(f"   Active baselines: {[b.name for b in baseline_instances]}")
    print()

    # --- Run experiment ---
    print("[3/4] Running experiment...")
    print(f"   K-values: {k_values}")
    print(f"   Seed: {args.seed}")
    print()

    runner = ExperimentRunner(
        baselines=baseline_instances,
        dataset=dataset,
        k_values=k_values,
        seed=args.seed,
    )
    results = runner.run()

    # --- Print and save results ---
    summary_text = runner.summary_table(results)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(summary_text)
    print()

    print(f"[4/4] Saving results to {run_dir}...")
    save_results(results, summary_text, run_dir, base_output_dir)
    save_config(args, run_dir)
    print()

    print(f"Split info: {results.split_info}")
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
