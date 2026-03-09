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
import sys
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
) -> list[BaseBaseline]:
    """Instantiate baseline objects from the registry.

    SPMF-dependent baselines (bide_coverage, taspm) are skipped with a
    warning if the SPMF jar is not found on disk.

    Args:
        names: Baseline names to instantiate (keys in BASELINES).
        spmf_jar: Path to spmf.jar for SPMF-dependent baselines.

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


def save_results(
    results: ExperimentResults,
    summary_text: str,
    output_dir: Path,
) -> None:
    """Persist experiment results and summary table to disk.

    Args:
        results: ExperimentResults from the runner.
        summary_text: ASCII + LaTeX summary from summary_table().
        output_dir: Directory to write output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "split_info": results.split_info,
        "k_values": results.k_values,
        "failure_rate": results.failure_rate,
        "per_baseline": _serialize_value(results.per_baseline),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    table_path = output_dir / "summary_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"   Saved {results_path}")
    print(f"   Saved {table_path}")


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
    return parser.parse_args()


# =========================================================================
# Main
# =========================================================================

def main() -> int:
    args = parse_args()

    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    spmf_jar = Path(args.spmf_jar)
    k_values = [int(v.strip()) for v in args.k_values.split(",")]

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

    # --- Build PrefixDataset ---
    print("[1/4] Symbolizing traces and extracting k-prefixes...")
    symbolizer = TraceSymbolizer(abstraction_level=args.abstraction_level)
    dataset = batch_extract_prefixes(traces, k=args.k, symbolizer=symbolizer)

    summary = dataset.summary()
    print(f"   Prefixes: {len(dataset)}")
    print(f"   Outcomes: {summary['outcomes']}")
    print()

    # --- Instantiate baselines ---
    print("[2/4] Instantiating baselines...")
    baseline_instances = instantiate_baselines(baseline_names, spmf_jar)

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

    print("[4/4] Saving results...")
    save_results(results, summary_text, output_dir)
    print()

    print(f"Split info: {results.split_info}")
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
