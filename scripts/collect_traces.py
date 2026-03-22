#!/usr/bin/env python3
"""
MiniWoB Trace Collection — CLI Entry Point

Runs an LLM agent against MiniWoB++ tasks via BrowserGym, captures
structured execution traces (reasoning -> action -> observation per step),
and saves them using the existing trace logging infrastructure.

Designed to be called by SLURM batch jobs on SJSU HPC GPU nodes.

Usage:
    python scripts/collect_traces.py \
        --model-name llama-3.2-3b \
        --model-path /home/017557527/cmpe299b/models/llama-3.2-3b \
        --output-dir data/raw_traces \
        --benchmark miniwob \
        --max-steps 30 \
        --tasks click-test click-button enter-text

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential
         Pattern Mining
"""

import argparse
import logging
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_collection.agent_runner import (
    AgentConfig,
    BrowserGymAgentRunner,
    TaskConfig,
)
from src.data_collection.trace_logger import TraceLogger
from src.data_collection.trace_schema import TaskOutcome

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================================
# Task enumeration
# =========================================================================

def enumerate_miniwob_tasks() -> list[str]:
    """Return sorted list of all registered MiniWoB task names.

    Each name is in the form ``miniwob.<task-slug>``, e.g.
    ``miniwob.click-test``.
    """
    import gymnasium
    import browsergym.miniwob  # noqa: F401 — registers envs

    prefix = "browsergym/miniwob."
    tasks = sorted(
        name.split("browsergym/")[1]
        for name in gymnasium.envs.registry.keys()
        if name.startswith(prefix)
    )
    return tasks


def resolve_task_list(
    requested: list[str] | None,
    all_tasks: list[str],
) -> list[str]:
    """Filter *all_tasks* to those the user requested.

    Accepts short names (``click-test``) or fully-qualified names
    (``miniwob.click-test``).  Returns the fully-qualified versions.

    Args:
        requested: CLI ``--tasks`` values, or None for all.
        all_tasks: Full list from :func:`enumerate_miniwob_tasks`.

    Returns:
        Filtered (or complete) task list.
    """
    if not requested:
        return all_tasks

    selected: list[str] = []
    for name in requested:
        qualified = name if name.startswith("miniwob.") else f"miniwob.{name}"
        if qualified in all_tasks:
            selected.append(qualified)
        else:
            logger.warning("Task '%s' not found in registry — skipping", name)

    return selected


# =========================================================================
# Main collection loop
# =========================================================================

def collect(
    runner: BrowserGymAgentRunner,
    trace_logger: TraceLogger,
    tasks: list[str],
    benchmark: str,
) -> dict:
    """Run the agent on every task, save traces, return summary stats.

    Args:
        runner: Configured agent runner (model will be lazy-loaded).
        trace_logger: Logger that auto-saves traces to disk.
        tasks: Fully-qualified task names.
        benchmark: Benchmark slug (e.g. ``miniwob``).

    Returns:
        Dict with ``success``, ``failure``, ``error``, ``timeout`` counts.
    """
    stats: dict[str, int] = {
        "success": 0,
        "failure": 0,
        "timeout": 0,
        "error": 0,
    }
    total = len(tasks)
    collection_start = time.time()

    for i, task_name in enumerate(tasks):
        task = TaskConfig(
            task_id=task_name,
            task_description=task_name,
            website="miniwob",
            benchmark=benchmark,
        )

        task_start = time.time()
        try:
            trace = runner.run_task(task)
            elapsed = time.time() - task_start
            outcome = trace.metadata.outcome.value.upper()
            stats[trace.metadata.outcome.value] = (
                stats.get(trace.metadata.outcome.value, 0) + 1
            )
            print(
                f"Task {i + 1}/{total}: {task_name} -- {outcome} "
                f"({trace.total_steps} steps, {elapsed:.1f}s)"
            )
        except Exception:
            elapsed = time.time() - task_start
            stats["error"] += 1
            logger.exception(
                "Task %d/%d %s crashed after %.1fs",
                i + 1,
                total,
                task_name,
                elapsed,
            )

    wall_time = time.time() - collection_start
    stats["wall_time_s"] = round(wall_time, 1)
    return stats


def print_summary(stats: dict, total: int) -> None:
    """Print a human-readable summary table."""
    print()
    print("=" * 55)
    print("  Trace Collection Summary")
    print("=" * 55)

    completed = stats.get("success", 0) + stats.get("failure", 0) + stats.get("timeout", 0)
    errors = stats.get("error", 0)

    print(f"  Total tasks attempted : {total}")
    print(f"  Completed             : {completed}")
    print(f"  Errors (crashed)      : {errors}")
    print(f"  ---")
    print(f"  Success               : {stats.get('success', 0)}")
    print(f"  Failure               : {stats.get('failure', 0)}")
    print(f"  Timeout               : {stats.get('timeout', 0)}")

    if completed > 0:
        rate = stats.get("success", 0) / completed * 100
        print(f"  Success rate          : {rate:.1f}%")

    wall = stats.get("wall_time_s", 0)
    print(f"  Wall-clock time       : {wall:.1f}s ({wall / 60:.1f}m)")
    print("=" * 55)


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MiniWoB execution traces using a local LLM agent",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Short model identifier, e.g. llama-3.2-3b",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Filesystem path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw_traces",
        help="Base directory for trace output (default: data/raw_traces)",
    )
    parser.add_argument(
        "--benchmark",
        default="miniwob",
        help="Benchmark name (default: miniwob)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum agent steps per task (default: 30)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of task names (e.g. click-test enter-text)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature; 0 = greedy (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max new tokens per LLM call (default: 1024)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--rich-observations",
        action="store_true",
        help="Populate ObservationRecord from BrowserGym obs dict (last_action_error, open_pages_urls)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 55)
    print("  MiniWoB Trace Collection")
    print("=" * 55)
    print(f"  Model       : {args.model_name}")
    print(f"  Model path  : {args.model_path}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Benchmark   : {args.benchmark}")
    print(f"  Max steps   : {args.max_steps}")
    print(f"  Temperature : {args.temperature}")
    print(f"  4-bit       : {args.load_in_4bit}")
    print(f"  Rich obs    : {args.rich_observations}")
    print()

    # --- enumerate tasks ---
    print("Enumerating MiniWoB tasks...")
    all_tasks = enumerate_miniwob_tasks()
    tasks = resolve_task_list(args.tasks, all_tasks)

    if not tasks:
        print("ERROR: no tasks matched. Check --tasks values.")
        return 1

    print(f"Selected {len(tasks)} / {len(all_tasks)} tasks")
    print()

    # --- configure components ---
    agent_config = AgentConfig(
        model_name=args.model_name,
        model_path=args.model_path,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        load_in_4bit=args.load_in_4bit,
    )

    trace_logger = TraceLogger(
        base_dir=args.output_dir,
        benchmark=args.benchmark,
        model=args.model_name,
    )

    runner = BrowserGymAgentRunner(
        agent_config=agent_config,
        trace_logger=trace_logger,
        benchmark=args.benchmark,
        headless=True,
        rich_observations=args.rich_observations,
    )

    # --- pre-load the model so it doesn't count toward task 1 timing ---
    print("Loading model onto GPU...")
    runner.load_model()
    print("Model loaded.\n")

    # --- run ---
    stats = collect(runner, trace_logger, tasks, args.benchmark)
    print_summary(stats, len(tasks))

    trace_logger.print_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
