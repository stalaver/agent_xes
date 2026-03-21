#!/usr/bin/env python3
"""
WebArena Shopping Trace Collection — CLI Entry Point

Runs an LLM agent against WebArena shopping tasks via BrowserGym, captures
structured execution traces (reasoning -> action -> observation per step),
and saves them using the existing trace logging infrastructure.

Only the ~194 non-admin shopping tasks are included by default. Task IDs
are discovered dynamically from the ``webarena`` package's test.raw.json.

Designed to be called by SLURM batch jobs on SJSU HPC GPU nodes.

Usage:
    python scripts/collect_webarena_traces.py \\
        --model-name llama-3.2-3b \\
        --model-path /home/017557527/cmpe299b/models/llama-3.2-3b \\
        --max-steps 30 \\
        --output-dir data/raw_traces

    # Specific tasks only:
    python scripts/collect_webarena_traces.py \\
        --model-name llama-3.2-3b \\
        --model-path /home/017557527/cmpe299b/models/llama-3.2-3b \\
        --tasks 21 22 23 24 25

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential
         Pattern Mining
"""

import argparse
import json
import logging
import os
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BENCHMARK = "webarena"

REQUIRED_ENV_VARS = [
    "WA_SHOPPING",
    "WA_SHOPPING_ADMIN",
    "WA_REDDIT",
    "WA_GITLAB",
    "WA_WIKIPEDIA",
    "WA_MAP",
    "WA_HOMEPAGE",
]


# =========================================================================
# Environment validation
# =========================================================================

def check_env_vars() -> list[str]:
    """Return list of missing required WebArena environment variables."""
    return [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]


# =========================================================================
# Task discovery
# =========================================================================

def load_shopping_tasks() -> list[dict]:
    """Load shopping task definitions from the webarena package.

    Reads ``test.raw.json`` shipped with the installed ``webarena``
    package and filters to non-admin shopping tasks.

    Returns:
        List of dicts with ``task_id`` (int) and ``intent`` (str) keys.
    """
    import webarena

    config_path = Path(webarena.__path__[0]) / "test.raw.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find test.raw.json at {config_path}. "
            "Is the webarena package installed correctly?"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    shopping_tasks: list[dict] = []
    for task in all_tasks:
        sites = task.get("sites", [])
        start_url = task.get("start_url", "")
        if "shopping" in sites and "admin" not in start_url:
            shopping_tasks.append({
                "task_id": task["task_id"],
                "intent": task.get("intent", ""),
            })

    shopping_tasks.sort(key=lambda t: t["task_id"])
    return shopping_tasks


def resolve_task_list(
    requested: list[int] | None,
    all_tasks: list[dict],
) -> list[dict]:
    """Filter *all_tasks* to those the user requested.

    Args:
        requested: CLI ``--tasks`` values (ints), or None for all.
        all_tasks: Full list from :func:`load_shopping_tasks`.

    Returns:
        Filtered (or complete) task list.
    """
    if not requested:
        return all_tasks

    valid_ids = {t["task_id"] for t in all_tasks}
    selected: list[dict] = []
    for tid in requested:
        if tid in valid_ids:
            selected.append(next(t for t in all_tasks if t["task_id"] == tid))
        else:
            logger.warning(
                "Task ID %d not in shopping task set — skipping", tid
            )

    return selected


# =========================================================================
# Main collection loop
# =========================================================================

def collect(
    runner: BrowserGymAgentRunner,
    trace_logger: TraceLogger,
    tasks: list[dict],
) -> dict:
    """Run the agent on every task, save traces, return summary stats.

    Args:
        runner: Configured agent runner (model will be lazy-loaded).
        trace_logger: Logger that auto-saves traces to disk.
        tasks: Shopping task dicts with ``task_id`` and ``intent``.

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

    for i, task_info in enumerate(tasks):
        tid = task_info["task_id"]
        intent = task_info["intent"]

        task = TaskConfig(
            task_id=f"webarena.{tid}",
            task_description=intent,
            website="webarena",
            benchmark=BENCHMARK,
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
                f"Task {i + 1}/{total}: webarena.{tid} -- {outcome} "
                f"({trace.total_steps} steps, {elapsed:.1f}s)"
            )
        except Exception:
            elapsed = time.time() - task_start
            stats["error"] += 1
            logger.exception(
                "Task %d/%d webarena.%d crashed after %.1fs",
                i + 1,
                total,
                tid,
                elapsed,
            )

    wall_time = time.time() - collection_start
    stats["wall_time_s"] = round(wall_time, 1)
    return stats


def print_summary(stats: dict, total: int) -> None:
    """Print a human-readable summary table."""
    print()
    print("=" * 55)
    print("  WebArena Trace Collection Summary")
    print("=" * 55)

    completed = (
        stats.get("success", 0)
        + stats.get("failure", 0)
        + stats.get("timeout", 0)
    )
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
        description="Collect WebArena shopping execution traces using a local LLM agent",
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
        "--max-steps",
        type=int,
        default=30,
        help="Maximum agent steps per task (default: 30)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of task IDs (e.g. 21 22 23 24 25)",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --- validate env vars ---
    missing = check_env_vars()
    if missing:
        print("ERROR: Missing required WebArena environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nSet them before running, e.g.:")
        print('  export WA_SHOPPING="http://shopping.webarena.dev:7770"')
        return 1

    print("=" * 55)
    print("  WebArena Shopping Trace Collection")
    print("=" * 55)
    print(f"  Model       : {args.model_name}")
    print(f"  Model path  : {args.model_path}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Benchmark   : {BENCHMARK}")
    print(f"  Max steps   : {args.max_steps}")
    print(f"  Temperature : {args.temperature}")
    print(f"  4-bit       : {args.load_in_4bit}")
    print()

    # --- discover shopping tasks ---
    print("Loading WebArena shopping tasks from test.raw.json...")
    all_tasks = load_shopping_tasks()
    tasks = resolve_task_list(args.tasks, all_tasks)

    if not tasks:
        print("ERROR: no tasks matched. Check --tasks values.")
        return 1

    print(f"Selected {len(tasks)} / {len(all_tasks)} shopping tasks")
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
        benchmark=BENCHMARK,
        model=args.model_name,
    )

    runner = BrowserGymAgentRunner(
        agent_config=agent_config,
        trace_logger=trace_logger,
        benchmark=BENCHMARK,
        headless=True,
    )

    # --- pre-load the model so it doesn't count toward task 1 timing ---
    print("Loading model onto GPU...")
    runner.load_model()
    print("Model loaded.\n")

    # --- run ---
    stats = collect(runner, trace_logger, tasks)
    print_summary(stats, len(tasks))

    trace_logger.print_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
