#!/usr/bin/env python3
"""
Reprocess Traces — Re-parse action fields in existing trace JSON files.

Walks all JSON files under a trace directory, re-parses each step's
raw_action using the fixed ActionParser, and overwrites the action
fields (type, selector, selector_type, value, bid) in place.  All
other trace data (metadata, observations, reasoning) is left untouched.

Safe to run multiple times (idempotent): the parser is deterministic
and raw_action is never modified.

Usage:
    python scripts/reprocess_traces.py --trace-dir data/raw_traces
    python scripts/reprocess_traces.py --trace-dir data/raw_traces --dry-run

Author: Sergio Talavera
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_collection.agent_runner import ActionParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def reparse_action(action_dict: dict) -> dict:
    """Re-parse a single action dict using the fixed ActionParser.

    Only the fields produced by the parser are overwritten; raw_action
    is preserved verbatim.

    Args:
        action_dict: Original action dict from a trace step.

    Returns:
        Updated action dict.
    """
    raw = action_dict.get("raw_action")
    if not raw:
        return action_dict

    record = ActionParser.parse(raw)
    action_dict["type"] = record.type.value
    action_dict["selector"] = record.selector
    action_dict["selector_type"] = record.selector_type.value if record.selector_type else None
    action_dict["value"] = record.value
    action_dict["bid"] = record.bid
    return action_dict


def process_file(filepath: Path, dry_run: bool) -> dict:
    """Re-parse every step's action in a single trace JSON file.

    Args:
        filepath: Path to the JSON trace file.
        dry_run: If True, don't write changes to disk.

    Returns:
        Dict of per-file stats (actions_total, changed, before/after
        type and selector_type counts).
    """
    stats: dict = {
        "actions_total": 0,
        "actions_changed": 0,
        "before_types": defaultdict(int),
        "after_types": defaultdict(int),
        "before_selector_types": defaultdict(int),
        "after_selector_types": defaultdict(int),
    }

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid JSON: %s", filepath)
            return stats

    steps = data.get("steps", [])
    if not steps:
        return stats

    changed = False
    for step in steps:
        action = step.get("action")
        if not action or "raw_action" not in action:
            continue

        stats["actions_total"] += 1
        old_type = action.get("type", "unknown")
        old_sel_type = action.get("selector_type")
        stats["before_types"][old_type] += 1
        stats["before_selector_types"][old_sel_type or "none"] += 1

        old_snapshot = {
            "type": action.get("type"),
            "selector": action.get("selector"),
            "selector_type": action.get("selector_type"),
            "value": action.get("value"),
            "bid": action.get("bid"),
        }

        reparse_action(action)

        new_snapshot = {
            "type": action.get("type"),
            "selector": action.get("selector"),
            "selector_type": action.get("selector_type"),
            "value": action.get("value"),
            "bid": action.get("bid"),
        }

        stats["after_types"][action.get("type", "unknown")] += 1
        stats["after_selector_types"][action.get("selector_type") or "none"] += 1

        if old_snapshot != new_snapshot:
            stats["actions_changed"] += 1
            changed = True

    if changed and not dry_run:
        _atomic_write(filepath, data)

    return stats


def _atomic_write(filepath: Path, data: dict) -> None:
    """Write JSON via a temp file + rename for crash safety."""
    dir_path = filepath.parent
    fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(filepath))
    except BaseException:
        os.unlink(tmp_path)
        raise


def collect_json_files(trace_dir: Path) -> list[Path]:
    """Recursively find all .json files under trace_dir."""
    return sorted(trace_dir.rglob("*.json"))


def merge_stats(total: dict, file_stats: dict) -> None:
    """Accumulate per-file stats into the running total."""
    total["files_processed"] += 1
    total["actions_total"] += file_stats["actions_total"]
    total["actions_changed"] += file_stats["actions_changed"]
    for key in ("before_types", "after_types", "before_selector_types", "after_selector_types"):
        for k, v in file_stats[key].items():
            total[key][k] += v


def print_summary(stats: dict, dry_run: bool) -> None:
    """Print a human-readable summary table."""
    tag = " (DRY RUN — no files written)" if dry_run else ""
    print()
    print("=" * 60)
    print(f"  Trace Reprocessing Summary{tag}")
    print("=" * 60)
    print(f"  Files processed     : {stats['files_processed']}")
    print(f"  Total actions       : {stats['actions_total']}")
    print(f"  Actions changed     : {stats['actions_changed']}")
    if stats["actions_total"] > 0:
        pct = stats["actions_changed"] / stats["actions_total"] * 100
        print(f"  Change rate         : {pct:.1f}%")

    print()
    print("  Action types  BEFORE -> AFTER:")
    all_types = sorted(set(stats["before_types"]) | set(stats["after_types"]))
    for t in all_types:
        b = stats["before_types"].get(t, 0)
        a = stats["after_types"].get(t, 0)
        delta = a - b
        sign = "+" if delta > 0 else ""
        print(f"    {t:20s}  {b:5d} -> {a:5d}  ({sign}{delta})")

    print()
    print("  Selector types  BEFORE -> AFTER:")
    all_sel = sorted(set(stats["before_selector_types"]) | set(stats["after_selector_types"]))
    for s in all_sel:
        b = stats["before_selector_types"].get(s, 0)
        a = stats["after_selector_types"].get(s, 0)
        delta = a - b
        sign = "+" if delta > 0 else ""
        print(f"    {s:20s}  {b:5d} -> {a:5d}  ({sign}{delta})")

    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-parse action fields in existing trace JSON files",
    )
    parser.add_argument(
        "--trace-dir",
        default="data/raw_traces",
        help="Root directory containing trace JSON files (default: data/raw_traces)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing to disk",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_dir = Path(args.trace_dir)

    if not trace_dir.is_dir():
        logger.error("Trace directory not found: %s", trace_dir)
        return 1

    json_files = collect_json_files(trace_dir)
    if not json_files:
        logger.info("No JSON files found under %s", trace_dir)
        return 0

    logger.info(
        "Found %d JSON files under %s%s",
        len(json_files),
        trace_dir,
        " (dry run)" if args.dry_run else "",
    )

    total_stats: dict = {
        "files_processed": 0,
        "actions_total": 0,
        "actions_changed": 0,
        "before_types": defaultdict(int),
        "after_types": defaultdict(int),
        "before_selector_types": defaultdict(int),
        "after_selector_types": defaultdict(int),
    }

    for filepath in json_files:
        file_stats = process_file(filepath, dry_run=args.dry_run)
        merge_stats(total_stats, file_stats)

    print_summary(total_stats, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
