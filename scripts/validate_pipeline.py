#!/usr/bin/env python3
"""
Minimal Pipeline Validation Test

Run this first to verify all components are working.
No SPMF required - just tests the Python components.

Usage:
    python scripts/validate_pipeline.py
"""

import sys
import tempfile
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def test_imports():
    """Test all required imports."""
    print("[1/4] Testing imports...")

    try:
        from src.data_collection.trace_schema import (  # noqa: F401
            AgentTrace, TraceStep, TraceMetadata,
            ActionRecord, ObservationRecord, ReasoningRecord,
            TaskOutcome, FailureType, ActionType, SelectorType, ElementState,
            generate_trace_id, get_current_timestamp,
        )
        print("   ✓ trace_schema")
    except ImportError as e:
        print(f"   ✗ trace_schema: {e}")
        return False

    try:
        from src.preprocessing.symbolizer import TraceSymbolizer, SymbolVocabulary  # noqa: F401
        print("   ✓ symbolizer")
    except ImportError as e:
        print(f"   ✗ symbolizer: {e}")
        return False

    try:
        from src.preprocessing.k_prefix import batch_extract_prefixes, PrefixDataset  # noqa: F401
        print("   ✓ k_prefix")
    except ImportError as e:
        print(f"   ✗ k_prefix: {e}")
        return False

    try:
        from src.mining.spmf_wrapper import SPMFWrapper, SPMFConfig, RawPattern  # noqa: F401
        print("   ✓ spmf_wrapper")
    except ImportError as e:
        print(f"   ✗ spmf_wrapper: {e}")
        return False

    try:
        from src.mining.pattern_ranker import PatternRanker, RankerConfig, ScoredPattern  # noqa: F401
        print("   ✓ pattern_ranker")
    except ImportError as e:
        print(f"   ✗ pattern_ranker: {e}")
        return False

    try:
        from src.mining.signature_library import SignatureLibrary  # noqa: F401
        print("   ✓ signature_library")
    except ImportError as e:
        print(f"   ✗ signature_library: {e}")
        return False

    return True


def test_trace_creation():
    """Test creating a trace."""
    print("\n[2/4] Testing trace creation...")

    from src.data_collection.trace_schema import (
        AgentTrace, TraceStep, TraceMetadata,
        ActionRecord, ObservationRecord, ReasoningRecord,
        TaskOutcome, FailureType, ActionType, SelectorType, ElementState,
        generate_trace_id, get_current_timestamp,
    )

    trace = AgentTrace(
        metadata=TraceMetadata(
            trace_id=generate_trace_id(),
            task_id="test_001",
            task_description="Test task",
            website="shopping.webarena.dev",
            model="llama-3.2-3b",
            outcome=TaskOutcome.FAILURE,
            failure_type=FailureType.NAVIGATION,
            start_time=get_current_timestamp(),
            benchmark="test",
        )
    )

    for i in range(5):
        step = TraceStep(
            step_number=i + 1,
            reasoning=ReasoningRecord(
                raw_reasoning=f"Step {i+1}",
                intent="click",
            ),
            action=ActionRecord(
                type=ActionType.CLICK,
                selector=f"#element-{i}",
                selector_type=SelectorType.ID,
            ),
            observation=ObservationRecord(
                element_found=i < 3,
                element_state=ElementState.VISIBLE if i < 3 else ElementState.NOT_FOUND,
                http_status=200 if i < 3 else 404,
            ),
            dom_hash=f"hash_{i}",
        )
        trace.add_step(step)

    print(f"   ✓ Created trace with {trace.total_steps} steps")
    print(f"   ✓ Outcome: {trace.metadata.outcome.value}")

    return trace


def test_symbolization(trace):
    """Test symbolizing a trace."""
    print("\n[3/4] Testing symbolization...")

    from src.preprocessing.symbolizer import TraceSymbolizer

    for level in [0, 1, 2]:
        symbolizer = TraceSymbolizer(abstraction_level=level)
        symbols = symbolizer.symbolize_trace(trace)
        print(f"   Level {level}: {symbols[:3]}...")

    symbolizer = TraceSymbolizer(abstraction_level=1)
    prefix_symbols = symbolizer.symbolize_prefix(trace, k=3)
    print(f"   ✓ K=3 prefix: {prefix_symbols}")

    return prefix_symbols


def test_spmf_format():
    """Test SPMF input format generation (without running SPMF).

    Uses SymbolVocabulary directly to bypass SPMFWrapper's jar validation.
    """
    print("\n[4/4] Testing SPMF format...")

    from src.preprocessing.symbolizer import SymbolVocabulary

    sequences = [
        ["CLICK_ID_OK", "TYPE_CLASS_OK", "CLICK_ID_FAIL"],
        ["CLICK_ID_OK", "CLICK_ID_FAIL", "NAV_ERROR"],
        ["TYPE_ID_OK", "CLICK_ID_OK", "CLICK_ID_FAIL"],
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_input.txt"
        vocab = SymbolVocabulary()

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for seq in sequences:
                    if not seq:
                        continue
                    parts = []
                    for symbol in seq:
                        sid = vocab.get_or_create_id(symbol)
                        parts.append(f"{sid} -1")
                    parts.append("-2")
                    f.write(" ".join(parts) + "\n")

            print(f"   ✓ Vocabulary: {len(vocab)} symbols")
            print(f"   ✓ Sample mapping: CLICK_ID_OK -> {vocab.symbol_to_id.get('CLICK_ID_OK', 'N/A')}")

            with open(output_path) as f:
                content = f.read()
            print("   ✓ SPMF format sample:")
            for line in content.strip().split("\n")[:2]:
                print(f"      {line}")

            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False


def check_spmf():
    """Check if SPMF is available."""
    print("\n[Bonus] Checking SPMF...")

    import shutil
    import subprocess

    for spmf_path in [Path("lib/spmf.jar"), Path("/opt/spmf/spmf.jar")]:
        if spmf_path.exists():
            print(f"   ✓ SPMF found at {spmf_path}")
            try:
                subprocess.run(
                    ["java", "-jar", str(spmf_path), "run"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                print("   ✓ SPMF is runnable")
            except Exception as e:
                print(f"   ⚠ SPMF may have issues: {e}")
            return

    print("   ⚠ SPMF not found at lib/spmf.jar or /opt/spmf/spmf.jar")
    print("   Download: mkdir -p lib && curl -L -o lib/spmf.jar https://www.philippe-fournier-viger.com/spmf/SPMF.jar")

    java_path = shutil.which("java")
    if java_path:
        print(f"   ✓ Java found at {java_path}")
    else:
        print("   ⚠ Java not found on PATH (required for SPMF)")


def main():
    print("=" * 60)
    print("Pipeline Validation Test")
    print("=" * 60)

    if not test_imports():
        print("\n✗ Import test failed. Fix imports first.")
        return 1

    trace = test_trace_creation()
    if trace is None:
        print("\n✗ Trace creation failed.")
        return 1

    symbols = test_symbolization(trace)
    if symbols is None:
        print("\n✗ Symbolization failed.")
        return 1

    if not test_spmf_format():
        print("\n✗ SPMF format test failed.")
        return 1

    check_spmf()

    print("\n" + "=" * 60)
    print("✓ All validation tests passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download SPMF if not present: mkdir -p lib && curl -L -o lib/spmf.jar https://www.philippe-fournier-viger.com/spmf/SPMF.jar")
    print("2. Run full pipeline: python scripts/run_offline_pipeline.py --mode synthetic")

    return 0


if __name__ == "__main__":
    exit(main())
