# Running the Phase 2 Offline Training Pipeline

## Overview

The offline training pipeline processes raw traces into a ranked pattern library:

```
Raw Traces → Symbolize → K-Prefix Extract → BIDE Mine → Rank → Pattern Library
```

## Prerequisites

### 1. SPMF Setup

Download SPMF (Java-based pattern mining):

```bash
# Create lib directory
mkdir -p lib

# Download SPMF
wget -O lib/spmf.jar https://www.philippe-fournier-viger.com/spmf/SPMF.jar

# Verify Java is available
java -version  # Needs Java 8+
```

### 2. Data Options

**Option A: Real Traces (WebArena/BrowserGym)**
- Requires WebArena environment running
- Will take hours to collect 1000+ traces

**Option B: Synthetic Traces (Testing)**
- Generate fake traces for pipeline validation
- Runs in seconds
- Use this first to verify everything works

---

## Option B: Test with Synthetic Data First

Create a test script to validate the pipeline before running on real data:

```python
# scripts/test_offline_pipeline.py

"""
Test the offline training pipeline with synthetic data.
Validates all components work together before running on real traces.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_collection.trace_schema import (
    AgentTrace, TraceStep, TraceMetadata,
    ActionRecord, ObservationRecord, ReasoningRecord,
    TaskOutcome, FailureType, ActionType, SelectorType, ElementState,
    generate_trace_id, get_current_timestamp,
)
from preprocessing.symbolizer import TraceSymbolizer
from mining.spmf_wrapper import SPMFWrapper
from mining.pattern_ranker import PatternRanker
from mining.signature_library import SignatureLibrary


def generate_synthetic_traces(n_traces: int = 100) -> list[AgentTrace]:
    """Generate synthetic traces for testing."""
    import random
    random.seed(42)
    
    websites = ["shopping.webarena.dev", "gitlab.webarena.dev", "reddit.webarena.dev"]
    models = ["llama-3.2-3b", "qwen-2.5-7b"]
    
    # Define some "failure patterns" that we'll inject
    failure_patterns = [
        # Pattern 1: Click not found → Click not found → Type fail
        [("click", "not_found"), ("click", "not_found"), ("type", "fail")],
        # Pattern 2: Navigate error → Click stale
        [("navigate", "error"), ("click", "stale")],
        # Pattern 3: Repeated retries
        [("click", "ok"), ("click", "not_found"), ("click", "not_found")],
    ]
    
    traces = []
    
    for i in range(n_traces):
        # 85% failure rate (realistic for WebArena)
        is_failure = random.random() < 0.85
        
        trace = AgentTrace(
            metadata=TraceMetadata(
                trace_id=generate_trace_id(),
                task_id=f"task_{i:04d}",
                task_description=f"Synthetic task {i}",
                website=random.choice(websites),
                model=random.choice(models),
                outcome=TaskOutcome.FAILURE if is_failure else TaskOutcome.SUCCESS,
                failure_type=random.choice([FailureType.NAVIGATION, FailureType.VALIDATION, FailureType.RECOVERY]) if is_failure else None,
                start_time=get_current_timestamp(),
                benchmark="synthetic",
            )
        )
        
        # Generate steps
        n_steps = random.randint(5, 20)
        
        # If failure, inject a failure pattern somewhere in the first 10 steps
        if is_failure and random.random() < 0.7:  # 70% of failures have pattern
            pattern = random.choice(failure_patterns)
            pattern_start = random.randint(0, min(5, n_steps - len(pattern)))
        else:
            pattern = None
            pattern_start = -1
        
        for step_num in range(n_steps):
            # Check if this step is part of injected pattern
            if pattern and pattern_start <= step_num < pattern_start + len(pattern):
                pattern_idx = step_num - pattern_start
                action_type, outcome = pattern[pattern_idx]
                element_found = outcome not in ["not_found", "stale", "error"]
                element_state = {
                    "ok": ElementState.VISIBLE,
                    "not_found": ElementState.NOT_FOUND,
                    "stale": ElementState.STALE,
                    "fail": ElementState.NOT_INTERACTABLE,
                    "error": ElementState.HIDDEN,
                }[outcome]
            else:
                # Random step
                action_type = random.choice(["click", "type", "navigate", "scroll"])
                element_found = random.random() < 0.8
                element_state = ElementState.VISIBLE if element_found else ElementState.NOT_FOUND
            
            step = TraceStep(
                step_number=step_num + 1,
                reasoning=ReasoningRecord(
                    raw_reasoning=f"Step {step_num + 1}: Attempting {action_type}",
                    intent=action_type,
                    keywords=["retry"] if "not_found" in str(element_state) else [],
                ),
                action=ActionRecord(
                    type=ActionType(action_type) if action_type in ["click", "type", "navigate", "scroll"] else ActionType.CLICK,
                    selector=f"#element-{random.randint(1, 100)}",
                    selector_type=random.choice([SelectorType.ID, SelectorType.CLASS, SelectorType.XPATH]),
                ),
                observation=ObservationRecord(
                    element_found=element_found,
                    element_state=element_state,
                    http_status=200 if element_found else 404,
                ),
                dom_hash=f"hash_{random.randint(1000, 9999)}",
                timestamp=get_current_timestamp(),
                prompt_tokens=random.randint(1000, 2000),
                completion_tokens=random.randint(50, 150),
            )
            trace.add_step(step)
        
        traces.append(trace)
    
    return traces


def run_offline_pipeline(
    traces: list[AgentTrace],
    k: int = 10,
    abstraction_level: int = 1,
    min_support: float = 0.05,
    output_dir: Path = Path("data/pipeline_test"),
) -> SignatureLibrary:
    """Run the complete offline training pipeline."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("OFFLINE TRAINING PIPELINE")
    print("="*60)
    print(f"Traces: {len(traces)}")
    print(f"K: {k}")
    print(f"Abstraction Level: {abstraction_level}")
    print(f"Min Support: {min_support}")
    print()
    
    # Step 1: Symbolize traces
    print("[1/5] Symbolizing traces...")
    symbolizer = TraceSymbolizer(abstraction_level=abstraction_level)
    
    symbolized_traces = []
    for trace in traces:
        symbols = symbolizer.symbolize_prefix(trace, k)
        symbolized_traces.append({
            'trace': trace,
            'symbols': symbols,
            'label': 1 if trace.metadata.outcome == TaskOutcome.FAILURE else 0,
            'website': trace.metadata.website,
        })
    
    print(f"   Symbolized {len(symbolized_traces)} traces")
    
    # Show sample
    sample = symbolized_traces[0]
    print(f"   Sample: {sample['symbols'][:5]}...")
    print()
    
    # Step 2: Prepare sequences for SPMF
    print("[2/5] Preparing SPMF input...")
    sequences = [st['symbols'] for st in symbolized_traces]
    
    spmf = SPMFWrapper(spmf_jar_path=Path("lib/spmf.jar"))
    spmf_input = output_dir / "spmf_input.txt"
    symbol_map = spmf.prepare_input(sequences, spmf_input)
    
    print(f"   Created {spmf_input}")
    print(f"   Vocabulary size: {len(symbol_map)}")
    print()
    
    # Step 3: Run BIDE
    print("[3/5] Running BIDE pattern mining...")
    spmf_output = output_dir / "spmf_output.txt"
    
    try:
        spmf.run_bide(spmf_input, spmf_output, min_support)
        print(f"   Output: {spmf_output}")
    except Exception as e:
        print(f"   ERROR: {e}")
        print("   Make sure SPMF is installed: lib/spmf.jar")
        return None
    
    # Step 4: Parse patterns and compute metrics
    print("[4/5] Ranking patterns...")
    id_to_symbol = {v: k for k, v in symbol_map.items()}
    patterns = spmf.parse_output(spmf_output, id_to_symbol)
    
    print(f"   Found {len(patterns)} raw patterns")
    
    # Compute precision and coverage
    ranker = PatternRanker(min_precision=0.5, min_sites=2)
    ranked_patterns = ranker.rank_patterns(patterns, symbolized_traces)
    
    print(f"   After filtering: {len(ranked_patterns)} patterns")
    print()
    
    # Step 5: Build signature library
    print("[5/5] Building signature library...")
    library = SignatureLibrary()
    
    for pattern in ranked_patterns:
        library.add_pattern(pattern)
    
    library_path = output_dir / "pattern_library.json"
    library.save(library_path)
    
    print(f"   Saved to {library_path}")
    print()
    
    # Summary
    print("="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total patterns: {len(ranked_patterns)}")
    
    if ranked_patterns:
        print(f"Top 5 patterns:")
        for i, p in enumerate(ranked_patterns[:5]):
            print(f"   {i+1}. {' → '.join(p.symbols[:4])}... (precision={p.precision:.2f}, coverage={p.coverage})")
    
    return library


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test offline training pipeline")
    parser.add_argument("--n-traces", type=int, default=100, help="Number of synthetic traces")
    parser.add_argument("--k", type=int, default=10, help="K-prefix length")
    parser.add_argument("--level", type=int, default=1, help="Abstraction level (0, 1, 2)")
    parser.add_argument("--min-support", type=float, default=0.05, help="Minimum support")
    parser.add_argument("--output-dir", type=str, default="data/pipeline_test", help="Output directory")
    
    args = parser.parse_args()
    
    # Generate synthetic traces
    print("Generating synthetic traces...")
    traces = generate_synthetic_traces(args.n_traces)
    
    failures = sum(1 for t in traces if t.metadata.outcome == TaskOutcome.FAILURE)
    print(f"Generated {len(traces)} traces ({failures} failures, {len(traces)-failures} successes)")
    print()
    
    # Run pipeline
    library = run_offline_pipeline(
        traces,
        k=args.k,
        abstraction_level=args.level,
        min_support=args.min_support,
        output_dir=Path(args.output_dir),
    )
    
    if library:
        print("\n✓ Pipeline test successful!")
    else:
        print("\n✗ Pipeline test failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
```

---

## Option A: Run on Real Traces

Once you've validated with synthetic data, run on real WebArena traces:

```python
# scripts/run_offline_training.py

"""
Run offline training pipeline on real WebArena/BrowserGym traces.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_collection.trace_logger import TraceLogger
from preprocessing.symbolizer import TraceSymbolizer
from mining.spmf_wrapper import SPMFWrapper
from mining.pattern_ranker import PatternRanker
from mining.signature_library import SignatureLibrary


def load_traces(trace_dir: Path) -> list:
    """Load all traces from directory."""
    logger = TraceLogger(base_dir=str(trace_dir.parent.parent))
    traces = list(logger.iter_traces(trace_dir))
    return traces


def run_training(
    trace_dirs: list[Path],
    output_dir: Path,
    k: int = 10,
    abstraction_level: int = 1,
    min_support: float = 0.03,
):
    """Run full offline training."""
    
    # Load all traces
    print("Loading traces...")
    all_traces = []
    for trace_dir in trace_dirs:
        traces = load_traces(trace_dir)
        all_traces.extend(traces)
        print(f"  {trace_dir}: {len(traces)} traces")
    
    print(f"Total: {len(all_traces)} traces")
    
    # Run pipeline (same as test script)
    # ... (use run_offline_pipeline from above)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dirs", nargs="+", required=True, help="Directories with traces")
    parser.add_argument("--output-dir", default="data/trained", help="Output directory")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--min-support", type=float, default=0.03)
    
    args = parser.parse_args()
    
    run_training(
        trace_dirs=[Path(d) for d in args.trace_dirs],
        output_dir=Path(args.output_dir),
        k=args.k,
        abstraction_level=args.level,
        min_support=args.min_support,
    )


if __name__ == "__main__":
    main()
```

---

## Cursor Prompt for Running Pipeline

Copy this prompt into Cursor to get help executing:

```
I need to run the Phase 2 offline training pipeline for my thesis project.

Current state:
- All Phase 1 & 2 components are implemented
- I have SPMF downloaded at lib/spmf.jar (or need to download it)
- I want to test with synthetic data first

Pipeline steps:
1. Generate/load traces
2. Symbolize traces (abstraction_level=1)
3. Extract K-prefixes (K=10)
4. Run BIDE via SPMF (min_support=0.05)
5. Rank patterns by precision × coverage
6. Filter to patterns on ≥2 sites
7. Save pattern library

Please help me:
1. Create scripts/run_offline_pipeline.py that orchestrates this
2. Verify all imports work correctly
3. Run with synthetic data first (100 traces)
4. Debug any issues

The script should:
- Use existing components from src/preprocessing/ and src/mining/
- Print progress at each step
- Save the pattern library to data/trained/pattern_library.json
- Handle errors gracefully (especially SPMF issues)
```

---

## Quick Start Commands

```bash
# 1. Setup SPMF
mkdir -p lib
wget -O lib/spmf.jar https://www.philippe-fournier-viger.com/spmf/SPMF.jar

# 2. Test with synthetic data
python scripts/test_offline_pipeline.py --n-traces 100 --k 10

# 3. If successful, run on real traces (when available)
python scripts/run_offline_training.py \
    --trace-dirs data/raw_traces/webarena/llama data/raw_traces/webarena/qwen \
    --output-dir data/trained \
    --k 10 \
    --min-support 0.03
```

---

## Expected Output

```
==============================================================
OFFLINE TRAINING PIPELINE
==============================================================
Traces: 100
K: 10
Abstraction Level: 1
Min Support: 0.05

[1/5] Symbolizing traces...
   Symbolized 100 traces
   Sample: ['CLICK_ID_VISIBLE_OK', 'TYPE_CLASS_VISIBLE_OK', ...]

[2/5] Preparing SPMF input...
   Created data/pipeline_test/spmf_input.txt
   Vocabulary size: 24

[3/5] Running BIDE pattern mining...
   Output: data/pipeline_test/spmf_output.txt

[4/5] Ranking patterns...
   Found 47 raw patterns
   After filtering: 12 patterns

[5/5] Building signature library...
   Saved to data/pipeline_test/pattern_library.json

==============================================================
PIPELINE COMPLETE
==============================================================
Total patterns: 12
Top 5 patterns:
   1. CLICK_ID_NOT_FOUND → CLICK_ID_NOT_FOUND... (precision=0.89, coverage=3)
   2. NAV_ERROR → CLICK_STALE... (precision=0.82, coverage=2)
   ...

✓ Pipeline test successful!
```

---

## Troubleshooting

### SPMF not found
```bash
# Check Java
java -version

# Download SPMF manually
curl -L -o lib/spmf.jar "https://www.philippe-fournier-viger.com/spmf/SPMF.jar"
```

### Import errors
```bash
# Make sure you're in project root
cd /path/to/thesis-project

# Run with explicit path
PYTHONPATH=src python scripts/test_offline_pipeline.py
```

### No patterns found
- Lower min_support (try 0.01)
- Check if traces have enough steps (need at least K steps)
- Verify failure rate is realistic (~85%)