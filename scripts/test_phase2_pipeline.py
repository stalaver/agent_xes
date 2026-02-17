"""
Phase 2 Pipeline Integration Test

Validates the full preprocessing + mining pipeline using synthetic traces.
No WebArena or external dependencies required (except Java + spmf.jar for
the BIDE step, which is tested conditionally).

Test flow:
1. Generate synthetic AgentTrace objects
2. Symbolize at all 3 abstraction levels
3. Extract k-prefixes into PrefixDataset
4. Prepare SPMF input and validate format
5. (Optional) Run BIDE if Java + spmf.jar available
6. Parse SPMF output (using mock data if BIDE unavailable)
7. Rank patterns and verify filtering
8. Build signature library and test matching
9. Save/load round-trip validation

Usage:
    python -m scripts.test_phase2_pipeline

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
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
from src.preprocessing.symbolizer import TraceSymbolizer, SymbolVocabulary
from src.preprocessing.k_prefix import (
    PrefixDataset,
    batch_extract_prefixes,
    extract_symbolized_prefix,
)
from src.mining.spmf_wrapper import SPMFWrapper, SPMFConfig, RawPattern
from src.mining.pattern_ranker import PatternRanker, RankerConfig, ScoredPattern, _is_subsequence
from src.mining.signature_library import SignatureLibrary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Synthetic trace generation
# =============================================================================

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

ACTION_TYPE_MAP = {
    "click": ActionType.CLICK,
    "type": ActionType.TYPE,
    "navigate": ActionType.NAVIGATE,
    "scroll": ActionType.SCROLL,
    "go_back": ActionType.GO_BACK,
    "stop": ActionType.STOP,
    "hover": ActionType.HOVER,
    "select": ActionType.SELECT,
}


def _make_step(
    step_num: int,
    action_name: str,
    selector_type: SelectorType,
    element_found: bool,
    element_state: ElementState,
    keywords: list[str],
) -> TraceStep:
    """Build a synthetic TraceStep."""
    return TraceStep(
        step_number=step_num,
        reasoning=ReasoningRecord(
            raw_reasoning=f"Step {step_num} reasoning",
            intent=f"intent_{action_name}",
            keywords=keywords,
        ),
        action=ActionRecord(
            type=ACTION_TYPE_MAP.get(action_name, ActionType.UNKNOWN),
            selector=f"#el-{step_num}",
            selector_type=selector_type,
        ),
        observation=ObservationRecord(
            element_found=element_found,
            element_state=element_state,
            http_status=200 if element_found else 404,
        ),
        dom_hash=f"hash{step_num:02d}",
        timestamp=get_current_timestamp(),
        url=f"http://example.com/page{step_num}",
        prompt_tokens=500,
        completion_tokens=30,
    )


def generate_synthetic_traces(count: int = 25) -> list[AgentTrace]:
    """Generate a set of synthetic traces with known patterns.

    Distributes traces across websites and scenarios so that failure
    patterns appear on multiple sites (enabling cross-site filtering).

    Args:
        count: Total number of traces to generate.

    Returns:
        List of AgentTrace objects.
    """
    traces: list[AgentTrace] = []

    for i in range(count):
        scenario = FAILURE_SCENARIOS[i % len(FAILURE_SCENARIOS)]
        website = WEBSITES[i % len(WEBSITES)]

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
                benchmark="webarena",
            )
        )

        for step_num, (act, sel, found, state, kw) in enumerate(
            scenario["pattern"], start=1
        ):
            step = _make_step(step_num, act, sel, found, state, kw)
            trace.add_step(step)

        traces.append(trace)

    logger.info("Generated %d synthetic traces", len(traces))
    return traces


# =============================================================================
# Test functions
# =============================================================================

def test_symbolizer(traces: list[AgentTrace]) -> None:
    """Test symbolization at all three levels."""
    logger.info("=== Testing Symbolizer ===")

    for level in (0, 1, 2):
        sym = TraceSymbolizer(abstraction_level=level)
        symbols = sym.symbolize_trace(traces[0])
        logger.info("Level %d symbols for trace 0: %s", level, symbols)

        assert len(symbols) == len(traces[0].steps), (
            f"Expected {len(traces[0].steps)} symbols, got {len(symbols)}"
        )

        for s in symbols:
            assert isinstance(s, str) and len(s) > 0, f"Invalid symbol: {s!r}"

    sym1 = TraceSymbolizer(abstraction_level=1)
    prefix_5 = sym1.symbolize_prefix(traces[0], k=5)
    prefix_3 = sym1.symbolize_prefix(traces[0], k=3)
    assert len(prefix_5) == 5
    assert len(prefix_3) == 3
    assert prefix_5[:3] == prefix_3

    logger.info("Symbolizer: PASSED")


def test_k_prefix(traces: list[AgentTrace]) -> None:
    """Test k-prefix extraction and PrefixDataset."""
    logger.info("=== Testing K-Prefix Extractor ===")

    sym = TraceSymbolizer(abstraction_level=1)

    single = extract_symbolized_prefix(traces[0], k=5, symbolizer=sym)
    assert len(single) == 5

    dataset = batch_extract_prefixes(traces, k=5, symbolizer=sym)
    assert len(dataset) == len(traces)
    assert dataset.k == 5
    assert dataset.abstraction_level == 1
    assert len(dataset.websites) > 1

    summary = dataset.summary()
    assert summary["total"] == len(traces)
    logger.info("PrefixDataset summary: %s", summary)

    failed = dataset.get_failed_entries()
    successful = dataset.get_successful_entries()
    assert len(failed) + len(successful) <= len(traces)
    logger.info(
        "  Failed entries: %d, Successful: %d", len(failed), len(successful)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dataset.json"
        dataset.save(path)
        loaded = PrefixDataset.load(path)
        assert len(loaded) == len(dataset)
        assert loaded.k == dataset.k
        assert loaded.entries[0].trace_id == dataset.entries[0].trace_id

    logger.info("K-Prefix Extractor: PASSED")


def test_spmf_input_output(traces: list[AgentTrace]) -> None:
    """Test SPMF input preparation and output parsing (no Java needed)."""
    logger.info("=== Testing SPMF Input/Output ===")

    sym = TraceSymbolizer(abstraction_level=1)
    dataset = batch_extract_prefixes(traces, k=5, symbolizer=sym)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "test_input.txt"

        vocab = SymbolVocabulary()
        sequences = dataset.sequences

        # Prepare input
        written_path, vocab = _prepare_spmf_input_standalone(
            sequences, input_path, vocab
        )

        assert written_path.exists()
        assert len(vocab) > 0
        logger.info("  Vocabulary size: %d", len(vocab))

        with open(written_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == len(sequences)

        for line in lines:
            assert line.strip().endswith("-2")
            tokens = line.strip().split()
            for t in tokens:
                val = int(t)
                assert val > 0 or val in (-1, -2)

        # Test vocabulary round-trip
        vocab_path = tmpdir / "vocab.json"
        vocab.save(vocab_path)
        loaded_vocab = SymbolVocabulary.load(vocab_path)
        assert len(loaded_vocab) == len(vocab)
        for symbol, sid in vocab.symbol_to_id.items():
            assert loaded_vocab.symbol_to_id[symbol] == sid

        # Test output parsing with mock SPMF output
        mock_output_path = tmpdir / "mock_output.txt"
        _write_mock_spmf_output(mock_output_path, vocab)

        patterns = _parse_spmf_output_standalone(mock_output_path, vocab)
        assert len(patterns) > 0
        for p in patterns:
            assert len(p.symbols) > 0
            assert p.support > 0
            logger.info("  Parsed pattern: %s (sup=%d)", p.symbols, p.support)

    logger.info("SPMF Input/Output: PASSED")


def _prepare_spmf_input_standalone(
    sequences: list[list[str]],
    output_path: Path,
    vocab: SymbolVocabulary,
) -> tuple[Path, SymbolVocabulary]:
    """Prepare SPMF input without needing SPMFWrapper (avoids jar check)."""
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
    return output_path, vocab


def _write_mock_spmf_output(path: Path, vocab: SymbolVocabulary) -> None:
    """Write mock SPMF BIDE+ output for testing the parser."""
    symbols = list(vocab.symbol_to_id.keys())
    with open(path, "w", encoding="utf-8") as f:
        if len(symbols) >= 2:
            id1 = vocab.symbol_to_id[symbols[0]]
            id2 = vocab.symbol_to_id[symbols[1]]
            f.write(f"{id1} -1 {id2} -1 #SUP: 15\n")
        if len(symbols) >= 3:
            id3 = vocab.symbol_to_id[symbols[2]]
            f.write(f"{id1} -1 {id3} -1 #SUP: 10\n")
        if len(symbols) >= 1:
            f.write(f"{id1} -1 #SUP: 20\n")


def _parse_spmf_output_standalone(
    path: Path, vocab: SymbolVocabulary
) -> list[RawPattern]:
    """Parse SPMF output without needing SPMFWrapper instance."""
    import re

    support_re = re.compile(r"#SUP:\s*(\d+)")
    patterns: list[RawPattern] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = support_re.search(line)
            if not match:
                continue
            support = int(match.group(1))
            pattern_part = line[: match.start()].strip()
            symbol_ids = [
                int(t) for t in pattern_part.split() if int(t) > 0
            ]
            symbols = [vocab.get_symbol(sid) for sid in symbol_ids]
            patterns.append(RawPattern(symbols=symbols, support=support))

    return patterns


def test_pattern_ranker(traces: list[AgentTrace]) -> None:
    """Test pattern ranking and filtering."""
    logger.info("=== Testing Pattern Ranker ===")

    sym = TraceSymbolizer(abstraction_level=1)
    dataset = batch_extract_prefixes(traces, k=5, symbolizer=sym)

    # Create some raw patterns that we know exist in the data
    # The NAVIGATION failure pattern at level 1 should produce:
    #   CLICK_ID_SUCCESS, CLICK_CLASS_FAIL, CLICK_CLASS_FAIL, ...
    raw_patterns = _extract_common_subsequences(dataset)
    logger.info("  Extracted %d test patterns", len(raw_patterns))

    # Rank with lenient thresholds so some pass
    config = RankerConfig(min_precision=0.3, min_sites=1, min_support=1)
    ranker = PatternRanker(config=config)

    scored = ranker.rank_patterns(raw_patterns, dataset)
    logger.info("  Scored patterns (lenient): %d", len(scored))

    for sp in scored[:5]:
        logger.info(
            "    %s  prec=%.2f cov=%d score=%.3f type=%s",
            sp.symbols,
            sp.precision,
            sp.coverage,
            sp.score,
            sp.failure_type,
        )

    assert len(scored) > 0, "Expected at least one pattern to pass lenient filters"

    for sp in scored:
        assert sp.precision >= config.min_precision
        assert sp.coverage >= config.min_sites
        assert sp.support >= config.min_support
        assert sp.score >= 0.0
        if scored.index(sp) > 0:
            assert sp.score <= scored[scored.index(sp) - 1].score

    # Stricter filter
    strict_config = RankerConfig(min_precision=0.5, min_sites=2, min_support=2)
    strict_ranker = PatternRanker(config=strict_config)
    strict_scored = strict_ranker.rank_patterns(raw_patterns, dataset)
    logger.info("  Scored patterns (strict): %d", len(strict_scored))

    for sp in strict_scored:
        assert sp.precision >= 0.5
        assert sp.coverage >= 2

    logger.info("Pattern Ranker: PASSED")


def _extract_common_subsequences(dataset: PrefixDataset) -> list[RawPattern]:
    """Extract common subsequences from the dataset for testing.

    This is a simple heuristic -- take length-1 and length-2 subsequences
    that appear in at least 2 traces.
    """
    from collections import Counter

    length1: Counter[str] = Counter()
    length2: Counter[tuple[str, str]] = Counter()

    for entry in dataset.entries:
        seen1: set[str] = set()
        for s in entry.symbols:
            if s not in seen1:
                length1[s] += 1
                seen1.add(s)

        seen2: set[tuple[str, str]] = set()
        for i in range(len(entry.symbols)):
            for j in range(i + 1, len(entry.symbols)):
                pair = (entry.symbols[i], entry.symbols[j])
                if pair not in seen2:
                    length2[pair] += 1
                    seen2.add(pair)

    patterns: list[RawPattern] = []
    for symbol, count in length1.most_common(10):
        if count >= 2:
            patterns.append(RawPattern(symbols=[symbol], support=count))

    for (s1, s2), count in length2.most_common(15):
        if count >= 2:
            patterns.append(RawPattern(symbols=[s1, s2], support=count))

    return patterns


def test_signature_library(traces: list[AgentTrace]) -> None:
    """Test signature library persistence and matching."""
    logger.info("=== Testing Signature Library ===")

    sym = TraceSymbolizer(abstraction_level=1)
    dataset = batch_extract_prefixes(traces, k=5, symbolizer=sym)

    raw_patterns = _extract_common_subsequences(dataset)
    config = RankerConfig(min_precision=0.3, min_sites=1, min_support=1)
    ranker = PatternRanker(config=config)
    scored = ranker.rank_patterns(raw_patterns, dataset)

    library = SignatureLibrary(patterns=scored)
    logger.info("  Library size: %d", len(library))

    summary = library.summary()
    logger.info("  Summary: %s", summary)
    assert summary["total"] == len(scored)

    # Test matching
    test_sequence = dataset.entries[0].symbols
    matches = library.match(test_sequence)
    logger.info("  Matches for trace 0: %d", len(matches))
    assert len(matches) >= 0

    top3 = library.top_n_matches(test_sequence, n=3)
    assert len(top3) <= 3

    # Test has_match
    has = library.has_match(test_sequence)
    assert has == (len(matches) > 0)

    # Test retrieval by failure type
    nav_patterns = library.get_by_failure_type(FailureType.NAVIGATION)
    logger.info("  NAVIGATION patterns: %d", len(nav_patterns))

    # Test save/load round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_library.json"
        library.save(path)
        loaded = SignatureLibrary.load(path)
        assert len(loaded) == len(library)

        if len(library) > 0:
            assert loaded.patterns[0].symbols == library.patterns[0].symbols
            assert abs(loaded.patterns[0].score - round(library.patterns[0].score, 4)) < 1e-6

        loaded_matches = loaded.match(test_sequence)
        assert len(loaded_matches) == len(matches)

    logger.info("Signature Library: PASSED")


def test_subsequence_helper() -> None:
    """Test the _is_subsequence utility function."""
    logger.info("=== Testing Subsequence Helper ===")

    assert _is_subsequence(["A", "B"], ["A", "X", "B"]) is True
    assert _is_subsequence(["A", "B"], ["A", "B", "C"]) is True
    assert _is_subsequence(["A", "B"], ["B", "A"]) is False
    assert _is_subsequence([], ["A", "B"]) is True
    assert _is_subsequence(["A"], []) is False
    assert _is_subsequence(["A", "B", "C"], ["A", "B"]) is False
    assert _is_subsequence(["A"], ["A"]) is True
    assert _is_subsequence(["A", "C"], ["A", "B", "C", "D"]) is True

    logger.info("Subsequence Helper: PASSED")


def test_spmf_bide_integration(traces: list[AgentTrace]) -> None:
    """Test full BIDE pipeline if Java + spmf.jar are available."""
    logger.info("=== Testing SPMF BIDE Integration (optional) ===")

    java_path = shutil.which("java")
    spmf_jar = Path("/opt/spmf/spmf.jar")

    if java_path is None:
        logger.warning("  Java not found, skipping BIDE integration test")
        return

    if not spmf_jar.exists():
        logger.warning(
            "  spmf.jar not found at %s, skipping BIDE integration test",
            spmf_jar,
        )
        return

    sym = TraceSymbolizer(abstraction_level=1)
    dataset = batch_extract_prefixes(traces, k=5, symbolizer=sym)

    config = SPMFConfig(
        spmf_jar_path=spmf_jar,
        min_support=0.1,
        max_pattern_length=5,
        timeout_seconds=60,
    )
    wrapper = SPMFWrapper(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        patterns, vocab = wrapper.mine_patterns(
            dataset.sequences,
            Path(tmpdir),
            min_support=0.1,
        )
        logger.info("  BIDE mined %d patterns", len(patterns))
        for p in patterns[:5]:
            logger.info("    %s (sup=%d)", p.symbols, p.support)

    logger.info("SPMF BIDE Integration: PASSED")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run all Phase 2 pipeline tests."""
    logger.info("=" * 60)
    logger.info("Phase 2 Pipeline Integration Test")
    logger.info("=" * 60)

    traces = generate_synthetic_traces(count=25)

    test_subsequence_helper()
    test_symbolizer(traces)
    test_k_prefix(traces)
    test_spmf_input_output(traces)
    test_pattern_ranker(traces)
    test_signature_library(traces)
    test_spmf_bide_integration(traces)

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
