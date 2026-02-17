"""
SPMF Wrapper - Java subprocess integration for sequential pattern mining

Purpose: Interface with the SPMF library (Java) to run the BIDE+ algorithm
for mining closed sequential patterns from symbolized trace prefixes.

SPMF Input Format:
    1 -1 2 -1 3 -1 -2       (sequence of single-item itemsets)
    - Positive integers = symbol IDs
    - -1 = itemset separator
    - -2 = sequence terminator

SPMF Output Format (BIDE+):
    1 -1 2 -1 3 -1 #SUP: 45  (pattern with support count)

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.preprocessing.symbolizer import SymbolVocabulary

logger = logging.getLogger(__name__)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class RawPattern:
    """
    A sequential pattern as mined by SPMF.

    Attributes:
        symbols: Ordered list of symbol strings in the pattern.
        support: Absolute support count (number of sequences containing pattern).
    """

    symbols: list[str]
    support: int

    def __len__(self) -> int:
        return len(self.symbols)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "symbols": self.symbols,
            "support": self.support,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RawPattern":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with symbols and support keys.

        Returns:
            Reconstructed RawPattern.
        """
        return cls(
            symbols=data["symbols"],
            support=data["support"],
        )


@dataclass
class SPMFConfig:
    """
    Configuration for SPMF execution.

    Attributes:
        spmf_jar_path: Path to the spmf.jar file.
        min_support: Minimum support threshold (relative, 0.0-1.0).
        max_pattern_length: Maximum pattern length to mine.
        timeout_seconds: Subprocess timeout in seconds.
    """

    spmf_jar_path: Path = field(default_factory=lambda: Path("/opt/spmf/spmf.jar"))
    min_support: float = 0.03
    max_pattern_length: int = 5
    timeout_seconds: int = 300

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "spmf_jar_path": str(self.spmf_jar_path),
            "min_support": self.min_support,
            "max_pattern_length": self.max_pattern_length,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SPMFConfig":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with SPMFConfig fields.

        Returns:
            Reconstructed SPMFConfig.
        """
        return cls(
            spmf_jar_path=Path(data.get("spmf_jar_path", "/opt/spmf/spmf.jar")),
            min_support=data.get("min_support", 0.03),
            max_pattern_length=data.get("max_pattern_length", 5),
            timeout_seconds=data.get("timeout_seconds", 300),
        )


# =============================================================================
# SPMFWrapper
# =============================================================================

# Regex for parsing SPMF BIDE+ output lines
_SUPPORT_PATTERN = re.compile(r"#SUP:\s*(\d+)")


class SPMFWrapper:
    """
    Wrapper for running SPMF sequential pattern mining algorithms.

    Handles conversion between string symbol sequences and the integer-based
    SPMF format, subprocess execution of BIDE+, and output parsing.
    """

    def __init__(self, config: SPMFConfig):
        """Initialize with SPMF configuration.

        Args:
            config: SPMFConfig specifying jar path, thresholds, etc.

        Raises:
            FileNotFoundError: If spmf.jar does not exist.
            RuntimeError: If Java is not available on PATH.
        """
        self.config = config
        self._validate_environment()

    def _validate_environment(self) -> None:
        """Check that Java and the SPMF jar are available."""
        if not self.config.spmf_jar_path.exists():
            raise FileNotFoundError(
                f"SPMF jar not found at {self.config.spmf_jar_path}"
            )

        java_path = shutil.which("java")
        if java_path is None:
            raise RuntimeError(
                "Java not found on PATH. SPMF requires Java 8+."
            )
        logger.info(
            "SPMF environment validated: jar=%s, java=%s",
            self.config.spmf_jar_path,
            java_path,
        )

    # -----------------------------------------------------------------
    # Input preparation
    # -----------------------------------------------------------------

    def prepare_input(
        self,
        sequences: list[list[str]],
        output_path: Path,
        vocabulary: Optional[SymbolVocabulary] = None,
    ) -> tuple[Path, SymbolVocabulary]:
        """Convert symbol sequences to SPMF integer format and write to file.

        Each symbol becomes a single-item itemset. Sequences are terminated
        with -2, and itemsets are separated by -1.

        Args:
            sequences: List of symbol sequences (list of list of str).
            output_path: Path to write the SPMF input file.
            vocabulary: Optional existing vocabulary to extend. If None,
                a new vocabulary is created.

        Returns:
            Tuple of (path to written file, SymbolVocabulary used).
        """
        if vocabulary is None:
            vocabulary = SymbolVocabulary()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for sequence in sequences:
                if not sequence:
                    continue
                parts = []
                for symbol in sequence:
                    symbol_id = vocabulary.get_or_create_id(symbol)
                    parts.append(f"{symbol_id} -1")
                parts.append("-2")
                f.write(" ".join(parts) + "\n")

        logger.info(
            "Prepared SPMF input: %d sequences, %d unique symbols -> %s",
            len(sequences),
            len(vocabulary),
            output_path,
        )
        return output_path, vocabulary

    # -----------------------------------------------------------------
    # BIDE+ execution
    # -----------------------------------------------------------------

    def run_bide(
        self,
        input_path: Path,
        output_path: Path,
        min_support: Optional[float] = None,
        max_pattern_length: Optional[int] = None,
    ) -> Path:
        """Execute BIDE+ algorithm via SPMF Java subprocess.

        Args:
            input_path: Path to SPMF-format input file.
            output_path: Path for SPMF output file.
            min_support: Minimum relative support (overrides config).
            max_pattern_length: Maximum pattern length (overrides config).

        Returns:
            Path to the output file.

        Raises:
            FileNotFoundError: If input file does not exist.
            subprocess.CalledProcessError: If SPMF execution fails.
            subprocess.TimeoutExpired: If execution exceeds timeout.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"SPMF input file not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        support = min_support if min_support is not None else self.config.min_support
        max_len = (max_pattern_length if max_pattern_length is not None
                   else self.config.max_pattern_length)

        # SPMF expects percentage support as a decimal string
        support_str = f"{support}"

        cmd = [
            "java", "-jar", str(self.config.spmf_jar_path),
            "run", "BIDE+",
            str(input_path),
            str(output_path),
            support_str,
        ]

        logger.info("Running BIDE+: %s", " ".join(cmd))
        logger.info(
            "Parameters: min_support=%s, max_pattern_length=%d, timeout=%ds",
            support_str,
            max_len,
            self.config.timeout_seconds,
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            check=True,
        )

        if result.stdout:
            logger.debug("SPMF stdout: %s", result.stdout.strip())
        if result.stderr:
            logger.warning("SPMF stderr: %s", result.stderr.strip())

        if not output_path.exists():
            raise RuntimeError(
                f"SPMF did not produce output file: {output_path}"
            )

        logger.info("BIDE+ completed, output: %s", output_path)
        return output_path

    # -----------------------------------------------------------------
    # Output parsing
    # -----------------------------------------------------------------

    def parse_output(
        self,
        output_path: Path,
        vocabulary: SymbolVocabulary,
        max_pattern_length: Optional[int] = None,
    ) -> list[RawPattern]:
        """Parse SPMF output file into RawPattern objects.

        Each line has format: ``1 -1 2 -1 3 -1 #SUP: 45``

        Args:
            output_path: Path to SPMF output file.
            vocabulary: SymbolVocabulary for ID-to-symbol translation.
            max_pattern_length: Discard patterns longer than this.

        Returns:
            List of RawPattern objects.
        """
        if not output_path.exists():
            raise FileNotFoundError(f"SPMF output file not found: {output_path}")

        max_len = (max_pattern_length if max_pattern_length is not None
                   else self.config.max_pattern_length)

        patterns: list[RawPattern] = []

        with open(output_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    pattern = self._parse_line(line, vocabulary)
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "Skipping malformed line %d: %s (%s)",
                        line_num,
                        line[:80],
                        exc,
                    )
                    continue

                if len(pattern) <= max_len:
                    patterns.append(pattern)

        logger.info(
            "Parsed %d patterns from %s (max_length=%d)",
            len(patterns),
            output_path,
            max_len,
        )
        return patterns

    @staticmethod
    def _parse_line(line: str, vocabulary: SymbolVocabulary) -> RawPattern:
        """Parse a single SPMF output line.

        Args:
            line: A line like ``1 -1 2 -1 3 -1 #SUP: 45``.
            vocabulary: SymbolVocabulary for reverse lookup.

        Returns:
            A RawPattern.

        Raises:
            ValueError: If the line cannot be parsed.
        """
        support_match = _SUPPORT_PATTERN.search(line)
        if not support_match:
            raise ValueError(f"No #SUP found in line: {line[:80]}")

        support = int(support_match.group(1))

        # Extract the pattern part (everything before #SUP)
        pattern_part = line[:support_match.start()].strip()

        # Parse symbol IDs: tokens that are positive integers (skip -1, -2)
        symbol_ids: list[int] = []
        for token in pattern_part.split():
            val = int(token)
            if val > 0:
                symbol_ids.append(val)

        symbols = [vocabulary.get_symbol(sid) for sid in symbol_ids]

        return RawPattern(symbols=symbols, support=support)

    # -----------------------------------------------------------------
    # Convenience: full pipeline
    # -----------------------------------------------------------------

    def mine_patterns(
        self,
        sequences: list[list[str]],
        work_dir: Path,
        min_support: Optional[float] = None,
        vocabulary: Optional[SymbolVocabulary] = None,
    ) -> tuple[list[RawPattern], SymbolVocabulary]:
        """Run the full SPMF pipeline: prepare, mine, parse.

        Convenience method that chains prepare_input, run_bide, and
        parse_output into a single call.

        Args:
            sequences: Symbol sequences to mine.
            work_dir: Working directory for intermediate files.
            min_support: Minimum support threshold (overrides config).
            vocabulary: Optional existing vocabulary to extend.

        Returns:
            Tuple of (list of RawPattern, SymbolVocabulary).
        """
        input_path = work_dir / "spmf_input.txt"
        output_path = work_dir / "spmf_output.txt"

        input_path, vocab = self.prepare_input(
            sequences, input_path, vocabulary
        )
        self.run_bide(input_path, output_path, min_support)
        patterns = self.parse_output(output_path, vocab)

        # Save vocabulary alongside output for reproducibility
        vocab.save(work_dir / "vocabulary.json")

        return patterns, vocab
