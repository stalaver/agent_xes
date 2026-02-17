"""
Trace Symbolizer - Multi-level abstraction for agent execution traces

Purpose: Convert TraceStep objects into symbolic representations at three
abstraction levels (fine, medium, coarse) for sequential pattern mining.

Abstraction Levels:
- Level 0 (Fine):   {ACTION}_{SELECTOR}_{ELEMENT_STATE}_{HTTP}
- Level 1 (Medium): {ACTION}_{SELECTOR}_{SUCCESS/FAIL}
- Level 2 (Coarse): {CATEGORY}_{SUCCESS/FAIL}

Reasoning keywords are appended as suffixes when detected:
- R_RETRY, R_BACK, R_VERIFY, R_EXPLORE, R_STUCK, R_CONFUSED

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.data_collection.trace_schema import (
    AgentTrace,
    TraceStep,
    ActionRecord,
    ObservationRecord,
    ReasoningRecord,
    ActionType,
    SelectorType,
    ElementState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

ACTION_CATEGORY_MAP: dict[ActionType, str] = {
    ActionType.CLICK: "INTERACTION",
    ActionType.HOVER: "INTERACTION",
    ActionType.SELECT: "INTERACTION",
    ActionType.TYPE: "INPUT",
    ActionType.NAVIGATE: "NAVIGATION",
    ActionType.GO_BACK: "NAVIGATION",
    ActionType.GO_FORWARD: "NAVIGATION",
    ActionType.SCROLL: "NAVIGATION",
    ActionType.REFRESH: "NAVIGATION",
    ActionType.WAIT: "CONTROL",
    ActionType.STOP: "CONTROL",
    ActionType.SCREENSHOT: "CONTROL",
    ActionType.UNKNOWN: "UNKNOWN",
}

REASONING_SYMBOL_MAP: dict[str, str] = {
    "retry": "R_RETRY",
    "backtrack": "R_BACK",
    "verify": "R_VERIFY",
    "explore": "R_EXPLORE",
    "stuck": "R_STUCK",
    "confused": "R_CONFUSED",
}

SUCCESS_STATES: set[ElementState] = {
    ElementState.VISIBLE,
    ElementState.INTERACTABLE,
}

REASONING_SEPARATOR = "__"


# =============================================================================
# SymbolVocabulary
# =============================================================================

@dataclass
class SymbolVocabulary:
    """
    Bidirectional mapping between symbol strings and integer IDs.

    SPMF requires positive integer IDs. This vocabulary tracks the mapping
    built during input preparation and allows reverse lookup when parsing
    SPMF output.

    Attributes:
        symbol_to_id: Maps symbol string to integer ID (1-based).
        id_to_symbol: Maps integer ID back to symbol string.
    """

    symbol_to_id: dict[str, int] = field(default_factory=dict)
    id_to_symbol: dict[int, str] = field(default_factory=dict)

    def get_or_create_id(self, symbol: str) -> int:
        """Get existing ID for symbol, or assign the next available ID.

        Args:
            symbol: The symbol string.

        Returns:
            The integer ID (1-based) for the symbol.
        """
        if symbol not in self.symbol_to_id:
            new_id = len(self.symbol_to_id) + 1
            self.symbol_to_id[symbol] = new_id
            self.id_to_symbol[new_id] = symbol
        return self.symbol_to_id[symbol]

    def get_symbol(self, symbol_id: int) -> str:
        """Look up the symbol string for an integer ID.

        Args:
            symbol_id: The integer ID.

        Returns:
            The symbol string.

        Raises:
            KeyError: If the ID is not in the vocabulary.
        """
        return self.id_to_symbol[symbol_id]

    def __len__(self) -> int:
        return len(self.symbol_to_id)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "symbol_to_id": self.symbol_to_id,
            "id_to_symbol": {str(k): v for k, v in self.id_to_symbol.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SymbolVocabulary":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with symbol_to_id and id_to_symbol keys.

        Returns:
            Reconstructed SymbolVocabulary.
        """
        vocab = cls()
        vocab.symbol_to_id = data.get("symbol_to_id", {})
        vocab.id_to_symbol = {
            int(k): v for k, v in data.get("id_to_symbol", {}).items()
        }
        return vocab

    def save(self, path: Path) -> None:
        """Save vocabulary to JSON file.

        Args:
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved vocabulary (%d symbols) to %s", len(self), path)

    @classmethod
    def load(cls, path: Path) -> "SymbolVocabulary":
        """Load vocabulary from JSON file.

        Args:
            path: Input file path.

        Returns:
            Loaded SymbolVocabulary.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls.from_dict(data)
        logger.info("Loaded vocabulary (%d symbols) from %s", len(vocab), path)
        return vocab


# =============================================================================
# TraceSymbolizer
# =============================================================================

class TraceSymbolizer:
    """
    Convert agent execution trace steps into symbolic representations.

    Supports three abstraction levels:
    - Level 0 (Fine):   Full detail including element state and HTTP status
    - Level 1 (Medium): Action type, selector type, and binary outcome
    - Level 2 (Coarse): Action category and binary outcome
    """

    def __init__(self, abstraction_level: int = 1):
        """Initialize symbolizer with the desired abstraction level.

        Args:
            abstraction_level: 0 (fine), 1 (medium), or 2 (coarse).

        Raises:
            ValueError: If abstraction_level is not 0, 1, or 2.
        """
        if abstraction_level not in (0, 1, 2):
            raise ValueError(
                f"abstraction_level must be 0, 1, or 2, got {abstraction_level}"
            )
        self.abstraction_level = abstraction_level

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def symbolize_step(self, step: TraceStep) -> str:
        """Convert a single trace step to a symbol string.

        Args:
            step: A TraceStep from an agent execution trace.

        Returns:
            Symbol string at the configured abstraction level.
        """
        action_symbol = self._symbolize_action(step.action, step.observation)
        reasoning_suffix = self._symbolize_reasoning(step.reasoning)

        if reasoning_suffix:
            return f"{action_symbol}{REASONING_SEPARATOR}{reasoning_suffix}"
        return action_symbol

    def symbolize_trace(self, trace: AgentTrace) -> list[str]:
        """Convert an entire trace to a symbol sequence.

        Args:
            trace: Complete agent execution trace.

        Returns:
            List of symbol strings, one per step.
        """
        return [self.symbolize_step(step) for step in trace.steps]

    def symbolize_prefix(self, trace: AgentTrace, k: int) -> list[str]:
        """Convert the first K steps of a trace to a symbol sequence.

        Args:
            trace: Complete agent execution trace.
            k: Number of prefix steps to symbolize.

        Returns:
            List of symbol strings for the first K steps.
        """
        prefix_steps = trace.get_k_prefix(k)
        return [self.symbolize_step(step) for step in prefix_steps]

    # -----------------------------------------------------------------
    # Action symbolization by level
    # -----------------------------------------------------------------

    def _symbolize_action(
        self, action: ActionRecord, observation: ObservationRecord
    ) -> str:
        """Produce the action portion of the symbol based on abstraction level.

        Args:
            action: The action taken.
            observation: The observation after the action.

        Returns:
            Action symbol string (without reasoning suffix).
        """
        if self.abstraction_level == 0:
            return self._symbolize_level0(action, observation)
        elif self.abstraction_level == 1:
            return self._symbolize_level1(action, observation)
        else:
            return self._symbolize_level2(action, observation)

    def _symbolize_level0(
        self, action: ActionRecord, observation: ObservationRecord
    ) -> str:
        """Level 0 (Fine): {ACTION}_{SELECTOR}_{ELEMENT_STATE}_{HTTP}.

        Example: CLICK_ID_VISIBLE_OK, TYPE_CLASS_NOT_FOUND_ERR
        """
        action_str = action.type.name
        selector_str = (action.selector_type.name if action.selector_type
                        else "NONE")
        state_str = observation.element_state.name
        http_str = self._http_bucket(observation.http_status)
        return f"{action_str}_{selector_str}_{state_str}_{http_str}"

    def _symbolize_level1(
        self, action: ActionRecord, observation: ObservationRecord
    ) -> str:
        """Level 1 (Medium): {ACTION}_{SELECTOR}_{SUCCESS/FAIL}.

        Example: CLICK_ID_SUCCESS, TYPE_CLASS_FAIL
        """
        action_str = action.type.name
        selector_str = (action.selector_type.name if action.selector_type
                        else "NONE")
        outcome_str = self._binary_outcome(observation)
        return f"{action_str}_{selector_str}_{outcome_str}"

    def _symbolize_level2(
        self, action: ActionRecord, observation: ObservationRecord
    ) -> str:
        """Level 2 (Coarse): {CATEGORY}_{SUCCESS/FAIL}.

        Example: INTERACTION_SUCCESS, INPUT_FAIL
        """
        category = ACTION_CATEGORY_MAP.get(action.type, "UNKNOWN")
        outcome_str = self._binary_outcome(observation)
        return f"{category}_{outcome_str}"

    # -----------------------------------------------------------------
    # Reasoning symbolization
    # -----------------------------------------------------------------

    @staticmethod
    def _symbolize_reasoning(reasoning: ReasoningRecord) -> str:
        """Extract reasoning symbol suffix from keywords.

        If multiple reasoning keywords are present, they are joined with
        underscores (e.g. R_RETRY_R_CONFUSED).

        Args:
            reasoning: The reasoning record for the step.

        Returns:
            Reasoning suffix string, or empty string if none detected.
        """
        symbols = []
        for keyword in reasoning.keywords:
            mapped = REASONING_SYMBOL_MAP.get(keyword)
            if mapped:
                symbols.append(mapped)
        return "_".join(symbols)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _binary_outcome(observation: ObservationRecord) -> str:
        """Determine SUCCESS or FAIL from observation.

        SUCCESS when:
        - element_found is True
        - element_state is VISIBLE or INTERACTABLE
        - no error_message present

        Args:
            observation: The observation after the action.

        Returns:
            "SUCCESS" or "FAIL".
        """
        if (
            observation.element_found
            and observation.element_state in SUCCESS_STATES
            and not observation.error_message
        ):
            return "SUCCESS"
        return "FAIL"

    @staticmethod
    def _http_bucket(http_status: Optional[int]) -> str:
        """Bucket HTTP status code for Level 0 symbols.

        Args:
            http_status: HTTP status code, or None if no request.

        Returns:
            "OK" for 2xx/3xx or None, "ERR" for 4xx/5xx.
        """
        if http_status is None:
            return "OK"
        if 200 <= http_status < 400:
            return "OK"
        return "ERR"
