"""
Trace Schema Definitions for Web Agent Failure Detection

This module defines the data models for agent execution traces,
following the JSON format specified in the thesis proposal.

Trace Format:
- Metadata: trace_id, task, website, model, outcome, failure_type
- Steps: reasoning, action, observation, dom_hash

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import uuid
import json
import hashlib


class TaskOutcome(Enum):
    """Task completion outcome from WebArena programmatic validation."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"


class FailureType(Enum):
    """
    Four-category failure taxonomy for pattern analysis.
    
    Categories:
    - NAVIGATION: Agent cannot reach correct page/element
    - VALIDATION: Actions execute but don't achieve intended effect
    - RECOVERY: Ineffective error recovery attempts
    - CONTEXT: Misunderstanding of task or environment state
    """
    NAVIGATION = "navigation"
    VALIDATION = "validation"
    RECOVERY = "recovery"
    CONTEXT = "context"
    NATURAL = "natural"  # Uninjected, authentic failure
    UNKNOWN = "unknown"
    
    # Injection types (for tracking synthetic failures)
    INJECTED_SELECTOR = "injected_selector"
    INJECTED_DELETION = "injected_deletion"
    INJECTED_HALLUCINATION = "injected_hallucination"


class ActionType(Enum):
    """Web interaction action types."""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    SELECT = "select"
    HOVER = "hover"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    REFRESH = "refresh"
    STOP = "stop"
    UNKNOWN = "unknown"


class SelectorType(Enum):
    """CSS/XPath selector strategy types for symbolization."""
    ID = "id"
    CLASS = "class"
    XPATH = "xpath"
    TEXT = "text"
    CSS = "css"
    NAME = "name"
    TAG = "tag"
    BID = "bid"
    TAB = "tab"
    UNKNOWN = "unknown"


class ElementState(Enum):
    """DOM element state after action attempt."""
    VISIBLE = "visible"
    HIDDEN = "hidden"
    NOT_FOUND = "not_found"
    STALE = "stale"  # Previously valid reference now invalid
    INTERACTABLE = "interactable"
    NOT_INTERACTABLE = "not_interactable"
    UNKNOWN = "unknown"


@dataclass
class ActionRecord:
    """
    Records a single action taken by the agent.
    
    Attributes:
        type: The action type (click, type, navigate, etc.)
        selector: CSS selector or XPath used to target element
        selector_type: How the selector targets the element (id, class, xpath, text)
        value: Input value for type actions, URL for navigate, etc.
        raw_action: Original action string from agent output
        bid: BrowserGym element ID if available
    """
    type: ActionType
    selector: Optional[str] = None
    selector_type: Optional[SelectorType] = None
    value: Optional[str] = None
    raw_action: Optional[str] = None
    bid: Optional[str] = None  # BrowserGym element ID
    
    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "selector": self.selector,
            "selector_type": self.selector_type.value if self.selector_type else None,
            "value": self.value,
            "raw_action": self.raw_action,
            "bid": self.bid,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ActionRecord":
        return cls(
            type=ActionType(data.get("type", "unknown")),
            selector=data.get("selector"),
            selector_type=SelectorType(data["selector_type"]) if data.get("selector_type") else None,
            value=data.get("value"),
            raw_action=data.get("raw_action"),
            bid=data.get("bid"),
        )


@dataclass
class ObservationRecord:
    """
    Records the observation after an action.
    
    Attributes:
        element_found: Whether the target element was found
        element_state: State of the element (visible, hidden, not_found, stale)
        http_status: HTTP status code if a network request was made
        page_changed: Whether the page URL/content changed
        error_message: Any error message from the browser/environment
        visible_text: Relevant visible text snippet (truncated)
        screenshot_path: Path to screenshot if captured
    """
    element_found: bool = True
    element_state: ElementState = ElementState.UNKNOWN
    http_status: Optional[int] = None
    page_changed: bool = False
    error_message: Optional[str] = None
    visible_text: Optional[str] = None
    screenshot_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "element_found": self.element_found,
            "element_state": self.element_state.value,
            "http_status": self.http_status,
            "page_changed": self.page_changed,
            "error_message": self.error_message,
            "visible_text": self.visible_text[:500] if self.visible_text else None,
            "screenshot_path": self.screenshot_path,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ObservationRecord":
        return cls(
            element_found=data.get("element_found", True),
            element_state=ElementState(data.get("element_state", "unknown")),
            http_status=data.get("http_status"),
            page_changed=data.get("page_changed", False),
            error_message=data.get("error_message"),
            visible_text=data.get("visible_text"),
            screenshot_path=data.get("screenshot_path"),
        )


@dataclass
class ReasoningRecord:
    """
    Records the agent's reasoning/thought for a step.
    
    Attributes:
        raw_reasoning: Full reasoning text from agent
        intent: Extracted intent (what agent is trying to do)
        keywords: Extracted keywords for symbolization (retry, backtrack, verify, etc.)
        confidence: Agent's stated confidence if available
    """
    raw_reasoning: str = ""
    intent: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    confidence: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "raw_reasoning": self.raw_reasoning,
            "intent": self.intent,
            "keywords": self.keywords,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningRecord":
        return cls(
            raw_reasoning=data.get("raw_reasoning", ""),
            intent=data.get("intent"),
            keywords=data.get("keywords", []),
            confidence=data.get("confidence"),
        )


@dataclass
class TraceStep:
    """
    A single step in an agent execution trace.
    
    Corresponds to one iteration of the ReAct loop:
    Thought -> Action -> Observation
    """
    step_number: int
    reasoning: ReasoningRecord
    action: ActionRecord
    observation: ObservationRecord
    dom_hash: str = ""
    timestamp: Optional[str] = None
    url: Optional[str] = None
    
    # Token usage for computational savings analysis
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def to_dict(self) -> dict:
        return {
            "step_number": self.step_number,
            "reasoning": self.reasoning.to_dict(),
            "action": self.action.to_dict(),
            "observation": self.observation.to_dict(),
            "dom_hash": self.dom_hash,
            "timestamp": self.timestamp,
            "url": self.url,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TraceStep":
        return cls(
            step_number=data["step_number"],
            reasoning=ReasoningRecord.from_dict(data.get("reasoning", {})),
            action=ActionRecord.from_dict(data.get("action", {})),
            observation=ObservationRecord.from_dict(data.get("observation", {})),
            dom_hash=data.get("dom_hash", ""),
            timestamp=data.get("timestamp"),
            url=data.get("url"),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
        )


@dataclass
class TraceMetadata:
    """
    Metadata for an agent execution trace.
    """
    trace_id: str
    task_id: str
    task_description: str
    website: str
    model: str
    outcome: TaskOutcome
    failure_type: Optional[FailureType] = None
    
    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # Environment info
    benchmark: str = "webarena"  # webarena, browsergym, miniwob, workarena
    environment_version: Optional[str] = None
    
    # Annotation info
    annotator: Optional[str] = None
    annotation_notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "website": self.website,
            "model": self.model,
            "outcome": self.outcome.value,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "benchmark": self.benchmark,
            "environment_version": self.environment_version,
            "annotator": self.annotator,
            "annotation_notes": self.annotation_notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TraceMetadata":
        return cls(
            trace_id=data["trace_id"],
            task_id=data["task_id"],
            task_description=data["task_description"],
            website=data["website"],
            model=data["model"],
            outcome=TaskOutcome(data.get("outcome", "unknown")),
            failure_type=FailureType(data["failure_type"]) if data.get("failure_type") else None,
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            duration_seconds=data.get("duration_seconds"),
            benchmark=data.get("benchmark", "webarena"),
            environment_version=data.get("environment_version"),
            annotator=data.get("annotator"),
            annotation_notes=data.get("annotation_notes"),
        )


@dataclass
class AgentTrace:
    """
    Complete agent execution trace.
    
    This is the primary data structure for the thesis experiments,
    containing all information needed for:
    - Symbolization and pattern mining
    - Failure type annotation
    - Computational savings analysis
    - Agent-XES export
    """
    metadata: TraceMetadata
    steps: list[TraceStep] = field(default_factory=list)
    
    # Aggregate statistics
    total_steps: int = 0
    total_tokens: int = 0
    
    # Raw data references (for debugging)
    raw_trajectory_path: Optional[str] = None
    
    def __post_init__(self):
        self.total_steps = len(self.steps)
        self.total_tokens = sum(s.prompt_tokens + s.completion_tokens for s in self.steps)
    
    def add_step(self, step: TraceStep):
        """Add a step to the trace."""
        self.steps.append(step)
        self.total_steps = len(self.steps)
        self.total_tokens += step.prompt_tokens + step.completion_tokens
    
    def to_dict(self) -> dict:
        """Convert trace to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "raw_trajectory_path": self.raw_trajectory_path,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert trace to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentTrace":
        """Create trace from dictionary."""
        trace = cls(
            metadata=TraceMetadata.from_dict(data["metadata"]),
            steps=[TraceStep.from_dict(s) for s in data.get("steps", [])],
            raw_trajectory_path=data.get("raw_trajectory_path"),
        )
        trace.total_steps = data.get("total_steps", len(trace.steps))
        trace.total_tokens = data.get("total_tokens", 0)
        return trace
    
    @classmethod
    def from_json(cls, json_str: str) -> "AgentTrace":
        """Create trace from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_k_prefix(self, k: int) -> list[TraceStep]:
        """Get the first K steps of the trace."""
        return self.steps[:k]
    
    def compute_dom_hash(self, dom_content: str) -> str:
        """Compute a hash of DOM content for change detection."""
        return hashlib.md5(dom_content.encode()).hexdigest()[:8]


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()