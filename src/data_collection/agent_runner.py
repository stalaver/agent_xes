"""
Agent Runner for Web Agent Trace Collection

This module handles:
- Agent execution on WebArena and BrowserGym benchmarks
- Integration with multiple LLM backends (HuggingFace, vLLM)
- Real-time trace capture during execution
- Task validation and outcome determination

Supported Benchmarks:
- WebArena (primary)
- BrowserGym (MiniWoB, WorkArena, WebArena-Verified)

Supported Models:
- Llama-3.2-3B-Instruct
- Qwen-2.5-7B-Instruct  
- Mistral-7B-Instruct-v0.3

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .trace_schema import (
    AgentTrace, TraceStep, TraceMetadata,
    ActionRecord, ObservationRecord, ReasoningRecord,
    TaskOutcome, FailureType, ActionType, SelectorType, ElementState,
    generate_trace_id, get_current_timestamp,
)
from .trace_logger import TraceLogger

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for agent execution."""
    model_name: str
    model_path: str
    max_steps: int = 50
    temperature: float = 0.0  # Deterministic for reproducibility
    max_tokens: int = 1024
    timeout_seconds: int = 300
    
    # WebArena specific
    observation_type: str = "accessibility_tree"  # or "html", "screenshot"
    action_space: str = "bid"  # BrowserGym element IDs
    
    # Device settings
    device: str = "cuda"
    load_in_4bit: bool = False
    
    @classmethod
    def llama_3b(cls) -> "AgentConfig":
        return cls(
            model_name="llama-3.2-3b",
            model_path="meta-llama/Llama-3.2-3B-Instruct",
        )
    
    @classmethod
    def qwen_7b(cls) -> "AgentConfig":
        return cls(
            model_name="qwen-2.5-7b",
            model_path="Qwen/Qwen2.5-7B-Instruct",
        )
    
    @classmethod
    def mistral_7b(cls) -> "AgentConfig":
        return cls(
            model_name="mistral-7b",
            model_path="mistralai/Mistral-7B-Instruct-v0.3",
        )
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "observation_type": self.observation_type,
            "action_space": self.action_space,
            "device": self.device,
            "load_in_4bit": self.load_in_4bit,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentConfig":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with AgentConfig fields.

        Returns:
            Reconstructed AgentConfig.
        """
        return cls(
            model_name=data["model_name"],
            model_path=data["model_path"],
            max_steps=data.get("max_steps", 50),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 1024),
            timeout_seconds=data.get("timeout_seconds", 300),
            observation_type=data.get("observation_type", "accessibility_tree"),
            action_space=data.get("action_space", "bid"),
            device=data.get("device", "cuda"),
            load_in_4bit=data.get("load_in_4bit", False),
        )


@dataclass 
class TaskConfig:
    """Configuration for a single task."""
    task_id: str
    task_description: str
    website: str
    benchmark: str = "webarena"
    start_url: Optional[str] = None
    reference_answer: Optional[str] = None
    
    # Evaluation
    eval_type: str = "programmatic"  # or "llm_judge", "exact_match"
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "website": self.website,
            "benchmark": self.benchmark,
            "start_url": self.start_url,
            "reference_answer": self.reference_answer,
            "eval_type": self.eval_type,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TaskConfig":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with TaskConfig fields.

        Returns:
            Reconstructed TaskConfig.
        """
        return cls(
            task_id=data["task_id"],
            task_description=data["task_description"],
            website=data["website"],
            benchmark=data.get("benchmark", "webarena"),
            start_url=data.get("start_url"),
            reference_answer=data.get("reference_answer"),
            eval_type=data.get("eval_type", "programmatic"),
        )


# =============================================================================
# Action Parser
# =============================================================================

class ActionParser:
    """
    Parse agent output into structured actions.
    
    Handles various action formats from different agent architectures:
    - ReAct: "Action: click[#submit-btn]"
    - BrowserGym: "click('bid123')"
    - Raw: {"action": "click", "element": "#submit-btn"}
    """
    
    # Regex patterns for action parsing
    PATTERNS = {
        # ReAct style: Action: click[selector]
        "react": re.compile(
            r"Action:\s*(\w+)\[([^\]]*)\]",
            re.IGNORECASE
        ),
        # Function call style: click('selector') or click("selector")
        "function": re.compile(
            r"(\w+)\s*\(\s*['\"]([^'\"]*)['\"](?:\s*,\s*['\"]([^'\"]*)['\"])?\s*\)",
            re.IGNORECASE
        ),
        # BrowserGym bid style: click(bid='123')
        "bid": re.compile(
            r"(\w+)\s*\(\s*bid\s*=\s*['\"]?(\w+)['\"]?\s*\)",
            re.IGNORECASE
        ),
    }
    
    ACTION_MAP = {
        "click": ActionType.CLICK,
        "type": ActionType.TYPE,
        "fill": ActionType.TYPE,
        "input": ActionType.TYPE,
        "scroll": ActionType.SCROLL,
        "scroll_down": ActionType.SCROLL,
        "scroll_up": ActionType.SCROLL,
        "goto": ActionType.NAVIGATE,
        "navigate": ActionType.NAVIGATE,
        "go_to_url": ActionType.NAVIGATE,
        "select": ActionType.SELECT,
        "select_option": ActionType.SELECT,
        "hover": ActionType.HOVER,
        "wait": ActionType.WAIT,
        "go_back": ActionType.GO_BACK,
        "go_forward": ActionType.GO_FORWARD,
        "refresh": ActionType.REFRESH,
        "stop": ActionType.STOP,
        "send_msg_to_user": ActionType.STOP,
    }
    
    @classmethod
    def parse(cls, action_str: str) -> ActionRecord:
        """
        Parse an action string into an ActionRecord.
        
        Args:
            action_str: Raw action string from agent output
            
        Returns:
            Parsed ActionRecord
        """
        action_str = action_str.strip()
        
        # Try each pattern
        for pattern_name, pattern in cls.PATTERNS.items():
            match = pattern.search(action_str)
            if match:
                return cls._create_record(match, pattern_name, action_str)
        
        # Try JSON parsing
        try:
            data = json.loads(action_str)
            if isinstance(data, dict) and "action" in data:
                return cls._from_json(data, action_str)
        except json.JSONDecodeError:
            pass
        
        # Fallback: unknown action
        logger.warning(f"Could not parse action: {action_str[:100]}")
        return ActionRecord(
            type=ActionType.UNKNOWN,
            raw_action=action_str,
        )
    
    @classmethod
    def _create_record(cls, match, pattern_name: str, raw: str) -> ActionRecord:
        """Create ActionRecord from regex match."""
        groups = match.groups()
        action_name = groups[0].lower()
        action_type = cls.ACTION_MAP.get(action_name, ActionType.UNKNOWN)
        
        selector = groups[1] if len(groups) > 1 else None
        value = groups[2] if len(groups) > 2 else None
        
        # Determine selector type
        selector_type = cls._infer_selector_type(selector, pattern_name)
        
        # For BrowserGym, selector is the bid
        bid = selector if pattern_name == "bid" else None
        
        return ActionRecord(
            type=action_type,
            selector=selector,
            selector_type=selector_type,
            value=value,
            raw_action=raw,
            bid=bid,
        )
    
    @classmethod
    def _from_json(cls, data: dict, raw: str) -> ActionRecord:
        """Create ActionRecord from JSON dict."""
        action_name = data.get("action", "").lower()
        action_type = cls.ACTION_MAP.get(action_name, ActionType.UNKNOWN)
        
        selector = data.get("element") or data.get("selector") or data.get("bid")
        value = data.get("value") or data.get("text") or data.get("url")
        
        return ActionRecord(
            type=action_type,
            selector=selector,
            selector_type=cls._infer_selector_type(selector, "json"),
            value=value,
            raw_action=raw,
            bid=data.get("bid"),
        )
    
    @classmethod
    def _infer_selector_type(cls, selector: Optional[str], pattern: str) -> Optional[SelectorType]:
        """Infer the selector type from the selector string."""
        if not selector:
            return None
        
        if pattern == "bid":
            return SelectorType.ID  # BrowserGym IDs
        
        if selector.startswith("#"):
            return SelectorType.ID
        elif selector.startswith("."):
            return SelectorType.CLASS
        elif selector.startswith("//") or selector.startswith("(//"):
            return SelectorType.XPATH
        elif selector.startswith("["):
            return SelectorType.CSS
        elif "=" in selector:  # name=value or other attribute
            return SelectorType.CSS
        else:
            return SelectorType.TEXT


# =============================================================================
# Reasoning Parser  
# =============================================================================

class ReasoningParser:
    """
    Extract structured information from agent reasoning text.
    
    Identifies:
    - Intent keywords
    - Behavioral patterns (retry, backtrack, verify, explore)
    - Confidence indicators
    """
    
    # Keywords that indicate behavioral patterns
    BEHAVIOR_KEYWORDS = {
        "retry": ["retry", "try again", "attempt again", "re-try"],
        "backtrack": ["back", "go back", "return", "previous", "backtrack"],
        "verify": ["verify", "check", "confirm", "ensure", "validate"],
        "explore": ["explore", "look for", "search", "find", "navigate"],
        "stuck": ["stuck", "cannot", "unable", "failed", "error"],
        "confused": ["confused", "unclear", "not sure", "uncertain"],
    }
    
    @classmethod
    def parse(cls, reasoning_text: str) -> ReasoningRecord:
        """
        Parse reasoning text into a ReasoningRecord.
        
        Args:
            reasoning_text: Raw reasoning/thought from agent
            
        Returns:
            Parsed ReasoningRecord
        """
        reasoning_text = reasoning_text.strip()
        lower_text = reasoning_text.lower()
        
        # Extract keywords
        keywords = []
        for category, terms in cls.BEHAVIOR_KEYWORDS.items():
            if any(term in lower_text for term in terms):
                keywords.append(category)
        
        # Try to extract intent (first sentence or phrase before "I will/should")
        intent = cls._extract_intent(reasoning_text)
        
        # Look for confidence indicators
        confidence = cls._extract_confidence(reasoning_text)
        
        return ReasoningRecord(
            raw_reasoning=reasoning_text,
            intent=intent,
            keywords=keywords,
            confidence=confidence,
        )
    
    @classmethod
    def _extract_intent(cls, text: str) -> Optional[str]:
        """Extract the agent's intent from reasoning."""
        # Look for "I need to..." or "I will..." patterns
        patterns = [
            r"I need to ([^.]+)",
            r"I should ([^.]+)",
            r"I will ([^.]+)",
            r"Let me ([^.]+)",
            r"Going to ([^.]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:100]  # Truncate
        
        # Fallback: first sentence
        first_sentence = text.split(".")[0].strip()
        if len(first_sentence) < 200:
            return first_sentence
        
        return None
    
    @classmethod
    def _extract_confidence(cls, text: str) -> Optional[float]:
        """Extract confidence score if mentioned."""
        # Look for "confidence: X%" or similar
        patterns = [
            r"confidence[:\s]+(\d+(?:\.\d+)?)\s*%",
            r"(\d+(?:\.\d+)?)\s*%\s*confident",
            r"confidence[:\s]+(\d+(?:\.\d+)?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return value / 100 if value > 1 else value
        
        return None


# =============================================================================
# Base Agent Runner
# =============================================================================

class BaseAgentRunner(ABC):
    """Abstract base for agent execution.

    Subclasses must implement ``run_task`` which owns the full lifecycle
    of a single task execution (env creation, agent loop, trace finalization).
    Concrete helpers for step construction and DOM hashing are provided here.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        trace_logger: TraceLogger,
    ):
        self.config = agent_config
        self.trace_logger = trace_logger
        self.action_parser = ActionParser()
        self.reasoning_parser = ReasoningParser()

        self._model = None
        self._tokenizer = None

    @abstractmethod
    def run_task(self, task: TaskConfig) -> AgentTrace:
        """Execute a single task and return the complete trace.

        Args:
            task: Task configuration.

        Returns:
            Complete AgentTrace.
        """

    def _compute_dom_hash(self, dom_content: str) -> str:
        """Compute hash of DOM content for change tracking."""
        return hashlib.md5(dom_content.encode()).hexdigest()[:8]

    def _create_step(
        self,
        step_num: int,
        reasoning: str,
        action_str: str,
        observation: dict,
        url: str,
        dom_content: str,
        token_usage: dict,
    ) -> TraceStep:
        """Create a TraceStep from execution data."""
        return TraceStep(
            step_number=step_num,
            reasoning=self.reasoning_parser.parse(reasoning),
            action=self.action_parser.parse(action_str),
            observation=ObservationRecord(
                element_found=observation.get("element_found", True),
                element_state=ElementState(observation.get("element_state", "unknown")),
                http_status=observation.get("http_status"),
                page_changed=observation.get("page_changed", False),
                error_message=observation.get("error_message"),
                visible_text=observation.get("visible_text"),
            ),
            dom_hash=self._compute_dom_hash(dom_content),
            timestamp=get_current_timestamp(),
            url=url,
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("completion_tokens", 0),
        )


# =============================================================================
# BrowserGym Agent Runner
# =============================================================================

class BrowserGymAgentRunner(BaseAgentRunner):
    """
    Agent runner for BrowserGym-based benchmarks.
    
    Supports:
    - WebArena
    - WebArena-Verified
    - MiniWoB
    - WorkArena
    """
    
    BENCHMARK_MODULES = {
        "miniwob": "browsergym.miniwob",
        "webarena": "browsergym.webarena",
        "workarena": "browsergym.workarena",
    }

    def __init__(
        self,
        agent_config: AgentConfig,
        trace_logger: TraceLogger,
        benchmark: str = "webarena",
        headless: bool = True,
    ):
        super().__init__(agent_config, trace_logger)
        self.benchmark = benchmark
        self.headless = headless
        self._gymnasium = None

        try:
            import gymnasium
            self._gymnasium = gymnasium

            module_name = self.BENCHMARK_MODULES.get(benchmark)
            if module_name:
                import importlib
                importlib.import_module(module_name)
                logger.info(f"Registered BrowserGym environments for {benchmark}")
            else:
                logger.warning(f"Unknown benchmark '{benchmark}', skipping env registration")
        except ImportError as e:
            logger.warning(f"BrowserGym not installed: {e}")
    
    def load_model(self):
        """Load the LLM model for agent execution."""
        if self._model is not None:
            return
        
        logger.info(f"Loading model: {self.config.model_path}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
        
        if self.config.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **load_kwargs
        )
        
        logger.info(f"Model loaded: {self.config.model_name}")
    
    def _flatten_axtree(self, obs: dict) -> str:
        """Flatten the BrowserGym accessibility tree observation to text.

        Args:
            obs: Observation dict from gymnasium env.reset() / env.step().

        Returns:
            Text representation of the accessibility tree.
        """
        axtree_obj = obs.get("axtree_object")
        if axtree_obj is not None:
            try:
                from browsergym.utils.obs import flatten_axtree_to_str
                return flatten_axtree_to_str(axtree_obj)
            except ImportError:
                pass

            # Fallback: walk the tree manually
            return self._walk_axtree_node(axtree_obj, depth=0)

        # Last resort: return any text-like field available
        for key in ("axtree_txt", "dom_txt", "pruned_html"):
            if obs.get(key):
                return obs[key]
        return ""

    @staticmethod
    def _walk_axtree_node(node: dict, depth: int = 0) -> str:
        """Recursively format an axtree_object dict into indented text."""
        if not isinstance(node, dict):
            return str(node)

        role = node.get("role", "")
        name = node.get("name", "")
        bid = node.get("browsergym_id", "") or node.get("bid", "")
        value = node.get("value", "")

        parts: list[str] = []
        if role:
            parts.append(role)
        if name:
            parts.append(f"'{name}'")
        if bid:
            parts.append(f"[bid={bid}]")
        if value:
            parts.append(f"value='{value}'")

        indent = "  " * depth
        line = f"{indent}{' '.join(parts)}" if parts else ""

        lines = [line] if line else []
        for child in node.get("children", []):
            lines.append(BrowserGymAgentRunner._walk_axtree_node(child, depth + 1))

        return "\n".join(lines)

    def run_task(self, task: TaskConfig) -> AgentTrace:
        """Execute a task using BrowserGym's gymnasium API.

        Creates a gymnasium environment for the task, runs a ReAct agent loop
        (observe -> reason -> act) up to max_steps, and returns the full trace.

        Args:
            task: Task configuration with task_id, description, etc.

        Returns:
            Complete AgentTrace with all steps and outcome metadata.
        """
        if self._gymnasium is None:
            raise RuntimeError("BrowserGym not available — gymnasium failed to import")

        self.load_model()

        trace_id = self.trace_logger.start_trace(
            task_id=task.task_id,
            task_description=task.task_description,
            website=task.website,
            model=self.config.model_name,
        )

        env_id = f"browsergym/{self._get_task_entrypoint(task)}"
        env = self._gymnasium.make(env_id, headless=self.headless)

        outcome = TaskOutcome.ERROR
        failure_type = None
        reward = 0.0

        try:
            obs, info = env.reset()
            goal = obs.get("goal", task.task_description)

            for step_num in range(1, self.config.max_steps + 1):
                axtree_text = self._flatten_axtree(obs)
                obs_data = {
                    "url": obs.get("url", ""),
                    "axtree_text": axtree_text,
                    "focused_element": obs.get("focused_element_bid", ""),
                }

                reasoning, action_str, token_usage = self._generate_action(
                    goal, obs_data, step_num,
                )

                prev_url = obs.get("url", "")
                obs, reward, terminated, truncated, info = env.step(action_str)
                new_url = obs.get("url", "")

                action_error = info.get("action_error", False) if isinstance(info, dict) else False
                error_msg = str(info.get("error_message", "")) if isinstance(info, dict) and info.get("error_message") else None

                obs_record = {
                    "element_found": not action_error,
                    "element_state": "not_found" if action_error else "visible",
                    "page_changed": new_url != prev_url,
                    "error_message": error_msg,
                }

                step = self._create_step(
                    step_num=step_num,
                    reasoning=reasoning,
                    action_str=action_str,
                    observation=obs_record,
                    url=prev_url,
                    dom_content=axtree_text,
                    token_usage=token_usage,
                )
                self.trace_logger.log_step(step)

                if terminated or truncated:
                    break

            if reward > 0:
                outcome = TaskOutcome.SUCCESS
            elif terminated or truncated:
                outcome = TaskOutcome.FAILURE
            else:
                outcome = TaskOutcome.TIMEOUT

            if outcome == TaskOutcome.FAILURE:
                failure_type = FailureType.NATURAL

        except Exception as e:
            logger.error(f"Task execution error for {task.task_id}: {e}", exc_info=True)
            outcome = TaskOutcome.ERROR
            failure_type = None
        finally:
            env.close()

        trace = self.trace_logger.finalize_trace(
            outcome=outcome,
            failure_type=failure_type,
        )
        return trace
    
    def _get_task_entrypoint(self, task: TaskConfig) -> str:
        """Map TaskConfig.task_id to a BrowserGym gymnasium env name.

        If the task_id is already prefixed with the benchmark name
        (e.g. ``miniwob.click-test``), return it as-is.  Otherwise
        prepend the benchmark.

        Args:
            task: Task configuration.

        Returns:
            String like ``miniwob.click-test`` suitable for
            ``gymnasium.make(f"browsergym/{entrypoint}")``.
        """
        if task.task_id.startswith(f"{self.benchmark}."):
            return task.task_id
        return f"{self.benchmark}.{task.task_id}"
    
    def _generate_action(
        self,
        goal: str,
        observation: dict,
        step_num: int,
    ) -> tuple[str, str, dict]:
        """Generate agent action using the LLM.

        Args:
            goal: Natural-language task description.
            observation: Dict with ``url``, ``axtree_text``, etc.
            step_num: Current step number (1-based).

        Returns:
            Tuple of (reasoning, action_string, token_usage_dict).
        """
        prompt = self._construct_prompt(goal, observation, step_num)

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        reasoning, action = self._parse_response(response)

        token_usage = {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
        }

        return reasoning, action, token_usage

    def _construct_prompt(self, goal: str, observation: dict, step_num: int) -> str:
        """Build a ReAct-style prompt using BrowserGym function-call actions.

        Args:
            goal: The task the agent must accomplish.
            observation: Dict with ``url``, ``axtree_text``, etc.
            step_num: Current step number.

        Returns:
            Formatted prompt string.
        """
        axtree = observation.get("axtree_text", "")[:4000]
        url = observation.get("url", "unknown")

        return (
            "You are a web navigation agent. You interact with web pages "
            "using these actions:\n"
            '- click(element_id): Click on an element\n'
            '- fill(element_id, "text"): Type text into an input field\n'
            "- scroll(x, y): Scroll the page\n"
            '- goto("url"): Navigate to a URL\n'
            '- send_msg_to_user("message"): Send a message (use when done)\n'
            "\n"
            f"Current task: {goal}\n"
            "\n"
            f"Current URL: {url}\n"
            "\n"
            "Current page content (accessibility tree):\n"
            f"{axtree}\n"
            "\n"
            "Based on the current page, provide your reasoning and next action.\n"
            "\n"
            "Thought: <your step-by-step reasoning>\n"
            "Action: <one action to execute>\n"
            "\n"
            f"Step {step_num}:\n"
        )

    # Regex tiers for extracting a BrowserGym action from LLM output
    _RE_ACTION_LABEL = re.compile(
        r"Action:\s*(.+?)(?:\n|$)", re.IGNORECASE
    )
    _RE_FUNC_CALL = re.compile(
        r"(click|fill|scroll|goto|go_back|send_msg_to_user|select_option|hover|noop)"
        r"\s*\(.*?\)",
        re.IGNORECASE | re.DOTALL,
    )
    _RE_THOUGHT = re.compile(
        r"(?:Thought|Reasoning):\s*(.+?)(?=Action:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _FALLBACK_ACTION = 'send_msg_to_user("could not parse action")'

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response into (reasoning, action_string).

        Uses a multi-tier approach so that malformed output from small
        models still produces a usable (or gracefully failing) action
        rather than crashing.
        """
        reasoning = ""
        action = ""

        thought_match = self._RE_THOUGHT.search(response)
        if thought_match:
            reasoning = thought_match.group(1).strip()

        # Tier 1: look for "Action: <something>"
        action_match = self._RE_ACTION_LABEL.search(response)
        if action_match:
            candidate = action_match.group(1).strip()
            # Verify it looks like a function call
            if self._RE_FUNC_CALL.search(candidate):
                func = self._RE_FUNC_CALL.search(candidate)
                action = func.group(0)
            else:
                action = candidate

        # Tier 2: scan the whole response for any function-call pattern
        if not action:
            func_match = self._RE_FUNC_CALL.search(response)
            if func_match:
                action = func_match.group(0)

        # Tier 3: nothing parseable — emit a safe noop
        if not action:
            logger.warning(
                "Could not parse action from LLM output: %s",
                response[:200],
            )
            action = self._FALLBACK_ACTION

        return reasoning, action


# =============================================================================
# Batch Runner
# =============================================================================

class BatchRunner:
    """
    Run multiple tasks in batch and collect traces.
    
    Features:
    - Progress tracking
    - Error handling and recovery
    - Statistics collection
    """
    
    def __init__(
        self,
        agent_runner: BaseAgentRunner,
        trace_logger: TraceLogger,
    ):
        self.runner = agent_runner
        self.logger = trace_logger
        self.stats = {
            "total": 0,
            "success": 0,
            "failure": 0,
            "error": 0,
        }
    
    def run_tasks(
        self,
        tasks: list[TaskConfig],
        continue_on_error: bool = True,
    ) -> list[AgentTrace]:
        """
        Run a batch of tasks.
        
        Args:
            tasks: List of task configurations
            continue_on_error: Whether to continue after errors
            
        Returns:
            List of completed traces
        """
        traces = []
        
        for i, task in enumerate(tasks):
            logger.info(f"Running task {i+1}/{len(tasks)}: {task.task_id}")
            
            try:
                trace = self.runner.run_task(task)
                traces.append(trace)
                
                # Update stats
                self.stats["total"] += 1
                self.stats[trace.metadata.outcome.value] += 1
                
            except Exception as e:
                logger.error(f"Error running task {task.task_id}: {e}")
                self.stats["error"] += 1
                
                if not continue_on_error:
                    raise
        
        return traces
    
    def print_stats(self):
        """Print batch execution statistics."""
        print("\n" + "="*50)
        print("Batch Execution Statistics")
        print("="*50)
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        
        if self.stats["total"] > 0:
            success_rate = self.stats["success"] / self.stats["total"] * 100
            print(f"  success_rate: {success_rate:.1f}%")
        print("="*50)
