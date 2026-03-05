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

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Generator
import hashlib

from .trace_schema import (
    AgentTrace, TraceStep, TraceMetadata,
    ActionRecord, ObservationRecord, ReasoningRecord,
    TaskOutcome, FailureType, ActionType, SelectorType, ElementState,
    generate_trace_id, get_current_timestamp,
)
from .trace_logger import TraceLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    """
    Abstract base class for agent execution.
    
    Subclasses implement benchmark-specific execution logic.
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
        
        # Model (lazy loaded)
        self._model = None
        self._tokenizer = None
    
    @abstractmethod
    def run_task(self, task: TaskConfig) -> AgentTrace:
        """
        Execute a single task and return the trace.
        
        Args:
            task: Task configuration
            
        Returns:
            Complete AgentTrace
        """
        pass
    
    @abstractmethod
    def get_observation(self, env) -> dict:
        """Get observation from environment."""
        pass
    
    @abstractmethod
    def execute_action(self, env, action: ActionRecord) -> dict:
        """Execute action in environment and return result."""
        pass
    
    @abstractmethod
    def check_task_completion(self, env, task: TaskConfig) -> TaskOutcome:
        """Check if task is complete and determine outcome."""
        pass
    
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
        
        # Import BrowserGym components
        try:
            from browsergym.core.env import BrowserEnv
            self.BrowserEnv = BrowserEnv
        except ImportError:
            logger.warning("BrowserGym not installed. Install with: pip install browsergym")
            self.BrowserEnv = None
    
    def load_model(self):
        """Load the LLM model for agent execution."""
        if self._model is not None:
            return
        
        logger.info(f"Loading model: {self.config.model_path}")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
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
    
    def run_task(self, task: TaskConfig) -> AgentTrace:
        """Execute a task using BrowserGym."""
        if self.BrowserEnv is None:
            raise RuntimeError("BrowserGym not available")
        
        # Ensure model is loaded
        self.load_model()
        
        # Start trace
        trace_id = self.trace_logger.start_trace(
            task_id=task.task_id,
            task_description=task.task_description,
            website=task.website,
            model=self.config.model_name,
        )
        
        # Create environment
        env = self.BrowserEnv(
            task_entrypoint=self._get_task_entrypoint(task),
            headless=self.headless,
        )
        
        try:
            obs, info = env.reset()
            
            for step_num in range(1, self.config.max_steps + 1):
                # Get observation
                obs_data = self.get_observation(env)
                
                # Generate agent response
                reasoning, action_str, token_usage = self._generate_action(
                    task.task_description,
                    obs_data,
                    step_num,
                )
                
                # Parse and execute action
                action = self.action_parser.parse(action_str)
                result = self.execute_action(env, action)
                
                # Create and log step
                step = self._create_step(
                    step_num=step_num,
                    reasoning=reasoning,
                    action_str=action_str,
                    observation=result,
                    url=obs_data.get("url", ""),
                    dom_content=obs_data.get("dom", ""),
                    token_usage=token_usage,
                )
                self.trace_logger.log_step(step)
                
                # Check for stop action or completion
                if action.type == ActionType.STOP:
                    break
                
                # Check task completion
                outcome = self.check_task_completion(env, task)
                if outcome != TaskOutcome.UNKNOWN:
                    break
                
                # Get new observation
                obs, reward, done, truncated, info = env.step(action_str)
                
                if done or truncated:
                    break
            
            # Determine final outcome
            outcome = self.check_task_completion(env, task)
            failure_type = self._classify_failure(outcome) if outcome == TaskOutcome.FAILURE else None
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            outcome = TaskOutcome.ERROR
            failure_type = None
        finally:
            env.close()
        
        # Finalize trace
        trace = self.trace_logger.finalize_trace(
            outcome=outcome,
            failure_type=failure_type,
        )
        
        return trace
    
    def get_observation(self, env) -> dict:
        """Extract observation from BrowserGym environment."""
        obs = env.observation_handler.get_observation()
        
        return {
            "url": obs.get("url", ""),
            "dom": obs.get("dom_txt", "") or obs.get("axtree_txt", ""),
            "screenshot": obs.get("screenshot"),
            "focused_element": obs.get("focused_element_bid"),
        }
    
    def execute_action(self, env, action: ActionRecord) -> dict:
        """Execute action in BrowserGym environment."""
        try:
            # BrowserGym uses string actions
            action_str = action.raw_action or self._format_action(action)
            obs, reward, done, truncated, info = env.step(action_str)
            
            return {
                "element_found": not info.get("action_error", False),
                "element_state": "visible" if not info.get("action_error") else "not_found",
                "page_changed": info.get("page_changed", False),
                "error_message": info.get("error_message"),
            }
        except Exception as e:
            return {
                "element_found": False,
                "element_state": "not_found",
                "error_message": str(e),
            }
    
    def check_task_completion(self, env, task: TaskConfig) -> TaskOutcome:
        """Check task completion using environment's evaluator."""
        try:
            # BrowserGym provides reward signal
            if hasattr(env, 'get_reward'):
                reward = env.get_reward()
                if reward > 0:
                    return TaskOutcome.SUCCESS
                elif reward < 0:
                    return TaskOutcome.FAILURE
            
            return TaskOutcome.UNKNOWN
        except Exception:
            return TaskOutcome.UNKNOWN
    
    def _get_task_entrypoint(self, task: TaskConfig) -> str:
        """Get BrowserGym task entrypoint from task config."""
        # This maps task IDs to BrowserGym task entrypoints
        # Format varies by benchmark
        if self.benchmark == "webarena":
            return f"webarena.{task.task_id}"
        elif self.benchmark == "miniwob":
            return f"miniwob.{task.task_id}"
        elif self.benchmark == "workarena":
            return f"workarena.{task.task_id}"
        else:
            return task.task_id
    
    def _generate_action(
        self,
        task: str,
        observation: dict,
        step_num: int,
    ) -> tuple[str, str, dict]:
        """
        Generate agent action using the LLM.
        
        Returns:
            Tuple of (reasoning, action, token_usage)
        """
        # Construct prompt
        prompt = self._construct_prompt(task, observation, step_num)
        
        # Generate response
        inputs = self._tokenizer(prompt, return_tensors="pt")
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
        
        # Parse response into reasoning and action
        reasoning, action = self._parse_response(response)
        
        # Token usage
        token_usage = {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
        }
        
        return reasoning, action, token_usage
    
    def _construct_prompt(self, task: str, observation: dict, step_num: int) -> str:
        """Construct the prompt for the agent."""
        # ReAct-style prompt
        prompt = f"""You are a web navigation agent. Complete the following task:

Task: {task}

Current URL: {observation.get('url', 'unknown')}

Page Content (Accessibility Tree):
{observation.get('dom', '')[:4000]}

Think step by step about what action to take next, then provide your action.

Format your response as:
Thought: [your reasoning]
Action: [action_type][selector or value]

Available actions:
- click[element_id] - Click on an element
- type[element_id][text] - Type text into an element
- scroll[direction] - Scroll up or down
- goto[url] - Navigate to a URL
- stop[answer] - Complete the task

Step {step_num}:
"""
        return prompt
    
    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response into reasoning and action."""
        reasoning = ""
        action = ""
        
        # Look for Thought/Reasoning section
        thought_match = re.search(
            r"(?:Thought|Reasoning):\s*(.+?)(?=Action:|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        if thought_match:
            reasoning = thought_match.group(1).strip()
        
        # Look for Action section
        action_match = re.search(
            r"Action:\s*(.+?)(?:\n|$)",
            response,
            re.IGNORECASE
        )
        if action_match:
            action = action_match.group(1).strip()
        else:
            # Try to find any action pattern
            action_pattern = re.search(
                r"(click|type|scroll|goto|stop)\s*[\[\(]",
                response,
                re.IGNORECASE
            )
            if action_pattern:
                # Extract from this point
                start = action_pattern.start()
                action = response[start:].split("\n")[0].strip()
        
        return reasoning, action
    
    def _format_action(self, action: ActionRecord) -> str:
        """Format ActionRecord as string for BrowserGym."""
        if action.bid:
            if action.value:
                return f"{action.type.value}(bid='{action.bid}', text='{action.value}')"
            return f"{action.type.value}(bid='{action.bid}')"
        elif action.selector:
            if action.value:
                return f"{action.type.value}[{action.selector}][{action.value}]"
            return f"{action.type.value}[{action.selector}]"
        return action.raw_action or ""
    
    def _classify_failure(self, outcome: TaskOutcome) -> Optional[FailureType]:
        """
        Classify failure type based on trace patterns.
        
        Note: This is a placeholder. Full classification will be done
        during annotation phase using the failure taxonomy.
        """
        # For now, mark as natural (uninjected) failure
        # Detailed classification happens in annotation phase
        return FailureType.NATURAL


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
