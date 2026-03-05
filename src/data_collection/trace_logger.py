"""
Trace Logger for Web Agent Execution Traces

This module handles:
- Real-time trace capture during agent execution
- Persistent storage (JSON Lines format)
- Trace retrieval and filtering
- Statistics and summaries

Storage Format:
- Individual traces: data/raw_traces/{benchmark}/{model}/{trace_id}.json
- Batch traces: data/raw_traces/{benchmark}/{model}/batch_{date}.jsonl

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Callable
from collections import defaultdict
import threading
import gzip

from .trace_schema import (
    AgentTrace, TraceStep, TraceMetadata, 
    TaskOutcome, FailureType,
    generate_trace_id, get_current_timestamp
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TraceLogger:
    """
    Logger for capturing and storing agent execution traces.
    
    Features:
    - Real-time step-by-step logging
    - Automatic file organization by benchmark/model
    - JSON and JSON Lines output formats
    - Thread-safe for parallel agent execution
    - Compression support for large datasets
    """
    
    def __init__(
        self,
        base_dir: str = "data/raw_traces",
        benchmark: str = "webarena",
        model: str = "unknown",
        use_compression: bool = False,
        auto_save: bool = True,
        buffer_size: int = 10,
    ):
        """
        Initialize the trace logger.
        
        Args:
            base_dir: Base directory for trace storage
            benchmark: Benchmark name (webarena, browsergym, miniwob, workarena)
            model: Model name (llama-3.2-3b, qwen-2.5-7b, mistral-7b)
            use_compression: Whether to gzip compress output files
            auto_save: Automatically save trace when finalized
            buffer_size: Number of traces to buffer before batch write
        """
        self.base_dir = Path(base_dir)
        self.benchmark = benchmark
        self.model = model
        self.use_compression = use_compression
        self.auto_save = auto_save
        self.buffer_size = buffer_size
        
        # Current trace being recorded
        self._current_trace: Optional[AgentTrace] = None
        self._trace_lock = threading.Lock()
        
        # Batch buffer for JSON Lines output
        self._batch_buffer: list[AgentTrace] = []
        self._batch_lock = threading.Lock()
        
        # Statistics
        self._stats = defaultdict(int)
        
        # Ensure output directory exists
        self._output_dir = self.base_dir / benchmark / model
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TraceLogger initialized: {self._output_dir}")
    
    @property
    def output_dir(self) -> Path:
        """Get the output directory for traces."""
        return self._output_dir
    
    # =========================================================================
    # Trace Lifecycle Methods
    # =========================================================================
    
    def start_trace(
        self,
        task_id: str,
        task_description: str,
        website: str,
        model: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """
        Start recording a new trace.
        
        Args:
            task_id: Unique task identifier
            task_description: Human-readable task description
            website: Target website (e.g., shopping.webarena.dev)
            model: Model name (defaults to logger's model)
            trace_id: Optional custom trace ID
            
        Returns:
            The trace ID
        """
        with self._trace_lock:
            if self._current_trace is not None:
                logger.warning("Previous trace not finalized, auto-finalizing...")
                self._finalize_current_trace(TaskOutcome.ERROR)
            
            trace_id = trace_id or generate_trace_id()
            
            self._current_trace = AgentTrace(
                metadata=TraceMetadata(
                    trace_id=trace_id,
                    task_id=task_id,
                    task_description=task_description,
                    website=website,
                    model=model or self.model,
                    outcome=TaskOutcome.UNKNOWN,
                    start_time=get_current_timestamp(),
                    benchmark=self.benchmark,
                )
            )
            
            self._stats["traces_started"] += 1
            logger.info(f"Started trace {trace_id} for task {task_id}")
            
            return trace_id
    
    def log_step(self, step: TraceStep) -> None:
        """
        Log a single step to the current trace.
        
        Args:
            step: The TraceStep to log
        """
        with self._trace_lock:
            if self._current_trace is None:
                raise RuntimeError("No active trace. Call start_trace() first.")
            
            self._current_trace.add_step(step)
            self._stats["steps_logged"] += 1
            
            logger.debug(f"Logged step {step.step_number}: {step.action.type.value}")
    
    def finalize_trace(
        self,
        outcome: TaskOutcome,
        failure_type: Optional[FailureType] = None,
        annotation_notes: Optional[str] = None,
    ) -> AgentTrace:
        """
        Finalize the current trace with outcome information.
        
        Args:
            outcome: Task outcome (success, failure, timeout, error)
            failure_type: Failure category if outcome is failure
            annotation_notes: Optional notes for annotation
            
        Returns:
            The finalized AgentTrace
        """
        with self._trace_lock:
            trace = self._finalize_current_trace(
                outcome, failure_type, annotation_notes
            )
            return trace
    
    def _finalize_current_trace(
        self,
        outcome: TaskOutcome,
        failure_type: Optional[FailureType] = None,
        annotation_notes: Optional[str] = None,
    ) -> AgentTrace:
        """Internal method to finalize trace (must hold lock)."""
        if self._current_trace is None:
            raise RuntimeError("No active trace to finalize.")
        
        # Update metadata
        self._current_trace.metadata.outcome = outcome
        self._current_trace.metadata.failure_type = failure_type
        self._current_trace.metadata.annotation_notes = annotation_notes
        self._current_trace.metadata.end_time = get_current_timestamp()
        
        # Calculate duration
        if self._current_trace.metadata.start_time:
            start = datetime.fromisoformat(self._current_trace.metadata.start_time)
            end = datetime.fromisoformat(self._current_trace.metadata.end_time)
            self._current_trace.metadata.duration_seconds = (end - start).total_seconds()
        
        trace = self._current_trace
        self._current_trace = None
        
        # Update stats
        self._stats["traces_completed"] += 1
        self._stats[f"outcome_{outcome.value}"] += 1
        if failure_type:
            self._stats[f"failure_{failure_type.value}"] += 1
        
        logger.info(
            f"Finalized trace {trace.metadata.trace_id}: "
            f"{outcome.value}, {trace.total_steps} steps"
        )
        
        # Auto-save if enabled
        if self.auto_save:
            self.save_trace(trace)
        
        return trace
    
    # =========================================================================
    # Storage Methods
    # =========================================================================
    
    def save_trace(self, trace: AgentTrace, filename: Optional[str] = None) -> Path:
        """
        Save a single trace to a JSON file.
        
        Args:
            trace: The AgentTrace to save
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"{trace.metadata.trace_id}.json"
        
        filepath = self._output_dir / filename
        
        if self.use_compression:
            filepath = filepath.with_suffix(".json.gz")
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                f.write(trace.to_json())
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(trace.to_json())
        
        logger.debug(f"Saved trace to {filepath}")
        return filepath
    
    def save_batch_jsonl(
        self, 
        traces: list[AgentTrace], 
        filename: Optional[str] = None
    ) -> Path:
        """
        Save multiple traces to a JSON Lines file.
        
        Args:
            traces: List of traces to save
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_{timestamp}.jsonl"
        
        filepath = self._output_dir / filename
        
        if self.use_compression:
            filepath = filepath.with_suffix(".jsonl.gz")
            open_func = lambda p: gzip.open(p, "wt", encoding="utf-8")
        else:
            open_func = lambda p: open(p, "w", encoding="utf-8")
        
        with open_func(filepath) as f:
            for trace in traces:
                f.write(json.dumps(trace.to_dict()) + "\n")
        
        logger.info(f"Saved {len(traces)} traces to {filepath}")
        return filepath
    
    def add_to_batch(self, trace: AgentTrace) -> Optional[Path]:
        """
        Add a trace to the batch buffer, auto-flush if full.
        
        Args:
            trace: The trace to add
            
        Returns:
            Path to saved file if buffer was flushed, None otherwise
        """
        with self._batch_lock:
            self._batch_buffer.append(trace)
            
            if len(self._batch_buffer) >= self.buffer_size:
                return self._flush_batch()
        
        return None
    
    def _flush_batch(self) -> Path:
        """Flush the batch buffer to disk (must hold lock)."""
        if not self._batch_buffer:
            return None
        
        filepath = self.save_batch_jsonl(self._batch_buffer)
        self._batch_buffer = []
        return filepath
    
    def flush(self) -> Optional[Path]:
        """Flush any remaining traces in the batch buffer."""
        with self._batch_lock:
            return self._flush_batch()
    
    # =========================================================================
    # Retrieval Methods
    # =========================================================================
    
    def load_trace(self, trace_id: str) -> Optional[AgentTrace]:
        """
        Load a trace by ID from the output directory.
        
        Args:
            trace_id: The trace ID to load
            
        Returns:
            The loaded AgentTrace or None if not found
        """
        # Try regular JSON
        filepath = self._output_dir / f"{trace_id}.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return AgentTrace.from_json(f.read())
        
        # Try compressed
        filepath = self._output_dir / f"{trace_id}.json.gz"
        if filepath.exists():
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                return AgentTrace.from_json(f.read())
        
        logger.warning(f"Trace {trace_id} not found")
        return None
    
    def iter_traces(
        self,
        directory: Optional[Path] = None,
        filter_fn: Optional[Callable[[AgentTrace], bool]] = None,
    ) -> Iterator[AgentTrace]:
        """
        Iterate over all traces in a directory.
        
        Args:
            directory: Directory to scan (defaults to output_dir)
            filter_fn: Optional filter function
            
        Yields:
            AgentTrace objects
        """
        directory = directory or self._output_dir
        
        # Process individual JSON files
        for filepath in directory.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    trace = AgentTrace.from_json(f.read())
                if filter_fn is None or filter_fn(trace):
                    yield trace
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        
        # Process compressed files
        for filepath in directory.glob("*.json.gz"):
            try:
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    trace = AgentTrace.from_json(f.read())
                if filter_fn is None or filter_fn(trace):
                    yield trace
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        
        # Process JSON Lines files
        for filepath in list(directory.glob("*.jsonl")) + list(directory.glob("*.jsonl.gz")):
            try:
                if filepath.suffix == ".gz":
                    open_func = lambda: gzip.open(filepath, "rt", encoding="utf-8")
                else:
                    open_func = lambda: open(filepath, "r", encoding="utf-8")
                
                with open_func() as f:
                    for line in f:
                        if line.strip():
                            trace = AgentTrace.from_dict(json.loads(line))
                            if filter_fn is None or filter_fn(trace):
                                yield trace
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
    
    def count_traces(self, directory: Optional[Path] = None) -> dict:
        """
        Count traces by outcome and failure type.
        
        Args:
            directory: Directory to scan
            
        Returns:
            Dictionary with counts
        """
        counts = defaultdict(int)
        
        for trace in self.iter_traces(directory):
            counts["total"] += 1
            counts[f"outcome_{trace.metadata.outcome.value}"] += 1
            if trace.metadata.failure_type:
                counts[f"failure_{trace.metadata.failure_type.value}"] += 1
            counts[f"website_{trace.metadata.website}"] += 1
            counts[f"model_{trace.metadata.model}"] += 1
        
        return dict(counts)
    
    # =========================================================================
    # Statistics and Reporting
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get current session statistics."""
        return dict(self._stats)
    
    def print_stats(self):
        """Print current session statistics."""
        print("\n" + "="*50)
        print("Trace Logger Statistics")
        print("="*50)
        for key, value in sorted(self._stats.items()):
            print(f"  {key}: {value}")
        print("="*50)


class TraceLoggerContext:
    """
    Context manager for trace logging.
    
    Usage:
        with TraceLoggerContext(logger, task_id, ...) as trace_id:
            for step in agent.run():
                logger.log_step(step)
        # Trace is automatically finalized
    """
    
    def __init__(
        self,
        trace_logger: TraceLogger,
        task_id: str,
        task_description: str,
        website: str,
        model: Optional[str] = None,
    ):
        self.logger = trace_logger
        self.task_id = task_id
        self.task_description = task_description
        self.website = website
        self.model = model
        self.trace_id: Optional[str] = None
        self._outcome = TaskOutcome.UNKNOWN
        self._failure_type = None
    
    def __enter__(self) -> str:
        self.trace_id = self.logger.start_trace(
            self.task_id, self.task_description, self.website, self.model
        )
        return self.trace_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._outcome = TaskOutcome.ERROR
        
        self.logger.finalize_trace(self._outcome, self._failure_type)
        return False  # Don't suppress exceptions
    
    def set_outcome(self, outcome: TaskOutcome, failure_type: Optional[FailureType] = None):
        """Set the outcome before exiting the context."""
        self._outcome = outcome
        self._failure_type = failure_type


# Example usage
if __name__ == "__main__":
    from .trace_schema import ActionRecord, ObservationRecord, ReasoningRecord, ActionType, ElementState
    
    # Create logger
    logger_instance = TraceLogger(
        base_dir="data/raw_traces",
        benchmark="webarena",
        model="llama-3.2-3b",
        auto_save=True,
    )
    
    # Start a trace
    trace_id = logger_instance.start_trace(
        task_id="task_001",
        task_description="Find and purchase the cheapest laptop",
        website="shopping.webarena.dev",
    )
    
    # Log some steps
    for i in range(5):
        step = TraceStep(
            step_number=i + 1,
            reasoning=ReasoningRecord(
                raw_reasoning=f"Step {i+1}: Looking for element...",
                intent="search",
            ),
            action=ActionRecord(
                type=ActionType.CLICK,
                selector=f"#element-{i}",
            ),
            observation=ObservationRecord(
                element_found=True,
                element_state=ElementState.VISIBLE,
                http_status=200,
            ),
            dom_hash=f"hash_{i}",
            timestamp=get_current_timestamp(),
        )
        logger_instance.log_step(step)
    
    # Finalize
    trace = logger_instance.finalize_trace(
        outcome=TaskOutcome.FAILURE,
        failure_type=FailureType.NAVIGATION,
    )
    
    # Print stats
    logger_instance.print_stats()
    
    # Load and verify
    loaded = logger_instance.load_trace(trace_id)
    if loaded:
        print(f"\n✓ Successfully loaded trace with {loaded.total_steps} steps")