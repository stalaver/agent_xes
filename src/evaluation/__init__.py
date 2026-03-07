"""
Evaluation Framework - Uniform baseline comparison for failure detection

Provides data splitting, metric computation, and experiment orchestration
for comparing failure-detection baselines at varying prefix lengths.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from src.evaluation.data_split import DataSplit, DataSplitter
from src.evaluation.experiment import ExperimentResults, ExperimentRunner
from src.evaluation.metrics import MetricsCalculator

__all__ = [
    "DataSplit",
    "DataSplitter",
    "ExperimentResults",
    "ExperimentRunner",
    "MetricsCalculator",
]
