"""
Detection package - Online failure detection for web navigation agents

Provides pattern matching, threshold-based failure prediction, and
token savings estimation for the Phase 3A online detection system.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from src.detection.failure_predictor import (
    Decision,
    FailurePredictor,
    PredictionResult,
)
from src.detection.pattern_matcher import PatternMatch, PatternMatcher
from src.detection.token_calculator import SavingsEstimate, TokenCalculator

__all__ = [
    "Decision",
    "FailurePredictor",
    "PatternMatch",
    "PatternMatcher",
    "PredictionResult",
    "SavingsEstimate",
    "TokenCalculator",
]
