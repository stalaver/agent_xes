"""
Baselines package - Failure detection baselines for uniform evaluation

Exports all baseline implementations and a registry dict for convenient
lookup by name.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from src.baselines.base import BaseBaseline
from src.baselines.frequency_vector import FrequencyVectorBaseline
from src.baselines.ngram import NGramBaseline

BASELINES: dict[str, type[BaseBaseline]] = {
    FrequencyVectorBaseline.name: FrequencyVectorBaseline,
    NGramBaseline.name: NGramBaseline,
}

__all__ = [
    "BaseBaseline",
    "FrequencyVectorBaseline",
    "NGramBaseline",
    "BASELINES",
]
