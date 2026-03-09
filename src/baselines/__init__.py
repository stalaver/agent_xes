"""
Baselines package - Failure detection baselines for uniform evaluation

Exports all baseline implementations and a registry dict for convenient
lookup by name.

Author: Sergio Talavera
Project: Early Failure Detection in Web Navigation Agents via Closed Sequential Pattern Mining
"""

from src.baselines.base import BaseBaseline
from src.baselines.bide_coverage import BIDECoverageBaseline
from src.baselines.bilstm import BiLSTMBaseline
from src.baselines.deeplog import DeepLogBaseline
from src.baselines.frequency_vector import FrequencyVectorBaseline
from src.baselines.ngram import NGramBaseline
from src.baselines.process_conformance import ProcessConformanceBaseline
from src.baselines.taspm import TaSPMBaseline

BASELINES: dict[str, type[BaseBaseline]] = {
    FrequencyVectorBaseline.name: FrequencyVectorBaseline,
    NGramBaseline.name: NGramBaseline,
    TaSPMBaseline.name: TaSPMBaseline,
    ProcessConformanceBaseline.name: ProcessConformanceBaseline,
    DeepLogBaseline.name: DeepLogBaseline,
    BiLSTMBaseline.name: BiLSTMBaseline,
    BIDECoverageBaseline.name: BIDECoverageBaseline,
}

__all__ = [
    "BaseBaseline",
    "BIDECoverageBaseline",
    "BiLSTMBaseline",
    "DeepLogBaseline",
    "FrequencyVectorBaseline",
    "NGramBaseline",
    "TaSPMBaseline",
    "ProcessConformanceBaseline",
    "BASELINES",
]
