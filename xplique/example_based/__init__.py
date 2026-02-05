"""
Example-based methods available
"""

from .counterfactuals import LabelAwareCounterFactuals, NaiveCounterFactuals
from .prototypes import MMDCritic, ProtoDash, ProtoGreedy, Prototypes
from .semifactuals import KLEORGlobalSim, KLEORSimMiss
from .similar_examples import Cole, SimilarExamples

__all__ = [
    "LabelAwareCounterFactuals",
    "NaiveCounterFactuals",
    "MMDCritic",
    "ProtoDash",
    "ProtoGreedy",
    "Prototypes",
    "KLEORGlobalSim",
    "KLEORSimMiss",
    "Cole",
    "SimilarExamples",
]
