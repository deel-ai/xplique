"""
Example-based methods available
"""

from .similar_examples import SimilarExamples, Cole
from .prototypes import Prototypes, ProtoGreedy, ProtoDash, MMDCritic
from .counterfactuals import NaiveCounterFactuals, LabelAwareCounterFactuals
from .semifactuals import KLEORGlobalSim, KLEORSimMiss
