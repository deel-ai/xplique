"""
Example-based methods available
"""

from .cole import Cole
from .similar_examples import SimilarExamples
from .prototypes import Prototypes, ProtoGreedy, ProtoDash, MMDCritic
from .contrastive_examples import NaiveCounterFactuals, LabelAwareCounterFactuals, KLEOR
