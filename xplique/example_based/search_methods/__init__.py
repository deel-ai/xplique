"""
Search methods
"""

from .base import BaseSearchMethod

# from .sklearn_knn import SklearnKNN
from .knn import KNN
from .protogreedy import Protogreedy
from .mmd_critic import MMDCritic
from .protodash import Protodash
