"""
Search methods
"""

from .base import BaseSearchMethod

# from .sklearn_knn import SklearnKNN
from .knn import KNN
from .prototypes_search import PrototypesSearch
from .proto_greedy_search import ProtoGreedySearch
from .proto_dash_search import ProtoDashSearch
from .mmd_critic_search import MMDCriticSearch
