"""
Search methods
"""

from .base import BaseSearchMethod, ORDER

# from .sklearn_knn import SklearnKNN
from .proto_greedy_search import ProtoGreedySearch
from .proto_dash_search import ProtoDashSearch
from .mmd_critic_search import MMDCriticSearch
from .knn import BaseKNN, KNN, FilterKNN
from .kleor import KLEORSimMiss, KLEORGlobalSim
