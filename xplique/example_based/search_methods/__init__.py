"""
Search methods
"""

from .base import ORDER, BaseSearchMethod
from .kleor import KLEORGlobalSimSearch, KLEORSimMissSearch
from .knn import KNN, BaseKNN, FilterKNN
from .mmd_critic_search import MMDCriticSearch
from .proto_dash_search import ProtoDashSearch

# from .sklearn_knn import SklearnKNN
from .proto_greedy_search import ProtoGreedySearch
