"""
Search methods
"""

from .base import BaseSearchMethod, ORDER

from .knn import BaseKNN, KNN, FilterKNN
from .kleor import KLEORSimMiss, KLEORGlobalSim
