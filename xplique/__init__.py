"""
Xplique
-------

The goal of Xplique is to provide a simple interface to the latest explanation
techniques
"""

__version__ = "1.5.2"

from . import attributions, commons, concepts, example_based, features_visualizations, plots
from .commons import Tasks

__all__ = [
    "attributions",
    "commons",
    "concepts",
    "example_based",
    "features_visualizations",
    "plots",
    "Tasks",
]
