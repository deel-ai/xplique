"""
Functions to ease attributions
"""

from .segmentation import get_class_zone, get_common_border, get_connected_zone, get_in_out_border

__all__ = [
    "get_class_zone",
    "get_connected_zone",
    "get_common_border",
    "get_in_out_border",
]
