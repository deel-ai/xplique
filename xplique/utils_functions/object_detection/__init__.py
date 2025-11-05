"""
Object detection utilities
"""

from .object_detection_operators import _EPSILON, _box_iou, _format_objects

__all__ = [
    "_box_iou",
    "_format_objects",
    "_EPSILON",
]
