"""
Module for custom types or signature
"""

from typing import Callable, TypeVar

import numpy as np
import tensorflow as tf

OperatorSignature = Callable[[tf.keras.Model, tf.Tensor, tf.Tensor], float]

DatasetOrTensor = TypeVar(
    "DatasetOrTensor",
    tf.Tensor,
    np.ndarray,
    "torch.Tensor",
    tf.data.Dataset,
    "torch.utils.data.DataLoader",
)
