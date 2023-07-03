"""
Module for custom types or signature
"""
from typing import Callable
import tensorflow as tf

OperatorSignature = Callable[[tf.keras.Model, tf.Tensor, tf.Tensor], float]
