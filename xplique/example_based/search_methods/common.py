"""
Common functions for search methods.
"""
# pylint: disable=invalid-name

import numpy as np
import tensorflow as tf

from ...types import Callable, Union


def _manhattan_distance(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    """
    Compute the Manhattan distance between two vectors.

    Parameters
    ----------
    x1 : tf.Tensor
        First vector.
    x2 : tf.Tensor
        Second vector.

    Returns
    -------
    tf.Tensor
        Manhattan distance between the two vectors.
    """
    return tf.reduce_sum(tf.abs(x1 - x2), axis=-1)


def _euclidean_distance(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    """
    Compute the Euclidean distance between two vectors.

    Parameters
    ----------
    x1 : tf.Tensor
        First vector.
    x2 : tf.Tensor
        Second vector.

    Returns
    -------
    tf.Tensor
        Euclidean distance between the two vectors.
    """
    return tf.norm(x1 - x2, ord="euclidean", axis=-1)


def _cosine_distance(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    """
    Compute the cosine distance between two vectors.

    Parameters
    ----------
    x1 : tf.Tensor
        First vector.
    x2 : tf.Tensor
        Second vector.

    Returns
    -------
    tf.Tensor
        Cosine distance between the two vectors.
    """
    return 1 - tf.reduce_sum(x1 * x2, axis=-1) / (
        tf.norm(x1, axis=-1) * tf.norm(x2, axis=-1)
    )


def _chebyshev_distance(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    """
    Compute the Chebyshev distance between two vectors.

    Parameters
    ----------
    x1 : tf.Tensor
        First vector.
    x2 : tf.Tensor
        Second vector.

    Returns
    -------
    tf.Tensor
        Chebyshev distance between the two vectors.
    """
    return tf.reduce_max(tf.abs(x1 - x2), axis=-1)


def _minkowski_distance(x1: tf.Tensor, x2: tf.Tensor, p: int) -> tf.Tensor:
    """
    Compute the Minkowski distance between two vectors.

    Parameters
    ----------
    x1 : tf.Tensor
        First vector.
    x2 : tf.Tensor
        Second vector.
    p : int
        Order of the Minkowski distance.

    Returns
    -------
    tf.Tensor
        Minkowski distance between the two vectors.
    """
    return tf.norm(x1 - x2, ord=p, axis=-1)


_distances = {
    "manhattan": _manhattan_distance,
    "euclidean": _euclidean_distance,
    "cosine": _cosine_distance,
    "chebyshev": _chebyshev_distance,
    "inf": _chebyshev_distance,
}


def get_distance_function(distance: Union[int, str, Callable] = "euclidean",) -> Callable:
    """
    Function to obtain a distance function from different inputs.

    Parameters
    ----------
    distance : Union[int, str, Callable], optional
        Distance function to use. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable,
        by default "euclidean".
    """
    # set distance function
    if hasattr(distance, "__call__"):
        return distance
    if isinstance(distance, str) and distance in _distances:
        return _distances[distance]
    if isinstance(distance, int):
        return lambda x1, x2: _minkowski_distance(x1, x2, p=distance)
    if distance == np.inf:
        return _chebyshev_distance

    raise AttributeError(
        "The distance parameter is expected to be either a Callable, "\
        + f" an integer, 'inf', or a string in {_distances.keys()}. "\
        + f"But a {type(distance)} was received, with value {distance}."
    )
