import tensorflow as tf
import numpy as np

from xplique.commons import tensor_sanitize, numpy_sanitize

from ..utils import generate_data


def test_tensor_sanitize():
    """Ensure we get tf.Tensor for numpy array, tf tensor and tf.data.Dataset"""
    nb_samples = 71
    inputs_shapes = [
        (32, 32, 1), (32, 32, 3)
    ]

    for shape in inputs_shapes:
        inputs_np, targets_np = generate_data(shape, 10, nb_samples)
        inputs_tf, targets_tf = tf.cast(inputs_np, tf.float32), tf.cast(targets_np, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np))
        dataset_batched = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np)).batch(10)

        for inputs, targets in [
            (inputs_np, targets_np),
            (inputs_tf, targets_tf),
            (dataset, None),
            (dataset_batched, None),
        ]:
            inputs_sanitize_tf, targets_sanitize_tf = tensor_sanitize(inputs, targets)
            inputs_sanitize_np, targets_sanitize_np = numpy_sanitize(inputs, targets)

            assert isinstance(inputs_sanitize_tf, tf.Tensor)
            assert isinstance(targets_sanitize_tf, tf.Tensor)
            assert isinstance(inputs_sanitize_np, np.ndarray)
            assert isinstance(targets_sanitize_np, np.ndarray)

            assert len(inputs_sanitize_tf) == nb_samples
            assert len(targets_sanitize_tf) == nb_samples
            assert len(inputs_sanitize_np) == nb_samples
            assert len(targets_sanitize_np) == nb_samples
