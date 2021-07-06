import tensorflow as tf

from xplique.commons import tensor_sanitize

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
            inputs_sanitize, targets_sanitize = tensor_sanitize(inputs, targets)

            assert isinstance(inputs_sanitize, tf.Tensor)
            assert isinstance(targets_sanitize, tf.Tensor)

            assert len(inputs_sanitize) == nb_samples
            assert len(targets_sanitize) == nb_samples
