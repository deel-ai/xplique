"""
Tests for Kernel Shap Module
"""
import numpy as np
import tensorflow as tf

from xplique.attributions import KernelShap
from ..utils import generate_data, generate_model, almost_equal

def test_get_probs():
    """
    Ensure the get_probs function output has the right shape and return
    right values for simple example
    """
    num_features = 5
    probs = KernelShap._get_probs_nb_selected_feature(num_features)
    expected_probs = np.array([0.,1.,2/3,2/3,1.])
    assert almost_equal(probs.numpy(),expected_probs, epsilon=1e-5)
    num_features = [4,6,9,7]
    for num_feature in num_features:
        probs = KernelShap._get_probs_nb_selected_feature(num_feature)
        assert probs.shape == num_feature

def test_pertub_func():
    """
    Ensure the pertub function output has the right shape and
    values belong to (0, num_interp_features)
    """
    interpret_samples = KernelShap._kernel_shap_pertub_func(10,20)
    assert interpret_samples.shape == (20,10)
    for sample in interpret_samples:
        assert tf.less(tf.reduce_sum(sample, axis=0),10)
        assert tf.greater(tf.reduce_sum(sample, axis=0),0)

def test_output_explain():
    """The output shape must be the same as the input shape, except for the channels"""
    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    def map_four_by_four(inp):

        width = inp.shape[0]
        height = inp.shape[1]

        mapping = np.zeros((width,height))
        for i in range(width):
            if i%2 != 0:
                mapping[i] = mapping[i-1]
            else:
                for j in range(height):
                    mapping[i][j] = (width/2) * (i//2) + (j//2)

        mapping = tf.cast(mapping, dtype=tf.int32)
        return mapping

    for input_shape in input_shapes:
        samples, labels = generate_data(input_shape, nb_labels, 20)
        model = generate_model(input_shape, nb_labels)

        method = KernelShap(model, map_to_interpret_space=map_four_by_four, nb_samples=10)

        explanations = method.explain(samples, labels)
        assert samples.shape[:3] == explanations.shape

def test_inputs_batching():
    """ Ensure, that we can call explain with batched inputs """
    nb_labels = 10

    samples, labels = generate_data((32, 32, 3), nb_labels, 200)
    model = generate_model((32, 32, 3), nb_labels)

    method = KernelShap(model,
                    batch_size=15,
                    nb_samples=20)

    explanations = method.explain(samples, labels)

    assert samples.shape[:3] == explanations.shape
