import os
import random

import numpy as np
import tensorflow as tf

from xplique.attributions import Rise
from ..utils import generate_data, generate_model, almost_equal


def test_output_shape():
    """The output size (h, w) must be the same as the input"""

    input_shapes = [(28, 28, 1), (32, 32, 3), (30, 50, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        method = Rise(model, nb_samples=100)
        rise_maps = method.explain(x, y)
        assert x.shape[:-1] == rise_maps.shape[:-1]

        method = Rise(model, nb_samples=100, grid_size=(3, 5))
        rise_maps = method.explain(x, y)
        assert x.shape[:-1] == rise_maps.shape[:-1]


def reset_random_seed(seed: int = 0):
   os.environ['PYTHONHASHSEED'] = str(seed)
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)


def test_polymorphic_parameters():
    """Ensure we could pass tuple or int to define grid_size parameter when inputs are images"""
    grid_size = 3
    nb_samples = 10
    preservation_probability = 0.5

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        features, targets = generate_data(input_shape, nb_labels, 20)
        model = generate_model(input_shape, nb_labels)

        reset_random_seed(0)
        rise_int = Rise(model, grid_size=grid_size, nb_samples=nb_samples)
        reset_random_seed(0)
        rise_tuple = Rise(model, grid_size=(grid_size, grid_size), nb_samples=nb_samples)

        reset_random_seed(0)
        mask_int = tf.cast(
            rise_int._get_masks(features.shape, nb_samples, grid_size, preservation_probability), 
            tf.int32)
            
        reset_random_seed(0)
        mask_tuple = tf.cast(
            rise_tuple._get_masks(features.shape, nb_samples, grid_size, preservation_probability), 
            tf.int32)

        rise_int(features, targets)
        rise_tuple(features, targets)

        assert almost_equal(mask_int, mask_tuple)
