"""
Test Cole
"""
import os
import sys
sys.path.append(os.getcwd())

from math import prod, sqrt

import numpy as np
import scipy
import tensorflow as tf

from xplique.attributions import Occlusion, Saliency

from xplique.example_based import Cole, SimilarExamples
from xplique.example_based.projections import CustomProjection
from xplique.example_based.search_methods import SklearnKNN
from xplique.plots.image import plot_examples

from tests.utils import generate_data, generate_model, almost_equal, generate_timeseries_model


def get_setup(input_shape, nb_samples=10, nb_labels=10):
    """
    Generate data and model for Cole
    """
    # Data generation
    x_train = tf.stack([i * tf.ones(input_shape, tf.float32) for i in range(nb_samples)])
    x_test = x_train[1:-1]
    y_train = tf.one_hot(tf.range(len(x_train)) % nb_labels, depth=nb_labels)

    # Model generation
    model = generate_model(input_shape, nb_labels)

    return model, x_train, x_test, y_train


def test_cole_spliting():
    """
    Test examples plot function.
    """
    # Setup
    nb_samples = 10
    input_shape = (6, 6, 3)
    nb_labels = 5
    k = 1
    x_train = tf.random.uniform((nb_samples,) + input_shape, minval=0, maxval=1)
    x_test = tf.random.uniform((nb_samples,) + input_shape, minval=0, maxval=1)
    labels = tf.one_hot(indices=tf.repeat(input=tf.range(nb_labels), 
                                          repeats=[nb_samples // nb_labels]), 
                        depth=nb_labels)
    y_train = labels
    y_test = tf.random.shuffle(labels)

    # Model generation
    model = generate_model(input_shape, nb_labels)

    # Cole with attribution method constructor
    method = Cole(case_dataset=x_train,
                  labels_dataset=tf.argmax(y_train, axis=1),
                  dataset_targets=y_train,
                  search_method=SklearnKNN,
                  k=k,
                  case_returns="all",
                  model=model,
                  latent_layer="last_conv",
                  attribution_method=Occlusion,
                  patch_size=2,
                  patch_stride=1)

    # Generate explanation
    outputs = method.explain(x_test, y_test)

    # get predictions on examples
    predicted_labels = tf.map_fn(
        fn=lambda x: tf.cast(tf.argmax(model(x), axis=1), tf.int32),
        elems=outputs["examples"],
        fn_output_signature=tf.int32,
    )

    # test plot
    plot_examples(test_labels=tf.argmax(y_test, axis=1), 
                  predicted_labels=predicted_labels,
                  **outputs)
