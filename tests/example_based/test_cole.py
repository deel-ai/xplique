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

from xplique.example_based import Cole
from xplique.example_based.projections import Projection
from xplique.example_based.search_methods import SklearnKNN
from xplique.types import Union

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



def test_cole_basic():
    """
    Function to test the Cole method with a simple weighting
    """
    # Setup
    input_shape = (4, 4, 1)
    k = 3
    model, x_train, x_test, _ = get_setup(input_shape)

    # Method initialization
    method = Cole(model=model,
                  case_dataset=x_train,
                  projection=lambda inputs, targets=None: inputs,
                  search_method=SklearnKNN,
                  k=k,
                  distance="euclidean")

    # Generate explanation
    examples = method.explain(x_test)

    # Verifications
    # Shape should be (n, k, h, w, c)
    assert examples.shape == (len(x_test), k) + input_shape

    for i in range(len(x_test)):
        # test examples:
        assert almost_equal(examples[i, 0], x_train[i + 1])
        assert almost_equal(examples[i, 1], x_train[i + 2])\
            or almost_equal(examples[i, 1], x_train[i])
        assert almost_equal(examples[i, 2], x_train[i])\
            or almost_equal(examples[i, 2], x_train[i + 2])


def test_cole_return_multiple_elements():
    """
    ...
    Try to return distances and more, it is useful for plots
    test modifying k
    """
    # Setup
    input_shape = (5, 5, 1)
    k = 3
    model, x_train, x_test, y_train = get_setup(input_shape)

    nb_samples_test = len(x_test)
    assert nb_samples_test + 2 == len(y_train)

    # Method initialization
    method = Cole(model=model,
                  case_dataset=(x_train, y_train),
                  projection=Projection(None, None),
                  search_method=SklearnKNN,
                  k=1,
                  distance="euclidean")

    method.set_returns("all")
    
    method.set_k(k)

    # Generate explanation
    method_output = method.explain(x_test)

    assert isinstance(method_output, dict)

    examples = method_output["examples"]
    weights = method_output["weights"]
    distances = method_output["distances"]
    indices = method_output["indices"]
    labels = method_output["labels"]

    # test every outputs shape (with the include inputs)
    assert examples.shape == (nb_samples_test, k + 1) + input_shape
    assert weights.shape == (nb_samples_test, k + 1) + input_shape
    # the inputs distance ae zero and indices do not exist
    assert distances.shape == (nb_samples_test, k)
    assert indices.shape == (nb_samples_test, k)
    assert labels.shape == (nb_samples_test, k)

    for i in range(nb_samples_test):
        # test examples:
        assert almost_equal(examples[i, 0], x_test[i])
        assert almost_equal(examples[i, 1], x_train[i + 1])
        assert almost_equal(examples[i, 2], x_train[i + 2])\
            or almost_equal(examples[i, 2], x_train[i])
        assert almost_equal(examples[i, 3], x_train[i])\
            or almost_equal(examples[i, 3], x_train[i + 2])

        # test weights
        assert almost_equal(weights[i], tf.ones(weights[i].shape, dtype=tf.float32))

        # test distances
        assert almost_equal(distances[i, 0], 0)
        assert almost_equal(distances[i, 1], sqrt(prod(input_shape)))
        assert almost_equal(distances[i, 2], sqrt(prod(input_shape)))

        # test indices
        assert almost_equal(indices[i, 0], i + 1)
        assert almost_equal(indices[i, 1], i) or almost_equal(indices[i, 1], i + 2)
        assert almost_equal(indices[i, 2], i) or almost_equal(indices[i, 2], i + 2)

        # test labels
        assert almost_equal(labels[i, 0], y_train[i + 1])
        assert almost_equal(labels[i, 1], y_train[i]) or almost_equal(labels[i, 1], y_train[i + 2])
        assert almost_equal(labels[i, 2], y_train[i]) or almost_equal(labels[i, 2], y_train[i + 2])


def test_cole_weighting():
    """
    ...
    test if the weighting is indeed used
    """
    # Setup
    input_shape = (4, 4, 1)
    nb_samples = 10
    k = 3
    model, x_train, x_test, y_train = get_setup(input_shape, nb_samples)

    # Define the weighing function
    weights = np.zeros(x_train[0].shape)
    weights[1] = np.ones(weights[1].shape)

    # create huge noise on non interesting features
    noise = np.random.uniform(size=x_train.shape, low=-100, high=100)
    x_train = weights * np.array(x_train) +  (1 - weights) * noise

    weighting_function = Projection(weights=weights).project

    method = Cole(model=model,
                  case_dataset=(x_train, y_train),
                  projection=weighting_function,
                  search_method=SklearnKNN,
                  k=k,
                  distance="euclidean")

    # Generate explanation
    examples = method.explain(x_test)

    # Verifications
    # Shape should be (n, k, h, w, c)
    nb_samples_test = x_test.shape[0]
    assert examples.shape == (nb_samples_test, k) + input_shape

    for i in range(nb_samples_test):
        # test examples:
        assert almost_equal(examples[i, 0], x_train[i + 1])
        assert almost_equal(examples[i, 1], x_train[i + 2])\
            or almost_equal(examples[i, 1], x_train[i])
        assert almost_equal(examples[i, 2], x_train[i])\
            or almost_equal(examples[i, 2], x_train[i + 2])


def test_cole_attribution():
    """
    ...
    test if the weighting is indeed used
    """
    # Setup
    nb_samples = 20
    input_shape = (5, 5)
    nb_labels = 10
    k = 3
    x_train = tf.random.uniform((nb_samples,) + input_shape, minval=-1, maxval=1, seed=0)
    x_test = tf.random.uniform((nb_samples,) + input_shape, minval=-1, maxval=1, seed=2)
    labels = tf.one_hot(indices=tf.repeat(input=tf.range(nb_labels), 
                                           repeats=[nb_samples // nb_labels]), 
                         depth=nb_labels)
    y_train = labels
    y_test = tf.random.shuffle(labels, seed=1)
    
    # Model generation
    model = generate_timeseries_model(input_shape, nb_labels)

    # Cole with attribution method constructor
    method_constructor = Cole(case_dataset=(x_train, y_train),
                              search_method=SklearnKNN,
                              k=k,
                              distance="cosine",
                              model=model,
                              attribution_method=Saliency)

    # Cole with attribution explain
    attribution_method = Saliency(model)
    projection = lambda inputs, targets: inputs * attribution_method(inputs, targets)

    method_call = Cole(case_dataset=x_train,
                       dataset_targets=y_train,
                       search_method=SklearnKNN,
                       k=k,
                       distance=scipy.spatial.distance.cosine,
                       projection=projection)
    
    method_different_distance = Cole(case_dataset=(x_train, y_train),
                                     search_method=SklearnKNN,
                                     k=k,
                                     model=model,
                                     attribution_method=Saliency)

    # Generate explanation
    examples_constructor = method_constructor.explain(x_test, y_test)
    examples_call = method_call.explain(x_test, y_test)
    examples_different_distance = method_different_distance(x_test, y_test)

    # Verifications
    # Shape should be (n, k, h, w, c)
    assert examples_constructor.shape == (len(x_test), k) + input_shape
    assert examples_call.shape == (len(x_test), k) + input_shape
    assert examples_different_distance.shape == (len(x_test), k) + input_shape

    # both methods should be the same
    assert almost_equal(examples_constructor, examples_call)

    # a different distance should give different results
    assert not almost_equal(examples_constructor, examples_different_distance)
    
    # TODO Check weights are equal to the attribution directly on the input


def test_cole_spliting():
    """
    ...
    attribution with model splitting, return weights and channels
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
    method = Cole(case_dataset=(x_train, y_train),
                  search_method=SklearnKNN,
                  k=k,
                  returns=["examples", "weights", "include_inputs"],
                  model=model,
                  latent_layer="last_conv",
                  attribution_method=Occlusion,
                  patch_size=2,
                  patch_stride=1)

    # Generate explanation
    outputs = method.explain(x_test, y_test)
    examples, weights = outputs["examples"], outputs["weights"]

    # Verifications
    # Shape should be (n, k, h, w, c)
    nb_samples_test = x_test.shape[0]
    assert examples.shape == (nb_samples_test, k + 1) + input_shape
    assert weights.shape[:-1] == (nb_samples_test, k + 1) + input_shape[:-1]