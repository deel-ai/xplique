"""
Test Prototypes
"""
import os
import sys

sys.path.append(os.getcwd())

from math import prod, sqrt
import unittest
import time

import numpy as np
import tensorflow as tf

from xplique.commons import sanitize_dataset, are_dataset_first_elems_equal
from xplique.types import Union

from xplique.example_based import Prototypes, ProtoGreedy, ProtoDash, MMDCritic
from xplique.example_based.projections import Projection, LatentSpaceProjection

from tests.utils import almost_equal, get_Gaussian_Data, load_data, plot, plot_local_explanation


def test_proto_greedy_basic():
    """
    Test the Prototypes with an identity projection.
    """
    # Setup
    k = 3
    nb_prototypes = 3
    gamma = 0.026
    x_train, y_train = get_Gaussian_Data(nb_samples_class=20)
    x_test, y_test = get_Gaussian_Data(nb_samples_class=10)
    # x_train, y_train = load_data('usps')
    # x_test, y_test = load_data('usps.t')
    # x_test = tf.random.shuffle(x_test)
    # x_test = x_test[0:8]

    identity_projection = Projection(
        space_projection=lambda inputs, targets=None: inputs
    )

    kernel_type = "global"

    # Method initialization
    method = ProtoGreedy(
        cases_dataset=x_train,
        labels_dataset=y_train,
        k=k,
        projection=identity_projection,
        batch_size=32,
        distance=None, #"euclidean",
        nb_prototypes=nb_prototypes,
        kernel_type=kernel_type,
        gamma=gamma,
    )

    # Generate global explanation
    prototype_indices, prototype_weights = method.get_global_prototypes()

    prototypes = tf.gather(x_train, prototype_indices)
    prototype_labels = tf.gather(y_train, prototype_indices)

    # sort by label
    prototype_labels_sorted = prototype_labels.numpy().argsort()

    prototypes = tf.gather(prototypes, prototype_labels_sorted)
    prototype_indices = tf.gather(prototype_indices, prototype_labels_sorted)
    prototype_labels = tf.gather(prototype_labels, prototype_labels_sorted)
    prototype_weights = tf.gather(prototype_weights, prototype_labels_sorted)

    # Verifications
    # Shape
    assert prototype_indices.shape == (nb_prototypes,)
    assert prototypes.shape == (nb_prototypes, x_train.shape[1])
    assert prototype_weights.shape == (nb_prototypes,)

    # at least 1 prototype per class is selected
    assert tf.unique(prototype_labels)[0].shape == tf.unique(y_train)[0].shape

    # uniqueness test of prototypes
    assert prototype_indices.shape == tf.unique(prototype_indices)[0].shape

    # Check if all indices are between 0 and x_train.shape[0]-1
    assert tf.reduce_all(tf.math.logical_and(prototype_indices >= 0, prototype_indices <= x_train.shape[0]-1))

    # Generate local explanation
    examples = method.explain(x_test)

    # # Visualize all prototypes
    # plot(prototypes, prototype_weights, 'proto_greedy')

    # # Visualize local explanation
    # plot_local_explanation(examples, x_test, 'proto_greedy')

def test_proto_dash_basic():
    """
    Test the Prototypes with an identity projection.
    """
    # Setup
    k = 3
    nb_prototypes = 3
    gamma = 0.026
    x_train, y_train = get_Gaussian_Data(nb_samples_class=20)
    x_test, y_test = get_Gaussian_Data(nb_samples_class=10)
    # x_train, y_train = load_data('usps')
    # x_test, y_test = load_data('usps.t')
    # x_test = tf.random.shuffle(x_test)
    # x_test = x_test[0:8]

    identity_projection = Projection(
        space_projection=lambda inputs, targets=None: inputs
    )

    kernel_type = "global"

    # Method initialization
    method = ProtoDash(
        cases_dataset=x_train,
        labels_dataset=y_train,
        k=k,
        projection=identity_projection,
        batch_size=32,
        distance="euclidean",
        nb_prototypes=nb_prototypes,
        kernel_type=kernel_type,
        gamma=gamma,
    )

    # Generate global explanation
    prototype_indices, prototype_weights = method.get_global_prototypes()

    prototypes = tf.gather(x_train, prototype_indices)
    prototype_labels = tf.gather(y_train, prototype_indices)

    # sort by label
    prototype_labels_sorted = prototype_labels.numpy().argsort()

    prototypes = tf.gather(prototypes, prototype_labels_sorted)
    prototype_indices = tf.gather(prototype_indices, prototype_labels_sorted)
    prototype_labels = tf.gather(prototype_labels, prototype_labels_sorted)
    prototype_weights = tf.gather(prototype_weights, prototype_labels_sorted)

    # Verifications
    # Shape
    assert prototype_indices.shape == (nb_prototypes,)
    assert prototypes.shape == (nb_prototypes, x_train.shape[1])
    assert prototype_weights.shape == (nb_prototypes,)

    # at least 1 prototype per class is selected
    assert tf.unique(prototype_labels)[0].shape == tf.unique(y_train)[0].shape

    # uniqueness test of prototypes
    assert prototype_indices.shape == tf.unique(prototype_indices)[0].shape

    # Check if all indices are between 0 and x_train.shape[0]-1
    assert tf.reduce_all(tf.math.logical_and(prototype_indices >= 0, prototype_indices <= x_train.shape[0]-1))

    # Generate local explanation
    examples = method.explain(x_test)

    # # Visualize all prototypes
    # plot(prototypes, prototype_weights, 'proto_dash')

    # # Visualize local explanation
    # plot_local_explanation(examples, x_test, 'proto_dash')

def test_mmd_critic_basic():
    """
    Test the Prototypes with an identity projection.
    """
    # Setup
    k = 3
    nb_prototypes = 3
    gamma = 0.026
    x_train, y_train = get_Gaussian_Data(nb_samples_class=20)
    x_test, y_test = get_Gaussian_Data(nb_samples_class=10)
    # x_train, y_train = load_data('usps')
    # x_test, y_test = load_data('usps.t')
    # x_test = tf.random.shuffle(x_test)
    # x_test = x_test[0:8]

    identity_projection = Projection(
        space_projection=lambda inputs, targets=None: inputs
    )

    kernel_type = "global"

    # Method initialization
    method = MMDCritic(
        cases_dataset=x_train,
        labels_dataset=y_train,
        k=k,
        projection=identity_projection,
        batch_size=32,
        distance="euclidean",
        nb_prototypes=nb_prototypes,
        kernel_type=kernel_type,
        gamma=gamma,
    )

    # Generate global explanation
    prototype_indices, prototype_weights = method.get_global_prototypes()

    prototypes = tf.gather(x_train, prototype_indices)
    prototype_labels = tf.gather(y_train, prototype_indices)

    # sort by label
    prototype_labels_sorted = prototype_labels.numpy().argsort()

    prototypes = tf.gather(prototypes, prototype_labels_sorted)
    prototype_indices = tf.gather(prototype_indices, prototype_labels_sorted)
    prototype_labels = tf.gather(prototype_labels, prototype_labels_sorted)
    prototype_weights = tf.gather(prototype_weights, prototype_labels_sorted)

    # Verifications
    # Shape
    assert prototype_indices.shape == (nb_prototypes,)
    assert prototypes.shape == (nb_prototypes, x_train.shape[1])
    assert prototype_weights.shape == (nb_prototypes,)

    # at least 1 prototype per class is selected
    assert tf.unique(prototype_labels)[0].shape == tf.unique(y_train)[0].shape

    # uniqueness test of prototypes
    assert prototype_indices.shape == tf.unique(prototype_indices)[0].shape

    # Check if all indices are between 0 and x_train.shape[0]-1
    assert tf.reduce_all(tf.math.logical_and(prototype_indices >= 0, prototype_indices <= x_train.shape[0]-1))

    # Generate local explanation
    examples = method.explain(x_test)

    # # Visualize all prototypes
    # plot(prototypes, prototype_weights, 'mmd_critic')

    # # Visualize local explanation
    # plot_local_explanation(examples, x_test, 'mmd_critic')

# test_proto_greedy_basic()
# test_proto_dash_basic()
# test_mmd_critic_basic()
