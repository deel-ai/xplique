"""
Test MMDCritic
"""
import os
import sys
sys.path.append(os.getcwd())

from math import prod, sqrt, ceil

import numpy as np
import scipy
from sklearn.datasets import load_svmlight_file
from sklearn.metrics.pairwise import rbf_kernel
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from xplique.attributions import Occlusion, Saliency

from xplique.example_based import Prototypes
from xplique.example_based.projections import CustomProjection
from xplique.example_based.search_methods import MMDCritic
from xplique.types import Union

from tests.utils import almost_equal


def get_setup(nb_features, nb_samples_train=100, nb_samples_test=100, nb_labels=10): 
    """
    Generate data and model for Prototypes
    """
    # Training Data generation
    x_train = tf.constant(np.random.rand(nb_samples_train, nb_features).astype(np.float32))
    y_train = tf.constant(np.random.randint(0, nb_labels, size=nb_samples_train, dtype=np.int64))
    sort_indices = y_train.numpy().argsort()
    x_train = tf.gather(x_train, sort_indices, axis=0)
    y_train = tf.gather(y_train, sort_indices)
    # Test Data generation
    x_test = tf.constant(np.random.rand(nb_samples_test, nb_features).astype(np.float32))
    y_test = tf.constant(np.random.randint(0, nb_labels, size=nb_samples_test, dtype=np.int64))
    sort_indices = y_test.numpy().argsort()
    x_test = tf.gather(x_test, sort_indices, axis=0)
    y_test = tf.gather(y_test, sort_indices)

    return x_train, y_train, x_test, y_test

def test_prototypes_basic():
    """
    Test the Prototypes with an identity projection.
    """
    # Setup
    nb_features = 256
    k = 32 # the number of prototypes
    gamma = 0.026
    kernel_type = "local"

    x_train, y_train, _, _ = get_setup(nb_features)

    identity_projection = CustomProjection(space_projection=lambda inputs, targets=None: inputs)

    def custom_kernel_wrapper(gamma):
        def custom_kernel(x,y=None):
            return rbf_kernel(x,y,gamma)
        return custom_kernel
    
    kernel = custom_kernel_wrapper(gamma)
    # kernel = rbf_kernel(x_train)

    # Method initialization
    method = Prototypes(cases_dataset=x_train,
                             labels_dataset=y_train,                                                        
                             projection=identity_projection,
                             search_method=MMDCritic,
                             k=k,
                             case_returns=["indices","distances"],
                             kernel=kernel,
                             kernel_type=kernel_type)
    
    # Generate global explanation
    method_output = method.explain()
    prototype_indices = method_output["indices"]

    prototypes = tf.gather(x_train, prototype_indices)
    prototype_labels = tf.gather(y_train, prototype_indices)
    sorted_by_y_indices = prototype_labels.numpy().argsort()
    prototypes_sorted = tf.gather(prototypes, sorted_by_y_indices)
    prototype_labels = tf.gather(prototype_labels, sorted_by_y_indices)

    kernel_matrix = MMDCritic.compute_kernel_matrix(x_train, y_train, kernel, kernel_type)

    best_distance = MMDCritic.compute_MMD2(kernel_matrix, prototype_indices)

    for i in range(10):
        # select a random set of prototypes
        prototype_indices_random = tf.random.uniform(shape=(k,), minval=0, maxval=x_train.shape[0], dtype=tf.int32)
        # compute mmd2 distance for this random set
        distance = MMDCritic.compute_MMD2(kernel_matrix, prototype_indices_random)

        assert best_distance <= distance


# test_prototypes_basic()