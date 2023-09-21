"""
Test MMDCritic
"""
import os
import sys
sys.path.append(os.getcwd())

from math import prod, sqrt, ceil

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics.pairwise import rbf_kernel
from pathlib import Path
import tensorflow as tf

from xplique.example_based.projections import CustomProjection
from xplique.example_based.search_methods import Protogreedy
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

def load_data(fname):
    data_dir = Path('/home/mohamed-chafik.bakey/MMD-critic/data')
    X, y = load_svmlight_file(str(data_dir / fname))  
    X = tf.constant(X.todense(), dtype=tf.float32)
    y = tf.constant(np.array(y), dtype=tf.int64)
    sort_indices = y.numpy().argsort()
    X = tf.gather(X, sort_indices, axis=0)
    y = tf.gather(y, sort_indices)
    y -= 1
    return X, y

def test_protogreedy_basic():
    """
    Test the Prototypes with an identity projection.
    """
    # Setup
    nb_features = 256
    k = 32 # the number of prototypes
    gamma = 0.026
    kernel_type = "local"

    x_train, y_train, _, _ = get_setup(nb_features)
    # x_train, y_train = load_data('usps')
    # x_test, y_test = load_data('usps.t')
    # x_test = tf.random.shuffle(x_test)
    # x_test = x_test[0:8]

    identity_projection = CustomProjection(space_projection=lambda inputs, targets=None: inputs)

    def custom_kernel_wrapper(gamma):
        def custom_kernel(x,y=None):
            return rbf_kernel(x,y,gamma)
        return custom_kernel
    
    kernel = custom_kernel_wrapper(gamma)
    # kernel = rbf_kernel(x_train)

    # Method initialization
    method = Protogreedy(
            cases_dataset=x_train,
            labels_dataset=y_train,
            k=k,
            projection=identity_projection,
            search_returns=["indices","weights"],
            kernel=kernel,
            kernel_type="local"
        )
    
    # Generate global explanation
    method_output = method.find_examples(None)
    prototype_indices = method_output["indices"]
    prototype_weights = method_output["weights"]

    prototypes = tf.gather(x_train, prototype_indices)
    prototype_labels = tf.gather(y_train, prototype_indices)
    sorted_by_y_indices = prototype_labels.numpy().argsort()
    prototypes_sorted = tf.gather(prototypes, sorted_by_y_indices)
    prototype_labels = tf.gather(prototype_labels, sorted_by_y_indices)

    best_distance = method.compute_weighted_MMD_distance(prototype_indices, prototype_weights)

    for i in range(10):
        # select a random set
        random_indices = tf.random.uniform(shape=(k,), minval=0, maxval=x_train.shape[0], dtype=tf.int32)
        
        # select random weights
        random_weights = tf.random.uniform(shape=(k,), minval=0, maxval=1, dtype=tf.float32)
        normalized_weights = random_weights / tf.reduce_sum(random_weights)

        # compute mmd distance for this random set
        distance = method.compute_weighted_MMD_distance(random_indices, normalized_weights)

        assert best_distance <= distance


# test_protogreedy_basic()