"""
Test MMDCritic
"""
import os
import sys
sys.path.append(os.getcwd())
import time

from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

from xplique.example_based.projections import CustomProjection
from xplique.example_based.search_methods import MMDCritic

from tests.utils import get_Gaussian_Data, load_data, plot

def test_mmd_critic_basic():
    """
    Test the Prototypes with gaussian kernel.
    """
    # Setup
    k = 3
    gamma = 0.026

    x_train, y_train = get_Gaussian_Data(nb_samples_class=20)
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

    kernel_type = "global"

    # Method initialization
    method = MMDCritic(
            cases_dataset=x_train,
            labels_dataset=y_train,
            k=k,
            projection=identity_projection,
            search_returns=["indices"],
            kernel=kernel,
            kernel_type=kernel_type
        )
        
    # Generate global explanation
    start_time = time.time()
    prototype_indices = method.find_examples(None)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    prototypes = tf.gather(x_train, prototype_indices)
    prototype_labels = tf.gather(y_train, prototype_indices)

    # sort by label
    prototype_labels_sorted = prototype_labels.numpy().argsort()

    prototypes = tf.gather(prototypes, prototype_labels_sorted)
    prototype_indices = tf.gather(prototype_indices, prototype_labels_sorted)
    prototype_labels = tf.gather(prototype_labels, prototype_labels_sorted)

    # Verifications
    # Shape
    assert prototype_indices.shape == (k,)
    assert prototypes.shape == (k, x_train.shape[1])

    # at least 1 prototype per class is selected
    assert tf.unique(prototype_labels)[0].shape == tf.unique(y_train)[0].shape

    # uniqueness test of prototypes
    assert prototype_indices.shape == tf.unique(prototype_indices)[0].shape

    # Check if all indices are between 0 and x_train.shape[0]-1
    assert tf.reduce_all(tf.math.logical_and(prototype_indices >= 0, prototype_indices <= x_train.shape[0]-1))

    # # Visualize all prototypes
    # plot(prototypes, None, 'mmd_critic')


def test_mmd_critic_all_are_prototypes():
    """
    Test the Prototypes with gaussian kernel.
    """
    # Setup
    k = 60
    gamma = 0.026

    x_train, y_train = get_Gaussian_Data(nb_samples_class=20)
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

    kernel_type = "local"

    # Method initialization
    method = MMDCritic(
            cases_dataset=x_train,
            labels_dataset=y_train,
            k=k,
            projection=identity_projection,
            search_returns=["indices"],
            kernel=kernel,
            kernel_type=kernel_type
        )
        
    # Generate global explanation
    start_time = time.time()
    prototype_indices = method.find_examples(None)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    prototypes = tf.gather(x_train, prototype_indices)
    prototype_labels = tf.gather(y_train, prototype_indices)

    # sort by label
    prototype_labels_sorted = prototype_labels.numpy().argsort()

    prototypes = tf.gather(prototypes, prototype_labels_sorted)
    prototype_indices = tf.gather(prototype_indices, prototype_labels_sorted)
    prototype_labels = tf.gather(prototype_labels, prototype_labels_sorted)

    # Verifications
    # Shape
    assert prototype_indices.shape == (k,)
    assert prototypes.shape == (k, x_train.shape[1])

    # at least 1 prototype per class is selected
    assert tf.unique(prototype_labels)[0].shape == tf.unique(y_train)[0].shape

    # uniqueness test of prototypes
    assert prototype_indices.shape == tf.unique(prototype_indices)[0].shape

    # Check if all indices are between 0 and x_train.shape[0]-1
    assert tf.reduce_all(tf.math.logical_and(prototype_indices >= 0, prototype_indices <= x_train.shape[0]-1))

    # # Visualize all prototypes
    # plot(prototypes, None, 'mmd_critic')

test_mmd_critic_basic()
test_mmd_critic_all_are_prototypes()