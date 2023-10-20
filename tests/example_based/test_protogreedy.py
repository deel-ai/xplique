"""
Test Protogreedy
"""
import os
import sys
sys.path.append(os.getcwd())
import time

from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

from xplique.example_based.projections import CustomProjection
from xplique.example_based.search_methods import Protogreedy

from tests.utils import get_Gaussian_Data, load_data, plot


def test_protogreedy_basic():
    """
    Test the Prototypes with an identity projection.
    """
    # Setup
    k = 32
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
    method = Protogreedy(
            cases_dataset=x_train,
            labels_dataset=y_train,
            k=k,
            projection=identity_projection,
            search_returns=["indices","weights"],
            kernel=kernel,
            kernel_type=kernel_type
        )
      
    # Generate global explanation
    start_time = time.time()
    method_output = method.find_examples(None)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    prototype_indices = method_output["indices"]
    prototype_weights = method_output["weights"]

    prototypes = tf.gather(x_train, prototype_indices)
    prototype_labels = tf.gather(y_train, prototype_indices)

    # sort by label indices
    sorted_by_y_indices = prototype_labels.numpy().argsort()
    prototypes = tf.gather(prototypes, sorted_by_y_indices)
    prototype_labels = tf.gather(prototype_labels, sorted_by_y_indices)
    prototype_weights = tf.gather(prototype_weights, sorted_by_y_indices)

    # Verifications
    # Shape
    assert prototype_indices.shape == (k,)
    assert prototypes.shape == (k, x_train.shape[1])
    assert prototype_weights.shape == (k,)

    # at least 1 prototype per class is selected
    assert tf.unique(prototype_labels)[0].shape == tf.unique(y_train)[0].shape

    # uniqueness test of prototypes
    assert prototype_indices.shape == tf.unique(prototype_indices)[0].shape

    # # Visualize all prototypes
    # plot(prototypes, prototype_weights, 'protogreedy')

test_protogreedy_basic()