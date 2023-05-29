"""
Test Cole
"""
from math import prod, sqrt

import numpy as np
from sklearn.metrics import DistanceMetric
import tensorflow as tf

from xplique.example_based import Cole
from xplique.types import Union

from ..utils import generate_data, generate_model, almost_equal, generate_agnostic_model


def test_neighbors_distance():
    """
    The function test every output of the explanation method
    """
    # Method parameters initialisation
    input_shape = (3, 3, 1)
    nb_labels = 10
    nb_samples = 10
    nb_samples_test = 8
    k = 3

    # Data generation
    matrix_train = tf.stack([i * tf.ones(input_shape) for i in range(nb_samples)])
    matrix_test = matrix_train[1:-1]
    labels_train = tf.range(nb_samples)
    labels_test = labels_train[1:-1]

    # Model generation
    model = generate_model(input_shape, nb_labels)

    # Initialisation of weights_extraction_function and distance_function
    # They will be used in CaseBasedExplainer initialisation
    distance_function = DistanceMetric.get_metric("euclidean")

    # CaseBasedExplainer initialisation
    method = Cole(
        model,
        matrix_train,
        labels_train,
        targets=None,
        distance_function=distance_function,
        weights_extraction_function=lambda inputs, targets: tf.ones(inputs.shape),
    )

    # Method explanation
    (
        examples,
        examples_distance,
        examples_weights,
        inputs_weights,
        examples_labels,
    ) = method.explain(matrix_test, labels_test)

    # test every outputs shape
    assert examples.shape == (nb_samples_test, k) + input_shape
    assert examples_distance.shape == (nb_samples_test, k)
    assert examples_weights.shape == (nb_samples_test, k) + input_shape
    assert inputs_weights.shape == (nb_samples_test,) + input_shape
    assert examples_labels.shape == (nb_samples_test, k)

    for i in range(len(labels_test)):
        # test examples:
        assert almost_equal(examples[i][0], matrix_train[i + 1])
        assert almost_equal(examples[i][1], matrix_train[i + 2]) or almost_equal(
            examples[i][1], matrix_train[i]
        )
        assert almost_equal(examples[i][2], matrix_train[i]) or almost_equal(
            examples[i][2], matrix_train[i + 2]
        )

        # test examples_distance
        assert almost_equal(examples_distance[i][0], 0)
        assert almost_equal(examples_distance[i][1], sqrt(prod(input_shape)))
        assert almost_equal(examples_distance[i][2], sqrt(prod(input_shape)))

        # test examples_labels
        assert almost_equal(examples_labels[i][0], labels_train[i + 1])
        assert almost_equal(examples_labels[i][1], labels_train[i + 2]) or almost_equal(
            examples_labels[i][1], labels_train[i]
        )
        assert almost_equal(examples_labels[i][2], labels_train[i]) or almost_equal(
            examples_labels[i][2], labels_train[i + 2]
        )


def weights_attribution(
    inputs: Union[tf.Tensor, np.ndarray], targets: Union[tf.Tensor, np.ndarray]
):
    """
    Custom weights extraction function
    Zeros everywhere and target at 0, 0, 0
    """
    weights = tf.Variable(tf.zeros(inputs.shape, dtype=tf.float32))
    weights[:, 0, 0, 0].assign(targets)
    return weights


def test_weights_attribution():
    """
    Function to test the weights attribution
    """
    # Method parameters initialisation
    input_shape = (3, 3, 1)
    nb_labels = 10
    nb_samples = 10

    # Data generation
    matrix_train = tf.stack(
        [i * tf.ones(input_shape, dtype=tf.float32) for i in range(nb_samples)]
    )
    matrix_test = matrix_train[1:-1]
    labels_train = tf.range(nb_samples, dtype=tf.float32)
    labels_test = labels_train[1:-1]

    # Model generation
    model = generate_model(input_shape, nb_labels)

    # Initialisation of distance_function
    # It will be used in CaseBasedExplainer initialisation
    distance_function = DistanceMetric.get_metric("euclidean")

    # CaseBasedExplainer initialisation
    method = Cole(
        model,
        matrix_train,
        labels_train,
        targets=labels_train,
        distance_function=distance_function,
        weights_extraction_function=weights_attribution,
    )

    # test case dataset weigth
    assert almost_equal(method.case_dataset_weight[:, 0, 0, 0], method.labels_train)
    assert almost_equal(
        tf.reduce_sum(method.case_dataset_weight, axis=[1, 2, 3]), method.labels_train
    )

    # Method explanation
    _, _, examples_weights, inputs_weights, examples_labels =\
        method.explain(matrix_test, labels_test)

    # test examples weights
    assert almost_equal(examples_weights[:, :, 0, 0, 0], examples_labels)
    assert almost_equal(
        tf.reduce_sum(examples_weights, axis=[2, 3, 4]), examples_labels
    )

    # test inputs weights
    assert almost_equal(inputs_weights[:, 0, 0, 0], labels_test)
    assert almost_equal(tf.reduce_sum(inputs_weights, axis=[1, 2, 3]), labels_test)


def test_tabular_inputs():
    """
    Function to test the acceptation of tabular data input in the method
    """
    # Method parameters initialisation
    data_shape = (3,)
    input_shape = data_shape
    nb_labels = 3
    nb_samples = 20
    nb_inputs = 5
    k = 3

    # Data generation
    dataset, targets = generate_data(data_shape, nb_labels, nb_samples)
    dataset_train = dataset[:-nb_inputs]
    dataset_test = dataset[-nb_inputs:]
    targets_train = targets[:-nb_inputs]
    targets_test = targets[-nb_inputs:]

    # Model generation
    model = generate_agnostic_model(input_shape, nb_labels)

    # Initialisation of weights_extraction_function and distance_function
    # They will be used in CaseBasedExplainer initialisation
    distance_function = DistanceMetric.get_metric("euclidean")

    # CaseBasedExplainer initialisation
    method = Cole(
        model,
        dataset_train,
        targets_train,
        targets=targets_train,
        distance_function=distance_function,
        weights_extraction_function=lambda inputs, targets: tf.ones(inputs.shape),
        k=k,
    )

    # Method explanation
    examples, _, _, _, _ = method.explain(dataset_test, targets_test)

    # test examples shape
    assert examples.shape == (nb_inputs, k) + input_shape
