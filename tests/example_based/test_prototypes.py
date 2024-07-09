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

from tests.utils import almost_equal, get_gaussian_data, load_data, plot, plot_local_explanation


def test_prototypes_global_explanations_basic():
    """
    Test prototypes shapes and uniqueness.
    """
    # Setup
    k = 3
    nb_prototypes = 5
    nb_classes =  3

    gamma = 0.026
    x_train, y_train = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=20)
    x_test, y_test = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=10)

    identity_projection = Projection(
        space_projection=lambda inputs, targets=None: inputs
    )

    for kernel_type in ["local", "global"]:
        for method_class in [ProtoGreedy, ProtoDash, MMDCritic]:
            # compute general prototypes
            method = method_class(
                cases_dataset=x_train,
                labels_dataset=y_train,
                k=k,
                projection=identity_projection,
                batch_size=8,
                distance="euclidean",
                nb_prototypes=nb_prototypes,
                kernel_type=kernel_type,
                gamma=gamma,
            )
            # extract prototypes
            prototypes_dict = method.get_global_prototypes()
            prototypes = prototypes_dict["prototypes"]
            prototypes_indices = prototypes_dict["prototypes_indices"]
            prototypes_labels = prototypes_dict["prototypes_labels"]
            prototypes_weights = prototypes_dict["prototypes_weights"]

            # check shapes
            assert prototypes.shape == (nb_prototypes,) + x_train.shape[1:]
            assert prototypes_indices.shape == (nb_prototypes,)
            assert prototypes_labels.shape == (nb_prototypes,)
            assert prototypes_weights.shape == (nb_prototypes,)

            # check uniqueness
            assert len(prototypes_indices) == len(tf.unique(prototypes_indices)[0])

            # for each prototype
            for i in range(nb_prototypes):
                # check prototypes are in the dataset and correspond to the index
                assert tf.reduce_all(tf.equal(prototypes[i], x_train[prototypes_indices[i]]))

                # same for labels
                assert tf.reduce_all(tf.equal(prototypes_labels[i], y_train[prototypes_indices[i]]))

                # check indices are in the dataset
                assert prototypes_indices[i] >= 0 and prototypes_indices[i] < x_train.shape[0]


def test_prototypes_local_explanations_basic():
    """
    Test prototypes local explanations.
    """
    # Setup
    k = 3
    nb_prototypes = 5
    nb_classes =  3
    batch_size = 8

    gamma = 0.026
    x_train, y_train = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=20)
    x_test, y_test = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=10)

    identity_projection = Projection(
        space_projection=lambda inputs, targets=None: inputs
    )

    for kernel_type in ["local", "global"]:
        for method_class in [ProtoGreedy, ProtoDash, MMDCritic]:
            # compute general prototypes
            method = method_class(
                cases_dataset=x_train,
                labels_dataset=y_train,
                k=k,
                projection=identity_projection,
                case_returns=["examples", "distances", "labels", "indices"],
                batch_size=batch_size,
                distance="euclidean",
                nb_prototypes=nb_prototypes,
                kernel_type=kernel_type,
                gamma=gamma,
            )
            # extract prototypes
            prototypes_dict = method.get_global_prototypes()
            prototypes = prototypes_dict["prototypes"]
            prototypes_indices = prototypes_dict["prototypes_indices"]
            prototypes_labels = prototypes_dict["prototypes_labels"]
            prototypes_weights = prototypes_dict["prototypes_weights"]

            # compute local explanations
            outputs = method.explain(x_test)
            examples = outputs["examples"]
            distances = outputs["distances"]
            labels = outputs["labels"]
            indices = outputs["indices"]

            # check shapes
            assert examples.shape == (x_test.shape[0], k) + x_train.shape[1:]
            assert distances.shape == (x_test.shape[0], k)
            assert labels.shape == (x_test.shape[0], k)
            assert indices.shape == (x_test.shape[0], k, 2)

            # for each sample
            for i in range(x_test.shape[0]):
                # check first closest prototype label is the same as the sample label
                assert tf.reduce_all(tf.equal(labels[i, 0], y_test[i]))

                for j in range(k):
                    # check indices in prototypes' indices
                    index = indices[i, j, 0] * batch_size + indices[i, j, 1]
                    assert index in prototypes_indices

                    # check examples are in prototypes
                    assert tf.reduce_all(tf.equal(prototypes[prototypes_indices == index], examples[i, j]))

                    # check indices are in the dataset
                    assert tf.reduce_all(tf.equal(x_train[index], examples[i, j]))

                    # check distances
                    assert almost_equal(distances[i, j], tf.norm(x_test[i] - x_train[index]), epsilon=1e-5)

                    # check labels
                    assert tf.reduce_all(tf.equal(labels[i, j], y_train[index]))


def test_prototypes_global_sanity_checks_1():
    """
    Test prototypes global explanations sanity checks.
    
    Check 1: For n separated gaussians, for n requested prototypes, there should be 1 prototype per gaussian.
    """

    # Setup
    k = 3
    nb_prototypes = 3

    gamma = 0.026
    x_train, y_train = get_gaussian_data(nb_classes=nb_prototypes, nb_samples_class=20)

    identity_projection = Projection(
        space_projection=lambda inputs, targets=None: inputs
    )

    for kernel_type in ["local", "global"]:
        for method_class in [ProtoGreedy, ProtoDash, MMDCritic]:
            # compute general prototypes
            method = method_class(
                cases_dataset=x_train,
                labels_dataset=y_train,
                k=k,
                projection=identity_projection,
                batch_size=8,
                distance="euclidean",
                nb_prototypes=nb_prototypes,
                kernel_type=kernel_type,
                gamma=gamma,
            )
            # extract prototypes
            prototypes_dict = method.get_global_prototypes()
            prototypes = prototypes_dict["prototypes"]
            prototypes_indices = prototypes_dict["prototypes_indices"]
            prototypes_labels = prototypes_dict["prototypes_labels"]
            prototypes_weights = prototypes_dict["prototypes_weights"]

            # check 1
            assert len(tf.unique(prototypes_labels)[0]) == nb_prototypes
            

def test_prototypes_global_sanity_checks_2():
    """
    Test prototypes global explanations sanity checks.

    Check 2: With local kernel_type, if there are more requested prototypes than classes, there should be at least 1 prototype per class.    
    """
    
    # Setup
    k = 3
    nb_prototypes = 5
    nb_classes = 3

    gamma = 0.026
    x_train, y_train = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=20)

    # randomize y_train
    y_train = tf.random.shuffle(y_train)

    identity_projection = Projection(
        space_projection=lambda inputs, targets=None: inputs
    )

    for method_class in [ProtoGreedy, ProtoDash, MMDCritic]:
        # compute general prototypes
        method = method_class(
            cases_dataset=x_train,
            labels_dataset=y_train,
            k=k,
            projection=identity_projection,
            batch_size=8,
            distance="euclidean",
            nb_prototypes=nb_prototypes,
            kernel_type="local",
            gamma=gamma,
        )
        # extract prototypes
        prototypes_dict = method.get_global_prototypes()
        prototypes = prototypes_dict["prototypes"]
        prototypes_indices = prototypes_dict["prototypes_indices"]
        prototypes_labels = prototypes_dict["prototypes_labels"]
        prototypes_weights = prototypes_dict["prototypes_weights"]

        # check 2
        assert len(tf.unique(prototypes_labels)[0]) == nb_classes


def test_prototypes_local_explanations_with_projection():
    """
    Test prototypes local explanations with a projection.
    """
    # Setup
    k = 3
    nb_prototypes = 5
    nb_classes = 3
    batch_size = 8

    gamma = 0.026
    x_train, y_train = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=20)
    x_train_bis, _ = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=20)
    x_train = tf.concat([x_train, x_train_bis], axis=1)  # make a dataset with two dimensions

    x_test, y_test = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=10)

    projection = Projection(
        space_projection=lambda inputs, targets=None: tf.reduce_mean(inputs, axis=1, keepdims=True)
    )

    for kernel_type in ["local", "global"]:
        for method_class in [ProtoGreedy, ProtoDash, MMDCritic]:
            # compute general prototypes
            method = method_class(
                cases_dataset=x_train,
                labels_dataset=y_train,
                k=k,
                projection=projection,
                case_returns=["examples", "distances", "labels", "indices"],
                batch_size=batch_size,
                distance="euclidean",
                nb_prototypes=nb_prototypes,
                kernel_type=kernel_type,
                gamma=gamma,
            )
            # extract prototypes
            prototypes_dict = method.get_global_prototypes()
            prototypes = prototypes_dict["prototypes"]
            prototypes_indices = prototypes_dict["prototypes_indices"]
            prototypes_labels = prototypes_dict["prototypes_labels"]
            prototypes_weights = prototypes_dict["prototypes_weights"]

            # check shapes
            assert prototypes.shape == (nb_prototypes,) + x_train.shape[1:]
            assert prototypes_indices.shape == (nb_prototypes,)
            assert prototypes_labels.shape == (nb_prototypes,)
            assert prototypes_weights.shape == (nb_prototypes,)

            # compute local explanations
            outputs = method.explain(x_test)
            examples = outputs["examples"]
            distances = outputs["distances"]
            labels = outputs["labels"]
            indices = outputs["indices"]

            # check shapes
            assert examples.shape == (x_test.shape[0], k) + x_train.shape[1:]
            assert distances.shape == (x_test.shape[0], k)
            assert labels.shape == (x_test.shape[0], k)
            assert indices.shape == (x_test.shape[0], k, 2)

            # for each sample
            for i in range(x_test.shape[0]):
                # check first closest prototype label is the same as the sample label
                assert tf.reduce_all(tf.equal(labels[i, 0], y_test[i]))

                for j in range(k):
                    # check indices in prototypes' indices
                    index = indices[i, j, 0] * batch_size + indices[i, j, 1]
                    assert index in prototypes_indices

                    # check examples are in prototypes
                    assert tf.reduce_all(tf.equal(prototypes[prototypes_indices == index], examples[i, j]))

                    # check indices are in the dataset
                    assert tf.reduce_all(tf.equal(x_train[index], examples[i, j]))

                    # check labels
                    assert tf.reduce_all(tf.equal(labels[i, j], y_train[index]))

                    # check distances
                    assert almost_equal(
                        distances[i, j], 
                        tf.norm(tf.reduce_mean(x_train[index]) - tf.reduce_mean(x_test[i])),
                        epsilon=1e-5
                    )
