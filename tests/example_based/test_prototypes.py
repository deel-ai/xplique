"""
Test Prototypes
"""
import os
import sys

sys.path.append(os.getcwd())

import tensorflow as tf

from xplique.example_based import Prototypes, ProtoGreedy, ProtoDash, MMDCritic
from xplique.example_based.projections import Projection, LatentSpaceProjection

from tests.utils import almost_equal, get_gaussian_data, generate_model


def test_prototypes_global_explanations_basic():
    """
    Test prototypes shapes and uniqueness.
    """
    # Setup
    k = 2
    nb_prototypes = 5
    nb_classes = 2
    gamma = 0.026
    batch_size = 8  # TODO: test avec batch_size plus petite que nb_prototypes

    x_train, y_train = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=20, n_dims=3)
    x_test, y_test = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=6, n_dims=3)

    for method_class in [ProtoGreedy, ProtoDash, MMDCritic]:
        # compute general prototypes
        method = method_class(
            cases_dataset=x_train,
            labels_dataset=y_train,
            k=k,
            batch_size=batch_size,
            case_returns=["examples", "distances", "labels", "indices"],
            distance="euclidean",
            nb_prototypes=nb_prototypes,
            gamma=gamma,
        )

        # ======================
        # Test global prototypes

        # extract prototypes
        prototypes = method.prototypes
        prototypes_indices = method.prototypes_indices
        prototypes_labels = method.prototypes_labels
        prototypes_weights = method.prototypes_weights

        # check shapes
        assert prototypes.shape == (nb_prototypes,) + x_train.shape[1:]
        assert prototypes_indices.shape == (nb_prototypes, 2)
        assert prototypes_labels.shape == (nb_prototypes,)
        assert prototypes_weights.shape == (nb_prototypes,)

        # check uniqueness
        flatten_indices = prototypes_indices[:, 0] * batch_size + prototypes_indices[:, 1]
        assert len(tf.unique(flatten_indices)[0]) == nb_prototypes

        # for each prototype
        for i in range(nb_prototypes):
            # check prototypes are in the dataset and correspond to the index
            assert tf.reduce_all(tf.equal(prototypes[i], x_train[flatten_indices[i]]))

            # same for labels
            assert tf.reduce_all(tf.equal(prototypes_labels[i], y_train[flatten_indices[i]]))

            # check indices are in the dataset
            assert flatten_indices[i] >= 0 and flatten_indices[i] < x_train.shape[0]
        
        # =====================
        # Test local prototypes
            
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

        assert tf.reduce_all(indices[:, :, 0] >= 0)
        assert tf.reduce_all(indices[:, :, 0] < (1 + x_train.shape[0] // batch_size))
        assert tf.reduce_all(indices[:, :, 1] >= 0)
        assert tf.reduce_all(indices[:, :, 1] < batch_size)
        flatten_indices = indices[:, :, 0] * batch_size + indices[:, :, 1]

        # for each sample
        for i in range(x_test.shape[0]):
            # check first closest prototype label is the same as the sample label
            assert tf.reduce_all(tf.equal(labels[i], y_test[i]))

            for j in range(k):
                # check prototypes are in the dataset and correspond to the index
                assert tf.reduce_all(tf.equal(examples[i, j], x_train[flatten_indices[i, j]]))

                # same for labels
                assert tf.reduce_all(tf.equal(labels[i, j], y_train[flatten_indices[i, j]]))


def test_prototypes_global_sanity_check():
    """
    Test prototypes global explanations sanity checks.
    
    Check: For n separated gaussians,
           for n requested prototypes,
           there should be 1 prototype per gaussian.
    """
    # TODO: the two first prototypes seem to always come from the same class, I should investigate
    # Setup
    k = 2
    nb_prototypes = 3
    gamma = 0.026

    x_train, y_train = get_gaussian_data(nb_classes=nb_prototypes, nb_samples_class=5, n_dims=3)

    print("DEBUG: test_prototypes_global_sanity_check: x_train", x_train)

    for method_class in [MMDCritic, ProtoDash, ProtoGreedy]:
        print("DEBUG: test_prototypes_global_sanity_check: method_class", method_class)
        # compute general prototypes
        method = method_class(
            cases_dataset=x_train,
            labels_dataset=y_train,
            k=k,
            batch_size=8,
            nb_prototypes=nb_prototypes,
            gamma=gamma,
        )
        # extract prototypes
        prototypes_labels = method.get_global_prototypes()["prototypes_labels"]
        print("DEBUG: test_prototypes_global_sanity_check: y_train", y_train)
        print("DEBUG: test_prototypes_global_sanity_check: prototypes_labels", prototypes_labels)

        # check 1
        assert len(tf.unique(prototypes_labels)[0]) == nb_prototypes


def test_prototypes_with_projection():
    """
    Test prototypes shapes and uniqueness.
    """
    # Setup
    k = 2
    nb_prototypes = 10
    nb_classes = 2
    gamma = 0.026
    batch_size = 8  # TODO: test avec batch_size plus petite que nb_prototypes

    x_train, y_train = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=20, n_dims=3)
    x_test, y_test = get_gaussian_data(nb_classes=nb_classes, nb_samples_class=6, n_dims=3)

    # [10, 10, 10] -> [15, 15]
    # [20, 20, 20] -> [30, 30]
    # [30, 30, 30] -> [45, 45]
    weights = tf.constant([[1.0, 0.0],
                           [0.5, 0.5],
                           [0.0, 1.0],],
                          dtype=tf.float32)

    weighted_projection = Projection(
        space_projection=lambda inputs, targets=None: inputs @ weights
    )

    for method_class in [ProtoGreedy, ProtoDash, MMDCritic]:
        # compute general prototypes
        method = method_class(
            cases_dataset=x_train,
            labels_dataset=y_train,
            k=k,
            projection=weighted_projection,
            batch_size=batch_size,
            case_returns=["examples", "distances", "labels", "indices"],
            nb_prototypes=nb_prototypes,
            gamma=gamma,
        )

        # ======================
        # Test global prototypes

        # extract prototypes
        prototypes = method.prototypes
        prototypes_indices = method.prototypes_indices
        prototypes_labels = method.prototypes_labels
        prototypes_weights = method.prototypes_weights

        # check shapes
        assert prototypes.shape == (nb_prototypes,) + x_train.shape[1:]
        assert prototypes_indices.shape == (nb_prototypes, 2)
        assert prototypes_labels.shape == (nb_prototypes,)
        assert prototypes_weights.shape == (nb_prototypes,)

        # check uniqueness
        flatten_indices = prototypes_indices[:, 0] * batch_size + prototypes_indices[:, 1]
        assert len(tf.unique(flatten_indices)[0]) == nb_prototypes

        # for each prototype
        for i in range(nb_prototypes):
            # check prototypes are in the dataset and correspond to the index
            assert tf.reduce_all(tf.equal(prototypes[i], x_train[flatten_indices[i]]))

            # same for labels
            assert tf.reduce_all(tf.equal(prototypes_labels[i], y_train[flatten_indices[i]]))

            # check indices are in the dataset
            assert flatten_indices[i] >= 0 and flatten_indices[i] < x_train.shape[0]
        
        # =====================
        # Test local prototypes
            
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

        assert tf.reduce_all(indices[:, :, 0] >= 0)
        assert tf.reduce_all(indices[:, :, 0] < (1 + x_train.shape[0] // batch_size))
        assert tf.reduce_all(indices[:, :, 1] >= 0)
        assert tf.reduce_all(indices[:, :, 1] < batch_size)
        flatten_indices = indices[:, :, 0] * batch_size + indices[:, :, 1]

        # for each sample
        for i in range(x_test.shape[0]):
            # check first closest prototype label is the same as the sample label
            assert tf.reduce_all(tf.equal(labels[i], y_test[i]))

            for j in range(k):
                # check prototypes are in the dataset and correspond to the index
                assert tf.reduce_all(tf.equal(examples[i, j], x_train[flatten_indices[i, j]]))

                # same for labels
                assert tf.reduce_all(tf.equal(labels[i, j], y_train[flatten_indices[i, j]]))
