"""
Tests for the contrastive methods.
"""

import numpy as np
import tensorflow as tf

from xplique.example_based import (
    KLEORGlobalSim,
    KLEORSimMiss,
    LabelAwareCounterFactuals,
    NaiveCounterFactuals,
)
from xplique.example_based.projections import LatentSpaceProjection, Projection

from ..utils import generate_data, generate_model


def test_naive_counter_factuals():
    """ """
    # setup the tests
    cases = tf.constant(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]], dtype=tf.float32
    )
    cases_targets = tf.constant([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=tf.float32)

    cases_dataset = tf.data.Dataset.from_tensor_slices(cases).batch(2)
    cases_targets_dataset = tf.data.Dataset.from_tensor_slices(cases_targets).batch(2)

    inputs = tf.constant([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]], dtype=tf.float32)
    targets = tf.constant([[0, 1], [1, 0], [1, 0]], dtype=tf.float32)

    projection = Projection(space_projection=lambda inputs: inputs)

    # build the NaiveCounterFactuals object
    counter_factuals = NaiveCounterFactuals(
        cases_dataset,
        cases_targets_dataset,
        k=2,
        projection=projection,
        case_returns=["examples", "indices", "distances", "include_inputs"],
        batch_size=2,
    )

    mask = counter_factuals.filter_fn(inputs, cases, targets, cases_targets)
    assert mask.shape == (inputs.shape[0], cases.shape[0])

    expected_mask = tf.constant(
        [
            [False, True, True, False, True],
            [True, False, False, True, False],
            [True, False, False, True, False],
        ],
        dtype=tf.bool,
    )
    assert tf.reduce_all(tf.equal(mask, expected_mask))

    return_dict = counter_factuals(inputs, targets)
    assert set(return_dict.keys()) == set(["examples", "indices", "distances"])

    examples = return_dict["examples"]
    distances = return_dict["distances"]
    indices = return_dict["indices"]

    assert examples.shape == (3, 3, 2)  # (n, k+1, W)
    assert distances.shape == (3, 2)  # (n, k)
    assert indices.shape == (3, 2, 2)  # (n, k, 2)

    expected_examples = tf.constant(
        [
            [[1.5, 2.5], [2.0, 3.0], [3.0, 4.0]],
            [[2.5, 3.5], [1.0, 2.0], [4.0, 5.0]],
            [[4.5, 5.5], [4.0, 5.0], [1.0, 2.0]],
        ],
        dtype=tf.float32,
    )
    assert tf.reduce_all(tf.equal(examples, expected_examples))

    expected_distances = tf.constant(
        [
            [np.sqrt(2 * 0.5**2), np.sqrt(2 * 1.5**2)],
            [np.sqrt(2 * 1.5**2), np.sqrt(2 * 1.5**2)],
            [np.sqrt(2 * 0.5**2), np.sqrt(2 * 3.5**2)],
        ],
        dtype=tf.float32,
    )
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)

    expected_indices = tf.constant(
        [[[0, 1], [1, 0]], [[0, 0], [1, 1]], [[1, 1], [0, 0]]], dtype=tf.int32
    )
    assert tf.reduce_all(tf.equal(indices, expected_indices))


def test_label_aware_cf():
    """
    Test suite for the LabelAwareCounterFactuals class
    """
    # Same tests as the previous one but with the LabelAwareCounterFactuals class
    # thus we only needs to use cf_targets = 1 - targets of the previous tests
    cases = tf.constant(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]], dtype=tf.float32
    )
    cases_targets = tf.constant([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=tf.float32)

    cases_dataset = tf.data.Dataset.from_tensor_slices(cases).batch(2)
    cases_targets_dataset = tf.data.Dataset.from_tensor_slices(cases_targets).batch(2)

    inputs = tf.constant([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]], dtype=tf.float32)
    # cf_targets = tf.constant([[0, 1], [1, 0], [1, 0]], dtype=tf.float32)
    cf_expected_classes = tf.constant([[1, 0], [0, 1], [0, 1]], dtype=tf.float32)

    projection = Projection(space_projection=lambda inputs: inputs)

    # build the LabelAwareCounterFactuals object
    counter_factuals = LabelAwareCounterFactuals(
        cases_dataset=cases_dataset,
        targets_dataset=cases_targets_dataset,
        k=1,
        projection=projection,
        case_returns=["examples", "indices", "distances", "include_inputs"],
        batch_size=2,
    )

    mask = counter_factuals.filter_fn(inputs, cases, cf_expected_classes, cases_targets)
    assert mask.shape == (inputs.shape[0], cases.shape[0])

    expected_mask = tf.constant(
        [
            [False, True, True, False, True],
            [True, False, False, True, False],
            [True, False, False, True, False],
        ],
        dtype=tf.bool,
    )
    assert tf.reduce_all(tf.equal(mask, expected_mask))

    return_dict = counter_factuals(inputs, targets=None, cf_expected_classes=cf_expected_classes)
    assert set(return_dict.keys()) == set(["examples", "indices", "distances"])

    examples = return_dict["examples"]
    distances = return_dict["distances"]
    indices = return_dict["indices"]

    assert examples.shape == (3, 2, 2)  # (n, k+1, W)
    assert distances.shape == (3, 1)  # (n, k)
    assert indices.shape == (3, 1, 2)  # (n, k, 2)

    expected_examples = tf.constant(
        [[[1.5, 2.5], [2.0, 3.0]], [[2.5, 3.5], [1.0, 2.0]], [[4.5, 5.5], [4.0, 5.0]]],
        dtype=tf.float32,
    )
    assert tf.reduce_all(tf.equal(examples, expected_examples))

    expected_distances = tf.constant(
        [[np.sqrt(2 * 0.5**2)], [np.sqrt(2 * 1.5**2)], [np.sqrt(2 * 0.5**2)]], dtype=tf.float32
    )
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)

    expected_indices = tf.constant([[[0, 1]], [[0, 0]], [[1, 1]]], dtype=tf.int32)
    assert tf.reduce_all(tf.equal(indices, expected_indices))

    # Now let's dive when multiple classes are available in 1D
    cases = tf.constant(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]], dtype=tf.float32
    )
    cases_targets = tf.constant(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=tf.float32,
    )

    cases_dataset = tf.data.Dataset.from_tensor_slices(cases).batch(2)
    cases_targets_dataset = tf.data.Dataset.from_tensor_slices(cases_targets).batch(2)

    counter_factuals = LabelAwareCounterFactuals(
        cases_dataset=cases_dataset,
        targets_dataset=cases_targets_dataset,
        k=1,
        projection=projection,
        case_returns=["examples", "indices", "distances", "include_inputs"],
        batch_size=2,
    )

    inputs = tf.constant([[1.5], [2.5], [4.5], [6.5], [8.5]], dtype=tf.float32)
    cf_expected_classes = tf.constant(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]], dtype=tf.float32
    )

    mask = counter_factuals.filter_fn(inputs, cases, cf_expected_classes, cases_targets)
    assert mask.shape == (inputs.shape[0], cases.shape[0])

    expected_mask = tf.constant(
        [
            [False, True, False, True, True, False, False, False, False, True],
            [True, False, False, False, False, False, True, False, True, False],
            [False, False, True, False, False, True, False, True, False, False],
            [False, False, True, False, False, True, False, True, False, False],
            [True, False, False, False, False, False, True, False, True, False],
        ],
        dtype=tf.bool,
    )
    assert tf.reduce_all(tf.equal(mask, expected_mask))

    return_dict = counter_factuals(inputs, cf_expected_classes=cf_expected_classes)
    assert set(return_dict.keys()) == set(["examples", "indices", "distances"])

    examples = return_dict["examples"]
    distances = return_dict["distances"]
    indices = return_dict["indices"]

    assert examples.shape == (5, 2, 1)  # (n, k+1, W)
    assert distances.shape == (5, 1)  # (n, k)
    assert indices.shape == (5, 1, 2)  # (n, k, 2)

    expected_examples = tf.constant(
        [[[1.5], [2.0]], [[2.5], [1.0]], [[4.5], [3.0]], [[6.5], [6.0]], [[8.5], [9.0]]],
        dtype=tf.float32,
    )
    assert tf.reduce_all(tf.equal(examples, expected_examples))

    expected_distances = tf.constant(
        [
            [np.sqrt(0.5**2)],
            [np.sqrt(1.5**2)],
            [np.sqrt(1.5**2)],
            [np.sqrt(0.5**2)],
            [np.sqrt(0.5**2)],
        ],
        dtype=tf.float32,
    )
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)

    expected_indices = tf.constant(
        [[[0, 1]], [[0, 0]], [[1, 0]], [[2, 1]], [[4, 0]]], dtype=tf.int32
    )
    assert tf.reduce_all(tf.equal(indices, expected_indices))


def test_kleor():
    """
    Test suite for the Kleor class
    """
    # setup the tests
    cases = tf.constant(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]], dtype=tf.float32
    )
    cases_targets = tf.constant([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=tf.float32)

    cases_dataset = tf.data.Dataset.from_tensor_slices(cases).batch(2)
    cases_targets_dataset = tf.data.Dataset.from_tensor_slices(cases_targets).batch(2)

    inputs = tf.constant([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]], dtype=tf.float32)
    targets = tf.constant([[0, 1], [1, 0], [1, 0]], dtype=tf.float32)

    projection = Projection(space_projection=lambda inputs: inputs)

    # start when strategy is sim_miss
    kleor_sim_miss = KLEORSimMiss(
        cases_dataset=cases_dataset,
        targets_dataset=cases_targets_dataset,
        k=1,
        projection=projection,
        case_returns=["examples", "indices", "distances", "include_inputs", "nuns"],
        batch_size=2,
    )

    return_dict = kleor_sim_miss(inputs, targets)
    assert set(return_dict.keys()) == set(["examples", "indices", "distances", "nuns"])

    examples = return_dict["examples"]
    distances = return_dict["distances"]
    indices = return_dict["indices"]
    nuns = return_dict["nuns"]

    expected_nuns = tf.constant([[[2.0, 3.0]], [[1.0, 2.0]], [[4.0, 5.0]]], dtype=tf.float32)
    assert tf.reduce_all(tf.equal(nuns, expected_nuns))

    assert examples.shape == (3, 2, 2)  # (n, k+1, W)
    assert distances.shape == (3, 1)  # (n, k)
    assert indices.shape == (3, 1, 2)  # (n, k, 2)

    expected_examples = tf.constant(
        [[[1.5, 2.5], [1.0, 2.0]], [[2.5, 3.5], [2.0, 3.0]], [[4.5, 5.5], [3.0, 4.0]]],
        dtype=tf.float32,
    )
    assert tf.reduce_all(tf.equal(examples, expected_examples))

    expected_distances = tf.constant(
        [[np.sqrt(2 * 0.5**2)], [np.sqrt(2 * 0.5**2)], [np.sqrt(2 * 1.5**2)]], dtype=tf.float32
    )
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)

    expected_indices = tf.constant([[[0, 0]], [[0, 1]], [[1, 0]]], dtype=tf.int32)
    assert tf.reduce_all(tf.equal(indices, expected_indices))

    # now strategy is global_sim
    kleor_global_sim = KLEORGlobalSim(
        cases_dataset,
        cases_targets_dataset,
        k=1,
        projection=projection,
        case_returns=["examples", "indices", "distances", "include_inputs", "nuns"],
        batch_size=2,
    )

    return_dict = kleor_global_sim(inputs, targets)
    assert set(return_dict.keys()) == set(["examples", "indices", "distances", "nuns"])

    nuns = return_dict["nuns"]
    assert tf.reduce_all(tf.equal(nuns, expected_nuns))

    examples = return_dict["examples"]
    distances = return_dict["distances"]
    indices = return_dict["indices"]

    assert examples.shape == (3, 2, 2)  # (n, k+1, W)
    assert distances.shape == (3, 1)  # (n, k)
    assert indices.shape == (3, 1, 2)  # (n, k, 2)

    expected_indices = tf.constant([[[-1, -1]], [[0, 1]], [[-1, -1]]], dtype=tf.int32)
    assert tf.reduce_all(tf.equal(indices, expected_indices))

    expected_distances = tf.constant([[np.inf], [np.sqrt(2 * 0.5**2)], [np.inf]], dtype=tf.float32)
    # create masks for inf values
    inf_mask_dist = tf.math.is_inf(distances)
    inf_mask_expected_distances = tf.math.is_inf(expected_distances)
    assert tf.reduce_all(tf.equal(inf_mask_dist, inf_mask_expected_distances))
    assert tf.reduce_all(
        tf.abs(
            tf.where(inf_mask_dist, 0.0, distances)
            - tf.where(inf_mask_expected_distances, 0.0, expected_distances)
        )
        < 1e-5
    )

    expected_examples = tf.constant(
        [[[1.5, 2.5], [np.inf, np.inf]], [[2.5, 3.5], [2.0, 3.0]], [[4.5, 5.5], [np.inf, np.inf]]],
        dtype=tf.float32,
    )
    # mask for inf values
    inf_mask_examples = tf.math.is_inf(examples)
    inf_mask_expected_examples = tf.math.is_inf(expected_examples)
    assert tf.reduce_all(tf.equal(inf_mask_examples, inf_mask_expected_examples))
    assert tf.reduce_all(
        tf.abs(
            tf.where(inf_mask_examples, 0.0, examples)
            - tf.where(inf_mask_expected_examples, 0.0, expected_examples)
        )
        < 1e-5
    )


def test_contrastive_with_projection():
    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10
    nb_samples = 50

    for input_shape in input_shapes:
        features, labels = generate_data(input_shape, nb_labels, nb_samples)
        model = generate_model(input_shape, nb_labels)

        projection = LatentSpaceProjection(model, latent_layer=-1)

        for contrastive_method_class in [
            NaiveCounterFactuals,
            LabelAwareCounterFactuals,
            KLEORGlobalSim,
            KLEORSimMiss,
        ]:
            contrastive_method = contrastive_method_class(
                features,
                labels,
                k=1,
                projection=projection,
                case_returns=["examples", "indices", "distances", "include_inputs"],
                batch_size=7,
            )

            if isinstance(contrastive_method, LabelAwareCounterFactuals):
                cf_expected_classes = tf.one_hot(
                    tf.argmax(labels, axis=-1) + 1 % nb_labels, nb_labels
                )
                contrastive_method(
                    features, targets=labels, cf_expected_classes=cf_expected_classes
                )
            else:
                contrastive_method(features, targets=labels)
