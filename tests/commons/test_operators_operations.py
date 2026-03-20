"""
Ensure we can use the operator functionnality on various models
"""

import pytest
import tensorflow as tf

import xplique
from xplique.commons.exceptions import InvalidOperatorException
from xplique.commons.operators import (
    object_detection_operator,
    predictions_operator,
    semantic_segmentation_operator,
)
from xplique.commons.operators_operations import (
    Tasks,
    check_operator,
    end_watch_layer,
    get_operator,
    watch_layer,
)


def _generate_conv_model(input_shape=(8, 8, 1), output_shape=3):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(4, kernel_size=(3, 3), activation="relu", name="conv"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_shape, activation="softmax"),
        ]
    )


def test_check_operator():
    # ensure that the check operator detects non-operator

    # operator must have at least 3 arguments
    def function_with_2_arguments(x, y):
        return 0

    # operator must be Callable
    not_a_function = [1, 2, 3]

    for operator in [function_with_2_arguments, not_a_function]:
        try:
            check_operator(operator)
            assert False
        except InvalidOperatorException:
            pass


def test_get_operator():
    possible_tasks = [
        "classification",
        "regression",
        "semantic segmentation",
        "object detection",
        "object detection box position",
        "object detection box proba",
        "object detection box class",
    ]

    tasks_name = [task.name for task in Tasks]
    assert tasks_name.sort() == possible_tasks.sort()

    # get by enum
    assert get_operator(Tasks.CLASSIFICATION) is predictions_operator
    assert (
        get_operator(Tasks.REGRESSION) is predictions_operator
    )  # TODO, change when there is a real regression operator
    assert get_operator(Tasks.OBJECT_DETECTION) is object_detection_operator
    assert get_operator(Tasks.SEMANTIC_SEGMENTATION) is semantic_segmentation_operator

    # get by string
    assert get_operator("classification") is predictions_operator
    assert (
        get_operator("regression") is predictions_operator
    )  # TODO, change when there is a real regression operator
    assert get_operator("object detection") is object_detection_operator
    assert get_operator("semantic segmentation") is semantic_segmentation_operator

    # assert a not valid string does not work
    with pytest.raises(AssertionError):
        get_operator("random")

    # operator must have at least 3 arguments
    def function_with_2_arguments(x, y):
        return 0

    # operator must be Callable
    not_a_function = [1, 2, 3]

    for operator in [function_with_2_arguments, not_a_function]:
        try:
            get_operator(operator)
        except InvalidOperatorException:
            pass


def test_proposed_operators():
    # ensure all proposed operators are operators
    for operator in [task.value for task in Tasks]:
        check_operator(operator)


def test_enum_shortcut():
    # ensure all proposed operators are operators
    for operator in [task.value for task in xplique.Tasks]:
        check_operator(operator)


def test_watch_layer_exposes_feature_maps_and_restore_layer_call():
    tf.keras.backend.clear_session()

    model = _generate_conv_model()
    conv_layer = model.get_layer("conv")
    original_call = conv_layer.call
    inputs = tf.random.uniform((2, 8, 8, 1))
    targets = tf.one_hot([0, 1], depth=3)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        watched_layer = watch_layer(conv_layer, tape)
        predictions = model(inputs)
        feature_maps = watched_layer.result
        scores = tf.reduce_sum(predictions * targets, axis=-1)

    feature_maps_gradients = tape.gradient(scores, feature_maps)

    assert watched_layer is conv_layer
    assert hasattr(conv_layer, "result")
    tf.debugging.assert_equal(feature_maps, conv_layer.result)
    assert feature_maps_gradients is not None
    assert feature_maps_gradients.shape == feature_maps.shape
    assert hasattr(conv_layer.call, "__wrapped__")

    end_watch_layer(conv_layer)

    assert not hasattr(conv_layer, "result")
    assert not hasattr(conv_layer.call, "__wrapped__")
    assert getattr(conv_layer.call, "__func__", conv_layer.call) is getattr(
        original_call, "__func__", original_call
    )
    assert getattr(conv_layer.call, "__self__", None) is getattr(original_call, "__self__", None)

    _ = model(inputs)
    assert not hasattr(conv_layer, "result")


def test_end_watch_layer_allows_rewatching_same_layer():
    tf.keras.backend.clear_session()

    model = _generate_conv_model()
    conv_layer = model.get_layer("conv")
    inputs = tf.random.uniform((1, 8, 8, 1))
    targets = tf.one_hot([1], depth=3)

    for _ in range(2):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            watch_layer(conv_layer, tape)
            predictions = model(inputs)
            feature_maps = conv_layer.result
            scores = tf.reduce_sum(predictions * targets, axis=-1)

        feature_maps_gradients = tape.gradient(scores, feature_maps)

        assert feature_maps_gradients is not None

        end_watch_layer(conv_layer)
        assert not hasattr(conv_layer, "result")
