"""
Ensure we can use the operator functionnality on various models
"""

import pytest

import xplique
from xplique.commons.operators_operations import check_operator, Tasks, get_operator
from xplique.commons.operators import (predictions_operator, regression_operator,
                                       semantic_segmentation_operator, object_detection_operator)
from xplique.commons.exceptions import InvalidOperatorException


def test_check_operator():
    # ensure that the check operator detects non-operator

    # operator must have at least 3 arguments
    function_with_2_arguments = lambda x,y: 0

    # operator must be Callable
    not_a_function = [1, 2, 3]

    for operator in [function_with_2_arguments, not_a_function]:
        try:
            check_operator(operator)
            assert False
        except InvalidOperatorException:
            pass


def test_get_operator():
    possible_tasks = ["classification", "regression", "semantic segmentation", "object detection",
                      "object detection box position", "object detection box proba",
                      "object detection box class"]

    tasks_name = [task.name for task in Tasks]
    assert tasks_name.sort() == possible_tasks.sort()

    # get by enum
    assert get_operator(Tasks.CLASSIFICATION) is predictions_operator
    assert get_operator(Tasks.REGRESSION) is predictions_operator  # TODO, change when there is a real regression operator
    assert get_operator(Tasks.OBJECT_DETECTION) is object_detection_operator
    assert get_operator(Tasks.SEMANTIC_SEGMENTATION) is semantic_segmentation_operator

    # get by string
    assert get_operator("classification") is predictions_operator
    assert get_operator("regression") is predictions_operator  # TODO, change when there is a real regression operator
    assert get_operator("object detection") is object_detection_operator
    assert get_operator("semantic segmentation") is semantic_segmentation_operator

    # assert a not valid string does not work
    with pytest.raises(AssertionError):
        get_operator("random")

    # operator must have at least 3 arguments
    function_with_2_arguments = lambda x,y: 0

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
