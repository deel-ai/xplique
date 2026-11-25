"""Shared scientific-report assertions."""

import numpy as np


def assert_report_equal(actual, expected, **kwargs):
    """Compare reports without hiding NaN degeneracies."""
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        if isinstance(value, (str, type(None))):
            assert value == expected[key]
        else:
            np.testing.assert_allclose(value, expected[key], equal_nan=True, **kwargs)
