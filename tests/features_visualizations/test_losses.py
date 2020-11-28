import numpy as np

from xplique.features_visualizations.losses import cosine_similarity
from ..utils import almost_equal


def test_cosine_similarity():
    vec = np.array([10.0, 20.0, 30.0])[np.newaxis, :]
    vec_colinear = np.array([1.0, 2.0, 3.0])[np.newaxis, :]
    vec_orthogonal = np.array([.0, .0, .0])[np.newaxis, :]
    vec_opposite = np.array([-0.01, -0.02, -.03])[np.newaxis, :]

    # cosine_similarity(a, b) = <a,b> / (|a| + |b|)
    assert almost_equal(cosine_similarity(vec, vec_colinear)[0],   1.0)
    assert almost_equal(cosine_similarity(vec, vec_orthogonal)[0], .0)
    assert almost_equal(cosine_similarity(vec, vec_opposite)[0],  -1.0)
