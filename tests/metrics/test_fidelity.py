import tensorflow as tf
import numpy as np

from ..utils import generate_model, generate_data, almost_equal
from xplique.metrics import Insertion, Deletion, MuFidelity

def test_mu_fidelity():
    # ensure we can compute the metric with consistents arguments
    input_shape, nb_labels, nb_samples = ((32, 32, 3), 10, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_model(input_shape, nb_labels)
    explanations = np.random.uniform(0, 1, x.shape[:-1])

    nb_estimation = 10 # number of samples to test correlation for each samples

    for grid_size in [None, 5]:
        for subset_percent in [0.1, 0.9]:
            for baseline_mode in [0.0, lambda x : x-0.5]:
                score = MuFidelity(model, x, y, grid_size=grid_size,
                                   subset_percent=subset_percent,
                                   baseline_mode=baseline_mode,
                                   nb_samples=nb_estimation)(explanations)
                assert -1.0 < score < 1.0


def test_causal_metrics():
    # ensure we can compute insertion/deletion metric with consistent arguments
    input_shape, nb_labels, nb_samples = ((32, 32, 3), 10, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_model(input_shape, nb_labels)
    explanations = np.random.uniform(0, 1, x.shape[:-1])

    for step in [5, 10]:
        for baseline_mode in [0.0, lambda x: x-0.5]:
            score_insertion = Insertion(model, x, y,
                                        baseline_mode=baseline_mode,
                                        steps=step)(explanations)
            score_deletion = Deletion(model, x, y,
                                      baseline_mode=baseline_mode,
                                      steps=step)(explanations)

            for score in [score_insertion, score_deletion]:
                assert 0.0 < score < 1.0


def test_perfect_correlation():
    """Ensure we get perfect score if the correlation is perfect"""
    # we ensure perfect correlation if the model return the sum of the input,
    # and the input is the explanations: corr( sum(phi), sum(x) - sum(x-phi) )
    # to do so we define f(x) -> sum(x) and phi = x
    nb_classes = 2

    input_shape, nb_labels, nb_samples = ((32, 32, 1), nb_classes, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = lambda x: tf.repeat(tf.reduce_sum(x, axis=(1, 2, 3))[:, None], nb_classes, -1)
    explanations = x

    perfect_score = MuFidelity(model, x, y, grid_size=None,
                               subset_percent=0.1,
                               baseline_mode=0.0,
                               nb_samples=200)(explanations)
    assert almost_equal(perfect_score, 1.0)


def test_worst_correlation():
    """Ensure we get worst score if the correlation is inversed"""
    # we ensure worst correlation if the model return the -sum of the input,
    # and the input is the explanations: corr( sum(phi), sum(x) - sum(x-phi) )
    # to do so we define f(x) -> sum(x) and phi = x
    nb_classes = 2

    input_shape, nb_labels, nb_samples = ((32, 32, 1), nb_classes, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = lambda x: tf.repeat(tf.reduce_sum(x, axis=(1, 2, 3))[:, None], nb_classes, -1)
    explanations = -x

    perfect_score = MuFidelity(model, x, y, grid_size=None,
                               subset_percent=0.1,
                               baseline_mode=0.0,
                               nb_samples=200)(explanations)
    assert almost_equal(perfect_score, -1.0)


def test_perfect_deletion():
    """Ensure we get perfect score if the model is sensible to deletion"""
    # we ensure perfect deletion if the model return 0.0 as soon as there is
    # one element set to baseline
    dim = 16
    steps = dim**2

    input_shape, nb_labels, nb_samples = ((dim, dim, 1), 2, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)

    model = lambda x: 1.0 - tf.reduce_max(tf.cast(x == 0.0, tf.float32), (1, 2))
    explanations = x

    perfect_score = Deletion(model, x, y, steps=steps)(explanations)
    assert almost_equal(perfect_score, ((1.0 + 0.0) / 2.0) / steps)


def test_perfect_insertion():
    """Ensure we get perfect score if the model is sensible to insertion"""
    # we ensure perfect deletion if the model return 1.0 as soon as there is
    # one element to non-baseline
    dim = 16
    steps = dim**2

    input_shape, nb_labels, nb_samples = ((dim, dim, 1), 2, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)

    model = lambda x: tf.reduce_max(tf.cast(x != 0.0, tf.float32), (1, 2))
    explanations = x

    perfect_score = Insertion(model, x, y, steps=steps)(explanations)
    assert almost_equal(perfect_score, 1.0 - ((0.0 + 1.0) / 2.0) / steps)
