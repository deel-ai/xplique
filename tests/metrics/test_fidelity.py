import numpy as np
import tensorflow as tf

from ..utils import generate_model, generate_data
from xplique.metrics import insertion, deletion, mu_fidelity

def test_mu_fidelity():
    # ensure we can compute the metric with consistents arguments
    input_shape, nb_labels, nb_samples = ((32, 32, 3), 10, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_model(input_shape, nb_labels)
    phi = np.random.uniform(0, 1, x.shape[:-1])

    nb_estimation = 10 # number of samples to test correlation for each samples

    for grid_size in [None, 5]:
        for subset_percent in [0.1, 0.9]:
            for baseline_mode in [0.0, lambda x : x-0.5]:
                score, preds_atributions = mu_fidelity(model, x, y, phi, grid_size=grid_size,
                                              subset_percent=subset_percent,
                                              baseline_mode=baseline_mode,
                                              nb_samples=nb_estimation)
                assert -1.0 < score < 1.0
                assert preds_atributions.shape == (2, nb_samples, nb_estimation)


def test_causal_metrics():
    # ensure we can compute insertion/deletion metric with consistent arguments
    input_shape, nb_labels, nb_samples = ((32, 32, 3), 10, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_model(input_shape, nb_labels)
    phi = np.random.uniform(0, 1, x.shape[:-1])

    for step in [5, 10]:
        for baseline_mode in [0.0, lambda x: x-0.5]:
            score_insertion, curve_insertion = insertion(model, x, y, phi,
                                                         baseline_mode=baseline_mode,
                                                         steps=step)
            score_deletion, curve_deletion = deletion(model, x, y, phi,
                                                      baseline_mode=baseline_mode,
                                                      steps=step)
            for score in [score_insertion, score_deletion]:
                assert 0.0 < score < 1.0
            for curve in [curve_insertion, curve_deletion]:
                assert curve.shape == (2, step)
