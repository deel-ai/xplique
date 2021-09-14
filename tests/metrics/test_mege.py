import tensorflow as tf
import numpy as np

from ..utils import generate_model, generate_data, almost_equal
from xplique.metrics import MeGe
from xplique.attributions import Saliency, GradientInput


def test_best_mege():
    # ensure we get perfect score when the models are the same
    input_shape, nb_labels, nb_samples = ((8, 8, 1), 10, 80)
    x, y = generate_data(input_shape, nb_labels, nb_samples)

    model = generate_model(input_shape, nb_labels)
    learning_algorithm = lambda x_train, y_train, x_test, y_test: model

    for method in [Saliency, GradientInput]:

        metric = MeGe(learning_algorithm, x, y, k_splits=4)
        mege, _ = metric.evaluate(method)

        assert almost_equal(mege, 1.0)
