import numpy as np
import tensorflow as tf

from ..utils import almost_equal
from ..utils import generate_data
from ..utils import generate_model
from xplique.attributions import GradientInput
from xplique.attributions import Saliency
from xplique.metrics import MeGe


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
