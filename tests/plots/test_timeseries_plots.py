import matplotlib
import numpy as np
import os

from xplique.plots.timeseries import plot_timeseries_attributions


def test_one_explanation():
    # test timeseries plot attribution method with one explanation
    nb_features = 7
    nb_time_steps = 26
    title = "test one explanation"
    filepath = "tests/plots/timeseries_one_attribution_test.png"
    cmap = "coolwarm"

    features = ["feature_" + str(i) for i in range(nb_features)]
    explanations = np.random.uniform(size=(1, nb_time_steps, nb_features))

    plot_timeseries_attributions(
        explanations,
        features,
        title,
        filepath=filepath,
        cmap=cmap,
        colorbar=True,
    )

    assert os.path.exists(filepath)
    os.remove(filepath)


def test_several_explanations():
    # test timeseries plot attribution method with several explanations
    nb_features = 7
    nb_time_steps = 26
    title = "test several explanations"
    filepath = "tests/plots/timeseries_several_attributions_test.png"
    cmap = "coolwarm"

    features = ["feature_" + str(i) for i in range(nb_features)]
    explanations = np.random.uniform(size=(10, nb_time_steps, nb_features))

    plot_timeseries_attributions(
        explanations,
        features,
        title,
        filepath=filepath,
        cmap=cmap,
        colorbar=True,
    )

    assert os.path.exists(filepath)
    os.remove(filepath)
