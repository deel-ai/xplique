import matplotlib
import numpy as np
import os

from xplique.plots.metrics import barplot, fidelity_curves


def test_bar_plot():
    # test metrics barplot
    cmap = matplotlib.cm.get_cmap("Set3")
    methods_colors = {"method_" + str(i): cmap(i / 9) for i in range(10)}

    scores = {}
    for metric in ["metric_1", "metric_2"]:
        scores[metric] = {
            "method_" + str(i): np.random.random_sample()
            for i in range(10)
        }

    filepath = "tests/plots/barplot_test.png"
    barplot(
        scores,
        sort_metric="metric_1",
        ascending=True,
        title="test bar-plot",
        filepath=filepath,
        methods_colors=methods_colors
    )

    assert os.path.exists(filepath)
    os.remove(filepath)


def test_curves():
    # test fidelity metric curves plot
    cmap = matplotlib.cm.get_cmap("Set3")
    methods_colors = {"method_" + str(i): cmap(i / 9) for i in range(10)}

    steps = np.linspace(0, 100, num=11)
    detailed_scores = {
        "method_" + str(i):
            {step: val for (step, val) in
             zip(steps, np.linspace(*np.random.randint(0, 100, 2), num=11))}
        for i in range(10)
    }

    filepath = "tests/plots/fidelity_curves_test.png"
    fidelity_curves(
        detailed_scores,
        title="test curves",
        filepath=filepath,
        methods_colors=methods_colors
    )

    assert os.path.exists(filepath)
    os.remove(filepath)
