"""Tests for tabular plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np

from xplique.plots import summary_plot_tabular


def test_summary_plot_tabular_adds_colorbar():
    """The summary plot attaches its colorbar to the plot axes."""
    explanations = np.array([[0.2, -0.1], [0.4, 0.3]])
    features_values = np.array([[0.1, 0.8], [0.9, 0.2]])

    summary_plot_tabular(explanations, features_values)

    figure = plt.gcf()
    assert len(figure.axes) == 2
    assert [tick.get_text() for tick in figure.axes[1].get_yticklabels()] == ["Low", "High"]
    plt.close(figure)
