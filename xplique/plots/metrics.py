"""
Plots for metric
"""

import matplotlib
import matplotlib.pyplot as plt

from ..types import Optional, Union, Dict, Any


def barplot(
        scores: Dict[str, Dict[str, float]],
        sort_metric: str = None,
        ascending: bool = False,
        title: str = "",
        filepath: str = None,
        methods_colors: Optional[Union[Dict[str, Any], str]] = None
):
    """
    Make a bar chart gathering the score of each method for the different metrics used.

    Parameters
    ----------
    scores
        Dictionary of dictionary, with metrics scores,
        1st dict: the keys are the metric names and the values are dictionaries,
        2nd dict: the keys are the method names
        and the values the corresponding method-metric score.
    sort_metric
        Name of the metric used to sort the methods (a key of the directory).
    ascending
        Specify the sorting direction (need a sort_metric specified).
    title
        Title of the graphic.
    filepath
        Path the file will be saved at. If None, the function will call plt.show().
    methods_colors
        Either a dictionary of colors (key are method name, value are matplotlib
        supported color) or string of a matplotlib cmap (e.g 'Set3', 'jet'...).
    """
    # pylint: disable=invalid-name

    metrics = list(scores.keys())
    # sort values
    if sort_metric is not None:
        methods = [
            k for k, _ in
            sorted(scores[sort_metric].items(), key=lambda item: item[1], reverse=not ascending)
        ]
    else:
        # get methods from a metric (order do not import)
        methods = list(scores[list(scores.keys())[0]])

    # set x abscissa for each metric
    metrics_x = {metrics[i]: i for i in range(len(metrics))}

    # set x delta for each method
    width = (1 - 0.2) / len(methods)
    # -0.4 to get 0.2 between each metric, 0.5 width to center the bars
    methods_d = {methods[i]: -0.4 + (i + 0.5) * width for i in range(len(methods))}

    # set methods colors
    if not isinstance(methods_colors, dict):
        # either None or string
        if methods_colors is None:
            # default cmap
            cmap = matplotlib.cm.get_cmap("Set3")
        else:
            # methods_color is a string linking to a cmap
            cmap = matplotlib.cm.get_cmap(methods_colors)

        methods_colors = {methods[i]: cmap((i + 1) / len(methods))
                          for i in range(len(methods))}

    _, ax = plt.subplots(figsize=(4 + 2 * len(metrics), 4))
    handles = {}
    for metric in metrics:
        for method in methods:
            handles[method] = ax.bar(
                metrics_x[metric] + methods_d[method],
                scores[metric][method],
                color=methods_colors[method],
                align="center",
                width=width
            )

    ax.set_title(title)
    plt.legend(handles.values(), methods, title="attribution methods", bbox_to_anchor=(1.05, 1))
    ax.set_xlabel("metrics")
    ax.set_ylabel("score")
    plt.xticks(list(range(0, len(metrics))))
    ax.set_xticklabels(metrics)
    plt.xticks(rotation=0)
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()


def fidelity_curves(
        detailed_scores: Dict[str, Dict[int, float]],
        title: str = "",
        filepath: Optional[str] = None,
        methods_colors: Optional[Union[Dict[str, Any], str]] = None
):
    """
    Plot the evolution curves for Insertion and Deletion metrics
    based on their detailed_evaluate() method.
    Plot the evolution of a model score depending on the number features perturbed

    Parameters
    ----------
    detailed_scores
        Dictionary of methods detailed scores.
        The keys are the methods names.
        The values are the output of the detailed_evaluate() from Insertion and Deletion.
        (Please refer to this method for further details).
    title
        Title of the graphic.
    filepath
        Path the file will be saved at. If None, the function will call plt.show().
    methods_colors
        Either a dictionary of colors (key are method name, value are matplotlib
        supported color) or string of a matplotlib cmap (e.g 'Set3', 'jet'...).
    """
    # pylint: disable=invalid-name

    methods = list(detailed_scores.keys())

    # set methods colors
    if not isinstance(methods_colors, dict):
        # either None or string
        if methods_colors is None:
            # default cmap
            cmap = matplotlib.cm.get_cmap("Set3")
        else:
            # methods_color is a string linking to a cmap
            cmap = matplotlib.cm.get_cmap(methods_colors)

        methods_colors = {methods[i]: cmap((i + 1) / len(methods))
                          for i in range(len(methods))}

    _, ax = plt.subplots(figsize=(10, 5))
    for method, method_scores in detailed_scores.items():
        ax.plot(
            method_scores.keys(),
            method_scores.values(),
            label=method,
            color=methods_colors[method],
            linewidth=2.0,
        )
    ax.set_title(title)
    ax.legend(title="attribution methods", bbox_to_anchor=(1.05, 1))
    ax.set_xlabel("number of features perturbed")
    ax.set_ylabel("score")
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
