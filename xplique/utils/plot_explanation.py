"""
Pretty plots option for explanations
"""
import numpy as np
from matplotlib import pyplot as plt

def plot_explanation(
    original_inp,
    explanation,
    mode="abs",
    percentile=0.05,
    alpha=0.7,
    cmap="cividis"):
    """
    A plot function which given the original input and the corresponding
    explanation highlights the original inputs more valuable features.

    Parameters
    ----------
    original_inp: ndarray (W, H, C)
        The array from which we ask our model an explanation

    explanation: ndarray (W, H, C)
        The explanation returned by an attribution method

    mode: optional, str
        How you decide to interpret your explanation
        - 'abs': consider the sensitivity, i.e the magnitude of features contributions
        to the decision
        - 'pos': consider only the features that positively contributes to the decision
        - 'neg': consider only the features that negatively contributes to the decision
        - 'none': consider an heatmap of contribution

    percentile: optional, float
        The percentile value you wish to clip your explanation

    alpha:
        The alpha parameter for the matplolib.pyplot.imshow method

    cmap: optional, str
        How to highlight the explanation. You can choose from:
        ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

    Returns
    -------
    A plot
    """

    if mode=="abs":
        exp = np.abs(explanation)
    elif mode=="pos":
        exp = np.where(explanation>0, explanation, 0)
    elif mode=="neg":
        exp = -np.where(explanation<0, explanation, 0)
    else:
        exp = explanation

    exp = np.clip(exp, np.percentile(exp, percentile), np.percentile(exp, 100.0-percentile))
    if len(exp.shape) > 2:
        exp = np.max(exp, axis=-1)

    exp -= exp.min()
    exp /= exp.max()

    plt.imshow(original_inp)
    plt.imshow(exp, cmap=cmap, alpha=alpha)
    plt.axis('off')
