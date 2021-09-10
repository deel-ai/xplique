"""
Plots for tabular data
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def _sanitize_features_name(explanations, features_name):
    """
    Helper that provide generic features name (with the feature index) if features name
    is None.
    """
    if features_name is None:
        single_explanation = len(explanations.shape)==1
        if single_explanation:
            features_name = ["Feature %s"%(j) for j in range(len(explanations))]
        else:
            features_name = ["Feature %s"%(j) for j in range(explanations.shape[1])]
    return features_name

def _select_features(explanations, max_display, features_name):
    """
    Helper, that select max_display features. Useful if the number of features is huge.
    We keep the features which have a mean absolute impact greater than the others.
    """
    # if we have a lot of feature we will keep only max display
    if max_display is None:
        num_features_kept = len(features_name)
        features_idx_kept = np.arange(num_features_kept)
    else:
        num_features_kept = min(max_display, len(features_name))
        single_explanation = len(explanations.shape)==1
        if single_explanation:
            ranked = np.argsort(
               np.abs(explanations)
            )
        else:
            ranked = np.argsort(
                np.mean(np.abs(explanations), axis=0)
            )
        features_idx_kept = ranked[::-1][:num_features_kept]
    return num_features_kept, features_idx_kept

def _add_colorbar(cmap):
    """
    Add the color bar corresponding to cmap to the plot
    """
    mappa = cm.ScalarMappable(cmap=cmap)
    colorbar = plt.colorbar(mappa, ticks=[0, 1])
    colorbar.set_ticklabels(['Low', 'High'])
    colorbar.set_label('Feature value', size=12, labelpad=0)
    colorbar.ax.tick_params(labelsize=11, length=0)
    colorbar.set_alpha(1)
    colorbar.outline.set_visible(False)
    bbox = colorbar.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    colorbar.ax.set_aspect((bbox.height - 0.9) * 20)

def _clip_values(fvalues, clip_percentile):
    """
    Clip value of fvalues between clip_percentile and 100-clip_percentile
    """
    if clip_percentile is not None:
        clip_min = np.percentile(fvalues, clip_percentile)
        clip_max = np.percentile(fvalues, 100 - clip_percentile)
        fvalues = np.clip(fvalues, clip_min, clip_max)
    return fvalues

def _get_offset_positions(explanation_val_feature, nb_points, row_height):
    """
    Helper to set an offset position on the y-axis for points with close x-value.


    Paramerers
    ----------
    explanation_val_feature
        All the explanation value for a single feature
    nb_points
        The number of explanation value per feature
    row_height
        The length of the available y-axis in which points will be displayed

    Returns
    -------
    offset
        The y position offset for each points
    """
    nbins = 100 # nb of division on the y-axis for one feature

    # get the points where explanation values are close
    max_feat_val = np.max(explanation_val_feature)
    min_feat_val = np.min(explanation_val_feature)
    num = nbins * (explanation_val_feature - min_feat_val)
    den = max_feat_val - min_feat_val + 1e-8 # in case max=min
    # points with the same quant value might overlap
    quant = np.round(num/den)

    # here we make sure points with the same quant value have different offset
    inds = np.argsort(quant + np.random.randn(nb_points) * 1e-6)
    layer = 0
    last_bin = -1
    offset = np.zeros(nb_points)
    for ind in inds:
        # check if the two points are close and might overlap
        if quant[ind] != last_bin:
            layer = 0
        # centered around the line, one time over the other below
        offset[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[ind]
    # adjust the offsets to the row height
    offset *= 0.9 * (row_height / np.max(offset + 1))

    return offset

def plot_feature_impact(
    explanation,
    features_name=None,
    features_value=None,
    max_display=None,
):
    """
    Bar plot of the explanation values. Some options are directly inspired from the
    [shap library](https://github.com/slundberg/shap).

    Parameters
    ----------
    explanation
        A single explanation
    features_name
        The name of the features. If not provided will be displayed as: Feature: idx_feature
    features_value
        The value of features from which the explanation was computed. If provided, it will be
        displayed next to the corresponding label
    max_display
        If there is too many features, set this parameter to display only the max_display features
        with the most significant impact on the output
    """
    # sanitize to numpy array
    explanation = np.array(explanation)

    # add default features name if not provided
    features_name = _sanitize_features_name(explanation, features_name)

    # if we have a lot of feature we will keep only max display
    num_features_kept, features_idx_kept = _select_features(explanation, max_display, features_name)
    explanation_kept = explanation[features_idx_kept]

    # build y-ticks label
    yticklabels = []
    for idx_kept in features_idx_kept:
        if features_value is None:
            yticklabels.append(features_name[idx_kept])
        else:
            yticklabels.append("%0.03f"%(features_value[idx_kept]) +" = "+ features_name[idx_kept])

    y_pos = np.arange(num_features_kept)  # the label locations

    fig, axes = plt.subplots()

    # the actual bar plot
    colors = [
        'slateblue' if explanation_kept[j]<=0 else 'yellowgreen' for j in range(num_features_kept)
    ]
    axes.barh(
        y_pos,
        explanation_kept,
        align='center',
        color=colors
    )

    # add the explanation value next to the bar
    xlen = plt.xlim()[1] - plt.xlim()[0]
    bbox = axes.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    xscale = xlen/bbox.width

    for i in y_pos:
        # put horizontal lines for each feature row
        plt.axhline(i, color="darkgrey", lw=0.5, dashes=(1, 5), zorder=-1)
        if explanation_kept[i] < 0:
            axes.text(
                explanation_kept[i] - 0.02*xscale,
                y_pos[i],
                str("%0.02f"%explanation_kept[i]),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=10
            )
        else:
            axes.text(
                explanation_kept[i] + 0.02*xscale,
                y_pos[i],
                str("%0.02f"%explanation_kept[i]),
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=10
            )

    # add some text for labels and custom y-axis tick labels
    axes.set_xlabel('Impact on output')
    axes.set_title('Features impact')
    axes.set_yticks(y_pos)
    axes.set_yticklabels(yticklabels)
    axes.legend()

    # make the plot prettier
    fig.tight_layout()

    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    xmin,xmax = plt.gca().get_xlim()

    # if we have negative values draw a vertical axis at zero
    neg_val = len(explanation_kept[explanation_kept<0])!=0
    if neg_val:
        plt.axvline(0, 0, 1, color="dimgray", linestyle="-", linewidth=1)
        axes.spines['left'].set_visible(False)
        plt.gca().set_xlim(xmin - (xmax-xmin)*0.1, xmax + (xmax-xmin)*0.1)
    else:
        plt.gca().set_xlim(xmin, xmax + (xmax-xmin)*0.1)

def plot_mean_feature_impact(
    explanations,
    features_name=None,
    max_display=None,
):
    """
    The same than plot_feature_importance but we will consider the mean explanation value
    grouped by feature.
    A more informative plot is the summary_plot_tabular.
    """
    # sanitize explanations to numpy array
    explanations = np.array(explanations)

    mean_explanation_per_feature = np.mean(explanations, axis=0)

    plot_feature_impact(
        mean_explanation_per_feature,
        features_name=features_name,
        max_display=max_display
    )

def summary_plot_tabular(
    explanations,
    features_values = None,
    features_name = None,
    max_display = None,
    cmap = 'viridis',
    clip_percentile = 1,
    alpha = 1,
    plot_size = None,
):
    """
    Summary plot adapted from the [shap library](https://github.com/slundberg/shap). Usefull to
    have an overall idea of the impact of each features on the output depending on their value.

    Parameters
    ----------
    explanations
        The explanations from which we want to draw insights
    features_values
        The features values of each samples leading to the different explanations.
        Shape must match explanations shape.
    features_name
        The name of the features. If not provided will be displayed as: Feature: idx_feature
    max_display
        If there is too many features, set this parameter to display only the max_display features
        with the most significant impact on the output
    cmap
        Matplotlib color map to apply.
    clip_percentile
        Percentile value to use if clipping is needed, e.g a value of 1 will perform a clipping
        between percentile 1 and 99. This parameter allows to avoid outliers  in case of too
        extreme values.
    alpha
        Opacity value for the feature value.
    plot_size
        If provided as a tuple or a list will set the figure size in inches.
        If provided as an int, will define the row_height for each features.
    """
    # sanitize to numpy array
    explanations = np.array(explanations)

    # add default features name if not provided
    features_name = _sanitize_features_name(explanations, features_name)

    # if we have a lot of feature we will keep only max display
    nb_features_kept, features_idx_kept = _select_features(explanations, max_display, features_name)

    # build our y-tick labels
    yticklabels = [features_name[idx_kept] for idx_kept in features_idx_kept]

    # build the figure
    row_height = 0.4
    if plot_size is None:
        plt.gcf().set_size_inches(8, nb_features_kept * row_height + 1.5)
    elif isinstance(plot_size,(list, tuple)):
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif isinstance(plot_size, float):
        row_height = plot_size
        plt.gcf().set_size_inches(8, nb_features_kept * row_height + 1.5)
    else:
        raise ValueError("Wrong plot_size argument, type can be [None, float, list, tuple]")

    plt.axvline(0, 0, 1, color="dimgray", linestyle="-", linewidth=1)

    # make the beeswarm dots
    for pos, i in enumerate(features_idx_kept):

        plt.axhline(y=pos, color="darkgrey", lw=0.5, dashes=(1, 5), zorder=-1)
        explanation_val_feature = explanations[:, i]

        if features_values is None:
            fvalues = None
        else:
            fvalues = features_values[:, i]

        # we need to set an offstet y position for points having close
        # explanation value in order to avoid points overlapping
        nb_points = len(explanation_val_feature)
        offset = _get_offset_positions(explanation_val_feature, nb_points, row_height)

        # if features values are not provided then make the points all grey
        if fvalues is None:
            plt.scatter(
                explanation_val_feature,
                pos + offset,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                color="slategrey",
                s=16,
                rasterized=True
            )

        else:
            # if needed, apply clip_percentile
            fvalues = _clip_values(fvalues, clip_percentile)

            plt.scatter(
                explanation_val_feature,
                pos + offset,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                cmap=cmap,
                s=16,
                c=fvalues,
                rasterized=True
            )

    # draw the color bar
    if features_values is not None:
        _add_colorbar(cmap)

    # make the plot prettier
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.yticks(range(nb_features_kept), yticklabels, fontsize=13)
    plt.gca().tick_params('y', length=20, width=0.5, which='major')
    plt.gca().tick_params('x', labelsize=11)
    plt.ylim(-1, nb_features_kept)
    plt.xlabel("Impact on output", fontsize=13)
    plt.tight_layout()
