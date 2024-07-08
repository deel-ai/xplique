"""
Commons for projections
"""

import tensorflow as tf

from ...commons import find_layer
from ...types import Callable, Union, Optional, Tuple


def model_splitting(model: tf.keras.Model,
                    latent_layer: Union[str, int],
                    return_layer: bool = False,
                    ) -> Tuple[Callable, Callable, Optional[tf.keras.layers.Layer]]:
    """
    Split the model into two parts, before and after the `latent_layer`.
    The parts will respectively be called `features_extractor` and `predictor`.

    Parameters
    ----------
    model
        Model to be split.
    latent_layer
        Layer used to split the `model`.

        Layer to target for the outputs (e.g logits or after softmax).
        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        To separate after the last convolution, `"last_conv"` can be used.
        Otherwise, `-1` could be used for the last layer before softmax.
    return_layer
        If True, return the latent layer found.
    
    Returns
    -------
    features_extractor
        Model used to project the inputs.
    predictor
        Model used to compute the attributions.
    latent_layer
        Layer used to split the `model`.
    """
    if latent_layer == "last_conv":
        latent_layer = next(
            layer for layer in model.layers[::-1] if hasattr(layer, "filters")
        )
    else:
        latent_layer = find_layer(model, latent_layer)

    features_extractor = tf.keras.Model(
        model.input, latent_layer.output, name="features_extractor"
    )
    # predictor = tf.keras.Model(
    #     latent_layer.output, model.output, name="predictor"
    # )
    second_input = tf.keras.Input(shape=latent_layer.output_shape[1:])
    
    # Reconstruct the second part of the model
    x = second_input
    layer_found = False
    for layer in model.layers:
        if layer_found:
            x = layer(x)
        if layer == latent_layer:
            layer_found = True
    
    # Create the second part of the model (predictor)
    predictor = tf.keras.Model(
        inputs=second_input,
        outputs=x,
        name="predictor"
    )

    if return_layer:
        return features_extractor, predictor, latent_layer
    return features_extractor, predictor