"""
Commons for projections
"""
import warnings

import tensorflow as tf

from ...commons import find_layer
from ...types import Callable, Union, Optional, Tuple


def model_splitting(
        model: Union[tf.keras.Model, 'torch.nn.Module'],
        latent_layer: Union[str, int],
        device: Union["torch.device", str] = None,
    ) -> Tuple[Union[tf.keras.Model, 'torch.nn.Module'], Union[tf.keras.Model, 'torch.nn.Module']]:
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
    device
        Device to use for the projection, if None, use the default device.
        Only used for PyTorch models. Ignored for TensorFlow models.
    
    Returns
    -------
    features_extractor
        Model used to project the inputs.
    predictor
        Model used to compute the attributions.
    latent_layer
        Layer used to split the `model`.
    """
    if isinstance(model, tf.keras.Model):
        return _tf_model_splitting(model, latent_layer)
    try:
        return _torch_model_splitting(model, latent_layer, device)
    except ImportError as exc:
        raise AttributeError(
            "Unknown model type, should be either `tf.keras.Model` or `torch.nn.Module`. "\
            +f"But got {type(model)} instead.") from exc


def _tf_model_splitting(model: tf.keras.Model,
                        latent_layer: Union[str, int],
                        ) -> Tuple[tf.keras.Model, tf.keras.Model]:
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
    
    Returns
    -------
    features_extractor
        Model used to project the inputs.
    predictor
        Model used to compute the attributions.
    latent_layer
        Layer used to split the `model`.
    """

    warnings.warn(
        "Automatically splitting the provided TensorFlow model into two parts. "\
        +"This splitting is not robust to all models. "\
        +"It is recommended to split the model manually. "\
        +"Then the splitted parts can be provided through the `from_splitted_model` method.")

    if latent_layer == "last_conv":
        latent_layer = next(
            layer for layer in model.layers[::-1] if hasattr(layer, "filters")
        )
    else:
        latent_layer = find_layer(model, latent_layer)

    features_extractor = tf.keras.Model(
        model.input, latent_layer.output, name="features_extractor"
    )
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

    return features_extractor, predictor


def _torch_model_splitting(
        model: 'torch.nn.Module',
        latent_layer: Union[str, int],
        device: Union["torch.device", str] = None,
    ) -> Tuple['torch.nn.Module', 'torch.nn.Module']:
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
    Device to use for the projection, if None, use the default device.
    
    Returns
    -------
    features_extractor
        Model used to project the inputs.
    predictor
        Model used to compute the attributions.
    latent_layer
        Layer used to split the `model`.
    """
    # pylint: disable=import-outside-toplevel
    import torch
    from torch import nn
    from ...wrappers import TorchWrapper

    warnings.warn(
        "Automatically splitting the provided PyTorch model into two parts. "\
        +"This splitting is based on `model.named_children()`. "\
        +"If the model cannot be reconstructed via sub-modules, errors are to be expected. "\
        +"It is recommended to split the model manually and wrap it with `TorchWrapper`. "\
        +"Then the wrapped parts can be provided through the `from_splitted_model` method.")

    if device is None:
        warnings.warn(
            "No device provided for the projection, using 'cuda' if available, else 'cpu'."
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

    first_model = nn.Sequential()
    second_model = nn.Sequential()
    split_flag = False

    if isinstance(latent_layer, int) and latent_layer < 0:
        latent_layer = len(list(model.children())) + latent_layer

    for layer_index, (name, module) in enumerate(model.named_children()):
        if latent_layer in [layer_index, name]:
            split_flag = True

        if not split_flag:
            first_model.add_module(name, module)
        else:
            second_model.add_module(name, module)

    # Define forward function for the first model
    def first_model_forward(x):
        for module in first_model:
            x = module(x)
        return x

    # Define forward function for the second model
    def second_model_forward(x):
        for module in second_model:
            x = module(x)
        return x

    # Set the forward functions for the models
    first_model.forward = first_model_forward
    second_model.forward = second_model_forward

    # Wrap models to obtain tensorflow ones
    first_model.eval()
    wrapped_first_model = TorchWrapper(first_model, device=device)
    second_model.eval()
    wrapped_second_model = TorchWrapper(second_model, device=device)

    return wrapped_first_model, wrapped_second_model


@tf.function
def target_free_classification_operator(model: Callable,
                                        inputs: tf.Tensor,
                                        targets: Optional[tf.Tensor] = None) -> tf.Tensor:
    """
    Compute predictions scores, only for the label class, for a batch of samples.
    It has the same behavior as `Tasks.CLASSIFICATION` operator
    but computes targets at the same time if not provided.
    Targets are a mask with 1 on the predicted class and 0 elsewhere.
    This operator should only be used for classification tasks.


    Parameters
    ----------
    model
        Model used for computing predictions.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

    Returns
    -------
    scores
        Predictions scores computed, only for the label class.
    """
    predictions = model(inputs)

    # the condition is always the same, hence this should not affect the graph
    if targets is None:
        targets = tf.one_hot(tf.argmax(predictions, axis=-1), predictions.shape[-1])

    # this implementation did not pass the tests, the cond shapes were different if targets is None
    # targets = tf.cond(
    #     pred=tf.constant(targets is None, dtype=tf.bool),
    #     true_fn=lambda: tf.one_hot(tf.argmax(predictions, axis=-1), predictions.shape[-1]),
    #     false_fn=lambda: targets,
    # )

    scores = tf.reduce_sum(predictions * targets, axis=-1)
    return scores
