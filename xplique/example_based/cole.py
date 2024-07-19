"""
Implementation of Cole method a simlilar examples method from example based module
"""
import numpy as np
import tensorflow as tf

from ..attributions.base import BlackBoxExplainer
from ..types import Callable, List, Optional, Union, Type

from .similar_examples import SimilarExamples
from .projections import AttributionProjection, HadamardProjection


class Cole(SimilarExamples):
    """
    Cole is a similar examples method that gives the most similar examples to a query in some specific projection space.
    Cole use the model (to be explained) to build a search space so that distances are meaningful for the model.
    It uses attribution methods to weight inputs.
    Those attributions may be computed in the latent space for high-dimensional data like images.

    It is an implementation of a method proposed by Kenny et Keane in 2019,
    Twin-Systems to Explain Artificial Neural Networks using Case-Based Reasoning:
    https://researchrepository.ucd.ie/handle/10197/11064

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from this dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    labels_dataset
        Labels associated to the examples in the dataset. Indices should match with cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    targets_dataset
        Targets associated to the cases_dataset for dataset projection, oftentimes the one-hot encoding of a model's
        predictions. See `projection` for detail.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve per input.
    distance
        Distance function for examples search. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable,
        by default "euclidean".
    case_returns
        String or list of string with the elements to return in `self.explain()`.
        See the base class returns property for details.
    batch_size
        Number of sample treated simultaneously for projection and search.
        Ignored if `tf.data.Dataset` are provided (those are supposed to be batched).
    latent_layer
        Layer used to split the model, the first part will be used for projection and
        the second to compute the attributions. By default, the model is not split.
        For such split, the `model` should be a `tf.keras.Model`.

        Layer to target for the outputs (e.g logits or after softmax).
        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        The method as described in the paper apply the separation on the last convolutional layer.
        To do so, the `"last_conv"` parameter will extract it.
        Otherwise, `-1` could be used for the last layer before softmax.
    attribution_method
        Class of the attribution method to use for projection.
        It should inherit from `xplique.attributions.base.BlackBoxExplainer`.
        By default, it computes the gradient to make the Hadamard product in the latent space.
    attribution_kwargs
        Parameters to be passed for the construction of the `attribution_method`.
    """
    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        model: tf.keras.Model,
        targets_dataset: Union[tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.Tensor, np.ndarray]] = None,
        k: int = 1,
        distance: Union[str, Callable] = "euclidean",
        case_returns: Optional[Union[List[str], str]] = "examples",
        batch_size: Optional[int] = 32,
        latent_layer: Optional[Union[str, int]] = None,
        attribution_method: Union[str, Type[BlackBoxExplainer]] = "gradient",
        **attribution_kwargs,
    ):
        assert targets_dataset is not None

        # build the corresponding projection
        if isinstance(attribution_method, str) and attribution_method.lower() == "gradient":

            operator = attribution_kwargs.get("operator", None)
            
            projection = HadamardProjection(
                model=model,
                latent_layer=latent_layer,
                operator=operator,
            )
        elif issubclass(attribution_method, BlackBoxExplainer):
            # build attribution projection
            projection = AttributionProjection(
                model=model,
                method=attribution_method,
                latent_layer=latent_layer,
                **attribution_kwargs,
            )
        else:
            raise ValueError(
                f"attribution_method should be 'gradient' or a subclass of BlackBoxExplainer," +\
                    "not {attribution_method}"
            )

        super().__init__(
            cases_dataset=cases_dataset,
            targets_dataset=targets_dataset,
            labels_dataset=labels_dataset,
            projection=projection,
            k=k,
            case_returns=case_returns,
            batch_size=batch_size,
            distance=distance,
        )
