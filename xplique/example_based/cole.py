"""
Implementation of Cole method a simlilar examples method from example based module
"""

import numpy as np
import tensorflow as tf

from ..attributions.base import BlackBoxExplainer
from ..attributions import Saliency
from ..types import Callable, List, Optional, Union, Type

from .similar_examples import SimilarExamples
from .projections import AttributionProjection
from .search_methods import KNN
from .search_methods import BaseSearchMethod


class Cole(SimilarExamples):
    """
    Cole is a similar examples methods that gives the most similar examples to a query.
    Cole use the model to build a search space so that distances are meaningful for the model.
    It uses attribution methods to weights inputs.
    Those attributions may be computed in the latent space for complex data types like images.

    It is an implementation of a method proposed by Kenny et Keane in 2019,
    Twin-Systems to Explain Artificial Neural Networks using Case-Based Reasoning:
    https://researchrepository.ucd.ie/handle/10197/11064

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Becareful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    labels_dataset
        Labels associated to the examples in the dataset. Indices should match with cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other dataset should match `cases_dataset`.
        Becareful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    targets_dataset
        Targets associated to the cases_dataset for dataset projection. See `projection` for detail.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other dataset should match `cases_dataset`.
        Becareful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve. Default value is `1`.
    distance
        Either a Callable, or a value supported by `tf.norm` `ord` parameter.
        Their documentation (https://www.tensorflow.org/api_docs/python/tf/norm) say:
        "Supported values are 'fro', 'euclidean', 1, 2, np.inf and any positive real number
        yielding the corresponding p-norm."
    case_returns
        String or list of string with the elements to return in `self.explain()`.
        See `self.set_returns()` from parent class `SimilarExamples` for detail.
        By default, the `explain()` method will only return the examples.
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

        The method as described in the paper apply the separation on the last convolutionnal layer.
        To do so, the `"last_conv"` parameter will extract it.
        Otherwise, `-1` could be used for the last layer before softmax.
    attribution_method
        Class of the attribution method to use for projection.
        It should inherit from `xplique.attributions.base.BlackBoxExplainer`.
        Ignored if a projection is given.
    attribution_kwargs
        Parameters to be passed at the construction of the `attribution_method`.
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
        attribution_method: Type[BlackBoxExplainer] = Saliency,
        **attribution_kwargs,
    ):
        # buil attribution projection
        projection = AttributionProjection(
            model=model,
            method=attribution_method,
            latent_layer=latent_layer,
            **attribution_kwargs,
        )

        assert targets_dataset is not None

        super().__init__(
            cases_dataset,
            labels_dataset,
            targets_dataset,
            k,
            projection,
            case_returns,
            batch_size,
            distance=distance,
        )
