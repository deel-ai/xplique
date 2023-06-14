"""
Module related to Case Base Explainer
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ..attributions.base import BlackBoxExplainer
from ..attributions import Saliency
from ..types import Callable, List, Optional, Union, Type

from .similar_examples import SimilarExamples
from .projections import AttributionProjection
from .search_methods import SklearnKNN
from .search_methods.base import BaseSearchMethod


class Cole(SimilarExamples):
    """
    Cole is a similar examples methods that gives the most similar examples to a query.
    Cole use the model to build a search space so that distances are meaningful for the model.
    It uses attribution methods to weights inputs.
    Those attributions may be computed in the latent space for complex data types like images.
    
    It is an implementation of a method proposed by Kenny et Keane in 2019.
    [https://researchrepository.ucd.ie/handle/10197/11064](Twin-Systems to Explain Artificial Neural Networks using Case-Based Reasoning)
    
    Parameters
    ----------
    case_dataset
        The dataset used to train the model, examples are extracted from the dataset.
    model
        The model from which we want to obtain explanations.
    dataset_targets
        Targets associated to the case_dataset for dataset projection.
        It is used by the `attribution_method` via `AttributionProjection`
        to compute attributions on the projected dataset.
        For more details on targets, refer to the documentation of attribution methods.
    dataset_labels
        Labels associated to the examples in the dataset. Indices should match with case_dataset.
    search_method
        An algorithm to search the examples in the projected space.
    k
        The number of examples to retrieve.
    distance
        Either a string or a Callable that computes distances between elements in the search space.
        It is used by the `seach_method` and one should refer to 
        [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html](`sklearn.neighbors.NearestNeighbors`)
        documentation for more details.
    case_returns
        String or list of string with the elements to return in `self.explain()`.
        See `self.set_returns()` for detail.
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
    def __init__(self,
                 case_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 model: tf.keras.Model,
                 dataset_targets: Union[tf.Tensor, np.ndarray],
                 labels_dataset: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 search_method: Type[BaseSearchMethod] = SklearnKNN,
                 k: int = 1,
                 distance: Union[str, Callable] = "euclidean",
                 case_returns: Optional[Union[List[str], str]] = "examples",
                 latent_layer: Optional[Union[str, int]] = None,
                 attribution_method: Type[BlackBoxExplainer] = Saliency,
                 **attribution_kwargs,
                 ):
        # buil attribution projection
        projection = AttributionProjection(model=model, method=attribution_method,
                                           latent_layer=latent_layer, **attribution_kwargs)

        assert dataset_targets is not None

        super().__init__(case_dataset, labels_dataset, dataset_targets,
                         search_method, k, projection, case_returns,
                         distance=distance, algorithm='auto')