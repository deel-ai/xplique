"""
Module related to Case Base Explainer
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ..attributions.base import BlackBoxExplainer, sanitize_input_output
from ..attributions import Saliency
from ..types import Callable, List, Optional, Union, Type

from .base import NaturalExampleBasedExplainer
from .projections import Projection, AttributionProjection
from .search_methods import BaseSearchMethod, SklearnKNN


class Cole(NaturalExampleBasedExplainer):
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
    dataset_labels
        Labels associated to the examples in the dataset. Indices should match with case_dataset.
    dataset_targets
        Targets associated to the case_dataset for dataset projection. See `projection` for detail.
    search_method
        An algorithm to search the examples in the projected space.
    k
        The number of examples to retrieve.
    distance
        Either a string or a Callable that computes distances between elements in the search space.
        It is used by the `seach_method` and one should refer to 
        [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html](`sklearn.neighbors.NearestNeighbors`)
        documentation for more details.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space sould be a space where distance make sense for the model.
        It can be set to `None` for the projection to be build with `model` and `attribution_method`.
        Otherwise, should not be `None`, otherwise, 
        all examples could be computed only with the `search_method`.
        
        Example of Callable:
        ```
        def projection_example(inputs: Union(tf.Tensor, np.ndarray), targets: Union(tf.Tensor, np.ndarray) = None):
            '''
            Example of projection,
            inputs are the elements to project.
            targets are optionnal parameters to orientated the projection.
            '''
            projected_inputs = ...  # do some magic on inputs, it should use the model.
            return projected_inputs
        ```
    returns
        String or list of string with the elements to return in `self.explain()`.
        See `self.set_returns()` for detail.
    model
        The model from which we want to obtain explanations.
        Ignored if a projection is given.
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
                 dataset_targets: Union[tf.Tensor, np.ndarray] = None,
                 labels_dataset: Union[tf.Tensor, np.ndarray] = None,
                 search_method: Type[BaseSearchMethod] = SklearnKNN,
                 k: int = 1,
                 distance: Union[str, Callable] = "euclidean",
                 projection: Optional[Union[Projection, Callable]] = None,
                 returns: Optional[Union[List[str], str]] = "examples",
                 model: Optional[Callable] = None,
                 latent_layer: Optional[Union[str, int]] = None,
                 attribution_method: Type[BlackBoxExplainer] = Saliency,
                 **attribution_kwargs,
                 ):
        # set attributes
        if isinstance(case_dataset, tuple):
            # assuming (x_train, y_train)
            self.case_dataset = case_dataset[0]
            self.labels_dataset = case_dataset[1]
        else:
            self.case_dataset = case_dataset
            self.labels_dataset = labels_dataset
        self.dataset_targets = dataset_targets
        self.k = k
        self.set_returns(returns)

        # set projection
        if projection is not None:
            self.projection = projection
        else:
            if model is None:
                raise ValueError("The Cole method use attribution projection,"+\
                                 "either provide a projection or a model and an attribution method")
            else:
                self.projection = AttributionProjection(model=model, method=attribution_method,
                                                        latent_layer=latent_layer,
                                                        **attribution_kwargs)
        
        # project dataset
        projected_dataset = self.projection(self.case_dataset, self.dataset_targets)

        # initiate search_method
        self.search_method = search_method(search_set=projected_dataset, k=k, distance=distance,
                                           algorithm='auto', returns=["indices", "distances"])

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """
        Compute examples to explain the inputs.
        It project inputs with `self.projection` in the search space
        and find examples with `self.search_method`.
        
        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Tensor or Array passed to the projection function.
            Here it is used by the explain function of attribution methods.
            Refer to the corresponding method documentation for more detail.
            Note that the default method is `Saliency`.
            
        Returns
        -------
        return_dict
            Dictionnary with listed elements in `self.returns`.
            If only one element is present it returns the element.
            The elements that can be returned are:
            examples, weights, distances, indices, and labels.
        """
        # project inputs
        projected_inputs = self.projection(inputs, targets)

        # look for closest elements to projected inputs
        search_output = self.search_method(projected_inputs)

        # manage returned elements
        return_dict = {}

        # add examples and weights
        if "examples" or "weights" in self.returns:
            # get examples from indices
            examples = tf.gather(self.case_dataset, search_output["indices"])
            if targets is not None:
                examples_targets = tf.gather(self.dataset_targets, search_output["indices"])
            if "include_inputs" in self.returns:
                # include inputs
                inputs = tf.expand_dims(inputs, axis=1)
                examples = tf.concat([inputs, examples], axis=1)
                if targets is not None:
                    targets = tf.expand_dims(targets, axis=1)
                    examples_targets = tf.concat([targets, examples_targets], axis=1)
                else:
                    examples_targets = [None] * len(examples)
            if "examples" in self.returns:
                return_dict["examples"] = examples
            if "weights" in self.returns:
                # get weights of examples (n, k, ...)
                # we iterate on the inputs dimension through maps 
                # and ask weights for batch of examples
                weights = []
                for ex, ex_targ in zip(examples, examples_targets):
                    weights.append(self.projection.get_input_weights(ex, ex_targ))
                return_dict["weights"] = tf.stack(weights, axis=0)

                # optimization test
                # return_dict["weights"] = tf.vectorized_map(
                #     fn=lambda x: self.projection.get_input_weights(x[0], x[1]),
                #     elems=(examples, examples_targets),
                #     # fn_output_signature=tf.float32,
                # )
        
        # add indices, distances, and labels
        if "distances" in self.returns:
            return_dict["distances"] = search_output["distances"]
        if "labels" in self.returns:
            assert self.labels_dataset is not None, "Cole method cannot return labels without a label dataset."
            return_dict["labels"] = tf.gather(self.labels_dataset, search_output["indices"])
        
        # Return a dict only different variables are returned
        if len(return_dict) == 1:
            return list(return_dict.values())[0]
        else:
            return return_dict