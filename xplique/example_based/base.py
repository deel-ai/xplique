"""
Base model for example-based 
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ..types import Callable, Dict, List, Optional, Tuple, Type, Union

from ..attributions.base import BlackBoxExplainer, sanitize_input_output
from .search_methods import BaseSearchMethod, SklearnKNN
from .projections import Projection


class NaturalExampleBasedExplainer(ABC):
    """
    Base class for natural example-base methods explaining models,
    they project the case_dataset into a pertinent space for the with a `Projection`,
    then they call the `BaseSearchMethod` on it.

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
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space sould be a space where distance make sense for the model.
        It should not be `None`, otherwise,
        all examples could be computed only with the `search_method`.
        Example of Callable:
        ```
        def customn_projection(inputs: tf.Tensor, np.ndarray, targets: tf.Tensor, np.ndarray = None):
            '''
            Example of projection,
            inputs are the elements to project.
            targets are optionnal parameters to orientated the projection.
            '''
            projected_inputs = # do some magic on inputs, it should use the model.
            return projected_inputs
        ```
    returns
        String or list of string with the elements to return in `self.explain()`.
        See `self.set_returns()` for detail.
    """

    @abstractmethod
    def __init__(self,
                 case_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 dataset_targets: Union[tf.Tensor, np.ndarray] = None,
                 search_method: Type[BaseSearchMethod] = SklearnKNN,
                 k: int = 1,
                 projection: Optional[Union[Projection, Callable]] = None,
                 returns: Union[List[str], str] = "examples"):
        raise NotImplementedError

    def set_k(self, k: int):
        assert isinstance(k, int) and k >= 1, f"k should be an int >= 1 and not {k}"
        self.k = k
        try:
            self.search_method.set_k(k)
        except AttributeError:
            pass

    def set_returns(self, returns: Union[List[str], str]):
        """
        Set `self.returns` used to define returned elements in `self.explain()`.
        
        Parameters
        ----------
        returns
            Most elements are useful in `xplique.plots.plot_examples()`.
            `returns` can be set to 'all' for all possible elements to be returned.
                - 'examples' correspond to the expected examples,
                the inputs may be included in first position. (n, k(+1), ...)
                - 'weights' the weights in the input space used in the projection.
                They are associated to the input and the examples. (n, k(+1), ...)
                - 'distances' the distances between the inputs and the corresponding examples.
                They are associated to the examples. (n, k, ...)
                - 'labels' if provided through `dataset_labels`,
                they are the labels associated with the examples. (n, k, ...)
                - 'include_inputs' specify if inputs should be included in the returned elements.
                Note that it changes the number of returned elements from k to k+1.
        """
        if returns is None:
            self.returns = ["examples"]
        elif isinstance(returns, str):
            possibilities = ["examples", "weights", "distances", "labels", "include_inputs"]
            if returns == "all":
                self.returns = possibilities
            elif returns in possibilities:
                self.returns = [returns]
            else:
                raise ValueError(f"{returns} should belong to {possibilities}")
        elif isinstance(returns, list):
            self.returns = returns
        else:
            raise ValueError(f"{returns} should either be `str` or `List[str]`")
        

    @abstractmethod
    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """
        Compute examples to explain the inputs.
        It project inputs with `self.projection` in the search space
        and find examples with `self.search_method`.
        
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
        raise NotImplementedError()

    def __call__(self,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """explain alias"""
        return self.explain(inputs, targets)