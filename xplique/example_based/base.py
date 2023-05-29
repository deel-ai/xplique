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
    they project the case_dataset into a pertinent space for the with Projection,
    then they call the NaturalExampleBasedMethod on it.

    Parameters
    ----------
    case_dataset
        The dataset used to train the model,
        also use by the function to calcul the closest examples.
    search_method
        An algorithm to search the examples in the projected space.
    k
        ...
    projection
        ...
        Say that it cannot be None otherwise we can just use search_method.
        ...
    """

    @abstractmethod
    def __init__(self,
                 case_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 dataset_targets: Union[tf.Tensor, np.ndarray] = None,
                 search_method: Type[BaseSearchMethod] = SklearnKNN,
                 k: int = 1,
                 projection: Optional[Union[Projection, Callable]] = None,
                 returns: Optional[Union[List[str], str]] = None):
        raise NotImplementedError

    def set_k(self, k: int):
        assert isinstance(k, int) and k >= 1, f"k should be an int >= 1 and not {k}"
        self.k = k
        try:
            self.search_method.set_k(k)
        except AttributeError:
            pass

    def set_returns(self, returns: Optional[Union[List[str], str]] = None):
        if returns is None:
            self.returns = ["examples"]
        elif isinstance(returns, str):
            possibilities = ["examples", "weights", "indices", "distances", "include_inputs"]
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
        ...
        """
        raise NotImplementedError()

    def __call__(self,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """explain alias"""
        return self.explain(inputs, targets)