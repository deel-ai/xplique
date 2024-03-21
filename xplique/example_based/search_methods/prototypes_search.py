"""
Prototypes search method in example-based module
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

from ...commons import dataset_gather
from ...types import Callable, List, Union, Optional, Tuple

from .base import BaseSearchMethod
from ..projections import Projection


class PrototypesSearch(BaseSearchMethod):
    """
    Prototypes search method to find prototypes and the examples closest to these prototypes.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods it is the train dataset.
    k
        The number of examples to retrieve.
    search_returns
        String or list of string with the elements to return in `self.find_examples()`.
        See `self.set_returns()` for detail.
    batch_size
        Number of sample treated simultaneously.
        It should match the batch size of the `search_set` in the case of a `tf.data.Dataset`.
    distance
        Either a Callable, or a value supported by `tf.norm` `ord` parameter.
        Their documentation (https://www.tensorflow.org/api_docs/python/tf/norm) say:
        "Supported values are 'fro', 'euclidean', 1, 2, np.inf and any positive real number
        yielding the corresponding p-norm." We also added 'cosine'.
    nb_prototypes : int
            Number of prototypes to find.    
    find_prototypes_kwargs
        Additional parameters passed to `find_prototypes` function.
    """

    # Avoid zero division during procedure. (the value is not important, as if the denominator is
    # zero, then the nominator will also be zero).
    EPSILON = tf.constant(1e-6)

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        k: int = 1,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = "euclidean",
        nb_prototypes: int = 1,
        **find_prototypes_kwargs,
    ): # pylint: disable=R0801
        super().__init__(
            cases_dataset, labels_dataset, k, search_returns, batch_size
        )

        if hasattr(distance, "__call__"):
            self.distance_fn = distance
        elif distance in ["fro", "euclidean", 1, 2, np.inf] or isinstance(
            distance, int
        ):
            self.distance_fn = lambda x1, x2: tf.norm(x1 - x2, ord=distance)
        else:
            raise AttributeError(
                "The distance parameter is expected to be either a Callable or in"
                + " ['fro', 'euclidean', 'cosine', 1, 2, np.inf] ",
                +f"but {distance} was received.",
            )
  
        self.prototype_indices, self.prototype_weights = self.find_prototypes(nb_prototypes, **find_prototypes_kwargs)

    @abstractmethod
    def find_prototypes(self, nb_prototypes: int, **find_prototypes_kwargs):
        """
        Search for prototypes and their corresponding weights.

        Parameters
        ----------
        nb_prototypes : int
            Number of prototypes to find.

        find_prototypes_kwargs
            Additional parameters passed to `find_prototypes` function.

        Returns
        -------
        prototype_indices : Tensor
            The indices of the selected prototypes.
        prototype_weights : 
            The normalized weights of the selected prototypes.
        """
        return NotImplementedError()

    def find_examples(self, inputs: Union[tf.Tensor, np.ndarray]):
        """
        Search the samples to return as examples. Called by the explain methods.
        It may also return the indices corresponding to the samples,
        based on `return_indices` value.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Assumed to have been already projected.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
        """
        # TODO: Find examples: here we provide a local explanation.
        # Find the nearest prototypes to inputs 
        # we use self.distance_fn and self.prototype_indices.
        examples_indices = None
        examples_distances = None

        # Set values in return dict
        return_dict = {}
        if "examples" in self.returns:
            return_dict["examples"] = dataset_gather(self.cases_dataset, examples_indices)
            if "include_inputs" in self.returns:
                inputs = tf.expand_dims(inputs, axis=1)
                return_dict["examples"] = tf.concat(
                    [inputs, return_dict["examples"]], axis=1
                )
        if "indices" in self.returns:
            return_dict["indices"] = examples_indices
        if "distances" in self.returns:
            return_dict["distances"] = examples_distances            

        # Return a dict only different variables are returned
        if len(return_dict) == 1:
            return list(return_dict.values())[0]
        return return_dict
