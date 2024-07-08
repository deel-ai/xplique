"""
Base model for prototypes
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ..types import Callable, Dict, List, Optional, Type, Union

from .search_methods import BaseSearchMethod, ProtoGreedySearch, MMDCriticSearch, ProtoDashSearch
from .projections import Projection
from .base_example_method import BaseExampleMethod


class Prototypes(BaseExampleMethod, ABC):
    """
    Base class for prototypes.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    labels_dataset
        Labels associated to the examples in the dataset. Indices should match with cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other dataset should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    targets_dataset
        Targets associated to the cases_dataset for dataset projection. See `projection` for detail.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other dataset should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space should be a space where distance make sense for the model.
        It should not be `None`, otherwise,
        all examples could be computed only with the `search_method`.

        Example of Callable:
        ```
        def custom_projection(inputs: tf.Tensor, np.ndarray, targets: tf.Tensor, np.ndarray = None):
            '''
            Example of projection,
            inputs are the elements to project.
            targets are optional parameters to orientated the projection.
            '''
            projected_inputs = # do some magic on inputs, it should use the model.
            return projected_inputs
        ```
    case_returns
        String or list of string with the elements to return in `self.explain()`.
        See `self.set_returns()` for detail.
    batch_size
        Number of sample treated simultaneously for projection and search.
        Ignored if `tf.data.Dataset` are provided (those are supposed to be batched).
    search_method_kwargs
        Parameters to be passed at the construction of the `search_method`.
    """

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        targets_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = None,
        nb_prototypes: int = 1,
        kernel_type: str = 'local', 
        kernel_fn: callable = None,
        gamma: float = None
    ):     
        # set common example-based parameters   
        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
        )

        # set prototypes parameters
        self.distance = distance
        self.nb_prototypes = nb_prototypes
        self.kernel_type = kernel_type
        self.kernel_fn = kernel_fn
        self.gamma = gamma

        # initiate search_method
        self.search_method = self.search_method_class(
            cases_dataset=self.cases_dataset,
            labels_dataset=self.labels_dataset,
            k=self.k,
            search_returns=self._search_returns,
            batch_size=self.batch_size,
            distance=self.distance,
            nb_prototypes=self.nb_prototypes,
            kernel_type=self.kernel_type,
            kernel_fn=self.kernel_fn,
            gamma=self.gamma
        )
    
    @property
    @abstractmethod
    def search_method_class(self) -> Type[ProtoGreedySearch]:
        raise NotImplementedError
  
    def get_global_prototypes(self):
        """
        Return all the prototypes computed by the search method, 
        which consist of a global explanation of the dataset.

        Returns:
            prototype_indices : Tensor
                prototype indices. 
            prototype_weights : Tensor    
                prototype weights.
        """
        return self.search_method.prototype_indices, self.search_method.prototype_weights


class ProtoGreedy(Prototypes):
    @property
    def search_method_class(self) -> Type[ProtoGreedySearch]:
        return ProtoGreedySearch


class MMDCritic(Prototypes):
    @property
    def search_method_class(self) -> Type[ProtoGreedySearch]:
        return MMDCriticSearch


class ProtoDash(Prototypes):
    """
    Protodash method for searching prototypes.

    References:
    .. [#] `Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi,
        "ProtoDash: Fast Interpretable Prototype Selection"
        <https://arxiv.org/abs/1707.01212>`_

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods it is the train dataset.
    labels_dataset
        Labels associated to the examples in the dataset. Indices should match with cases_dataset.
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
    kernel_type : str, optional
        The kernel type. It can be 'local' or 'global', by default 'local'.
        When it is local, the distances are calculated only within the classes.
    kernel_fn : Callable, optional
        Kernel function, by default the rbf kernel.
        This function must only use TensorFlow operations.
    gamma : float, optional
        Parameter that determines the spread of the rbf kernel, defaults to 1.0 / n_features.
    use_optimizer : bool, optional
        Flag indicating whether to use an optimizer for prototype selection, by default False.
    """

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = None,
        nb_prototypes: int = 1,
        kernel_type: str = 'local', 
        kernel_fn: callable = None,
        gamma: float = None,
        use_optimizer: bool = False,
    ): # pylint: disable=R0801
        self.use_optimizer = use_optimizer

        super().__init__(
            cases_dataset=cases_dataset, 
            labels_dataset=labels_dataset, 
            k=k,
            projection=projection,
            case_returns=case_returns, 
            batch_size=batch_size, 
            distance=distance, 
            nb_prototypes=nb_prototypes, 
            kernel_type=kernel_type, 
            kernel_fn=kernel_fn,
            gamma=gamma
        )

    @property
    def search_method_class(self) -> Type[ProtoGreedySearch]:
        return ProtoDashSearch
    
