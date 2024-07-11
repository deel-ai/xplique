"""
Implementation of both counterfactuals and semi factuals methods for classification tasks.
"""
import warnings

import numpy as np
import tensorflow as tf

from ..types import Callable, List, Optional, Union, Dict
from ..commons import sanitize_inputs_targets, dataset_gather

from .base_example_method import BaseExampleMethod
from .search_methods import ORDER, FilterKNN, KLEORSimMissSearch, KLEORGlobalSimSearch
from .projections import Projection

from .search_methods.base import _sanitize_returns

class NaiveCounterFactuals(BaseExampleMethod):
    """
    This class allows to search for counterfactuals by searching for the closest sample to a query in a projection space
    that do not have the same model's prediction. 
    It is a naive approach as it follows a greedy approach.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from this dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    targets_dataset
        Targets are expected to be the one-hot encoding of the model's predictions for the samples in cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    labels_dataset
        Labels associated to the examples in the dataset. Indices should match with cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve per input.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space should be a space where distances are relevant for the model.
        It should not be `None`, otherwise, the model is not involved thus not explained. If you are interested in
        searching the input space, you should use a `BaseSearchMethod` instead. 

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
        See the base class returns property for more details.
    batch_size
        Number of sample treated simultaneously for projection and search.
        Ignored if `tf.data.Dataset` are provided (those are supposed to be batched).
    distance
        Distance for the FilterKNN search method.
        Either a Callable, or a value supported by `tf.norm` `ord` parameter.
        Their documentation (https://www.tensorflow.org/api_docs/python/tf/norm) say:
        "Supported values are 'fro', 'euclidean', 1, 2, np.inf and any positive real number
        yielding the corresponding p-norm." We also added 'cosine'.
    """
    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = "euclidean",
    ):
        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
        )

        # set distance function and order for the search method
        self.distance = distance
        self.order = ORDER.ASCENDING

        # initiate search_method
        self.search_method = self.search_method_class(
            cases_dataset=self.projected_cases_dataset,
            targets_dataset=self.targets_dataset,
            k=self.k,
            search_returns=self._search_returns,
            batch_size=self.batch_size,
            distance=distance,
            filter_fn=self.filter_fn,
            order=self.order
        )

    @property
    def search_method_class(self):
        """
        This property defines the search method class to use for the search. In this case, it is the FilterKNN that
        is an efficient KNN search method ignoring non-acceptable cases, thus not considering them in the search.
        """
        return FilterKNN


    def filter_fn(self, _, __, targets, cases_targets) -> tf.Tensor:
        """
        Filter function to mask the cases for which the model's prediction is different from the model's prediction
        on the inputs.
        """
        # get the labels predicted by the model
        # (n, )
        predicted_labels = tf.argmax(targets, axis=-1)

        # for each input, if the target label is the same as the predicted label
        # the mask as a True value and False otherwise
        label_targets = tf.argmax(cases_targets, axis=-1) # (bs,)
        mask = tf.not_equal(tf.expand_dims(predicted_labels, axis=1), label_targets) #(n, bs)
        return mask

class LabelAwareCounterFactuals(BaseExampleMethod):
    """
    This method will search the counterfactuals of a query within an expected class. This class should be provided with
    the query when calling the explain method.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from this dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    targets_dataset
        Targets are expected to be the one-hot encoding of the model's predictions for the samples in cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    labels_dataset
        Labels associated to the examples in the dataset. Indices should match with cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve per input.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space should be a space where distances are relevant for the model.
        It should not be `None`, otherwise, the model is not involved thus not explained. If you are interested in
        searching the input space, you should use a `BaseSearchMethod` instead. 

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
        See the base class returns property for more details.
    batch_size
        Number of sample treated simultaneously for projection and search.
        Ignored if `tf.data.Dataset` are provided (those are supposed to be batched).
    distance
        Distance for the FilterKNN search method.
        Either a Callable, or a value supported by `tf.norm` `ord` parameter.
        Their documentation (https://www.tensorflow.org/api_docs/python/tf/norm) say:
        "Supported values are 'fro', 'euclidean', 1, 2, np.inf and any positive real number
        yielding the corresponding p-norm." We also added 'cosine'.
    """
    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = "euclidean",
    ):

        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
        )

        # raise a warning to specify that target in the explain method is not the same as the target used for
        # the target dataset
        warnings.warn("If your projection method requires the target, be aware that when using the explain method,"
                        " the target provided is the class within one should search for the counterfactual.\nThus,"
                        " it is possible that the projection of the query is going wrong.")

        # set distance function and order for the search method
        self.distance = distance
        self.order = ORDER.ASCENDING

        # initiate search_method
        self.search_method = self.search_method_class(
            cases_dataset=self.projected_cases_dataset,
            targets_dataset=self.targets_dataset,
            k=self.k,
            search_returns=self._search_returns,
            batch_size=self.batch_size,
            distance=distance,
            filter_fn=self.filter_fn,
            order=self.order
        )
    
    @property
    def search_method_class(self):
        """
        This property defines the search method class to use for the search. In this case, it is the FilterKNN that
        is an efficient KNN search method ignoring non-acceptable cases, thus not considering them in the search.
        """
        return FilterKNN


    def filter_fn(self, _, __, cf_targets, cases_targets) -> tf.Tensor:
        """
        Filter function to mask the cases for which the target is different from the target(s) expected for the
        counterfactuals.

        Parameters
        ----------
        cf_targets
            The one-hot encoding of the target class for the counterfactuals.
        cases_targets
            The one-hot encoding of the target class for the cases.
        """
        mask = tf.matmul(cf_targets, cases_targets, transpose_b=True) #(n, bs)
        # TODO: I think some retracing are done here
        mask = tf.cast(mask, dtype=tf.bool)
        return mask

    @sanitize_inputs_targets
    def explain(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        cf_targets: Union[tf.Tensor, np.ndarray],
    ):
        """
        Return the relevant CF examples to explain the inputs.
        The CF examples are searched within cases for which the target is the one provided in `cf_targets`.
        It projects inputs with `self.projection` in the search space and find examples with the `self.search_method`.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        cf_targets
            Tensor or Array. One-hot encoding of the target class for the counterfactuals.

        Returns
        -------
        return_dict
            Dictionary with listed elements in `self.returns`.
            The elements that can be returned are defined with _returns_possibilities static attribute of the class.
        """
        return super().explain(inputs, cf_targets)


class KLEORBase(BaseExampleMethod):
    """
    Base class for KLEOR methods. KLEOR methods search Semi-Factuals examples. In those methods, one should first
    retrieve the Nearest Unlike Neighbor (NUN) which is the closest example to the query that has a different prediction
    than the query. Then, the method search for the K-Nearest Neighbors (KNN) of the NUN that have the same prediction
    as the query. 
    
    All the searches are done in a projection space where distances are relevant for the model. The projection space is
    defined by the `projection` method.

    Depending on the KLEOR method some additional condition for the search are added. See the specific KLEOR method for
    more details.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from this dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    targets_dataset
        Targets are expected to be the one-hot encoding of the model's predictions for the samples in cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    labels_dataset
        Labels associated to the examples in the dataset. Indices should match with cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve per input.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space should be a space where distances are relevant for the model.
        It should not be `None`, otherwise, the model is not involved thus not explained. If you are interested in
        searching the input space, you should use a `BaseSearchMethod` instead. 

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
        See the base class returns property for more details.
    batch_size
        Number of sample treated simultaneously for projection and search.
        Ignored if `tf.data.Dataset` are provided (those are supposed to be batched).
    distance
        Distance for the FilterKNN search method.
        Either a Callable, or a value supported by `tf.norm` `ord` parameter.
        Their documentation (https://www.tensorflow.org/api_docs/python/tf/norm) say:
        "Supported values are 'fro', 'euclidean', 1, 2, np.inf and any positive real number
        yielding the corresponding p-norm." We also added 'cosine'.
    """
    _returns_possibilities = [
        "examples", "weights", "distances", "labels", "include_inputs", "nuns", "nuns_indices", "dist_to_nuns"
    ]

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = "euclidean",
    ):

        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
        )

        # set distance function and order for the search method
        self.distance = distance
        self.order = ORDER.ASCENDING

        # initiate search_method
        self.search_method = self.search_method_class(
            cases_dataset=self.projected_cases_dataset,
            targets_dataset=self.targets_dataset,
            k=self.k,
            search_returns=self._search_returns,
            batch_size=self.batch_size,
            distance=self.distance,
        )

    @property
    def returns(self) -> Union[List[str], str]:
        """Override the Base class returns' parameter."""
        return self._returns

    @returns.setter
    def returns(self, returns: Union[List[str], str]):
        """
        Set the returns parameter. The returns parameter is a string or a list of string with the elements to return
        in `self.explain()`. The elements that can be returned are defined with _returns_possibilities static attribute
        """
        default = "examples"
        self._returns = _sanitize_returns(returns, self._returns_possibilities, default)
        self._search_returns = ["indices", "distances"]

        if isinstance(self._returns, list) and ("nuns" in self._returns):
            self._search_returns.append("nuns_indices")
        elif isinstance(self._returns, list) and ("nuns_indices" in self._returns):
            self._search_returns.append("nuns_indices")
        elif isinstance(self._returns, list) and ("nuns_labels" in self._returns):
            self._search_returns.append("nuns_indices")

        if isinstance(self._returns, list) and ("dist_to_nuns" in self._returns):
            self._search_returns.append("dist_to_nuns")
        
        try:
            self.search_method.returns = self._search_returns
        except AttributeError:
            pass

    def format_search_output(
        self,
        search_output: Dict[str, tf.Tensor],
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None, 
    ):
        """
        Format the output of the `search_method` to match the expected returns in `self.returns`.

        Parameters
        ----------
        search_output
            Dictionary with the required outputs from the `search_method`.
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
        targets
            Targets associated to the cases_dataset for dataset projection.
            See `projection` for details.

        Returns
        -------
        return_dict
            Dictionary with listed elements in `self.returns`.
            The elements that can be returned are defined with _returns_possibilities static attribute of the class.
        """
        return_dict = super().format_search_output(search_output, inputs, targets)
        if "nuns" in self.returns:
            return_dict["nuns"] = dataset_gather(self.cases_dataset, search_output["nuns_indices"])
        if "nuns_labels" in self.returns:
            return_dict["nuns_labels"] = dataset_gather(self.labels_dataset, search_output["nuns_indices"])
        if "nuns_indices" in self.returns:
            return_dict["nuns_indices"] = search_output["nuns_indices"]
        if "dist_to_nuns" in self.returns:
            return_dict["dist_to_nuns"] = search_output["dist_to_nuns"]
        return return_dict


class KLEORSimMiss(KLEORBase):
    """
    The KLEORSimMiss method search for Semi-Factuals examples by searching for the Nearest Unlike Neighbor (NUN) of
    the query. The NUN is the closest example to the query that has a different prediction than the query. Then, the
    method search for the K-Nearest Neighbors (KNN) of the NUN that have the same prediction as the query.

    The search is done in a projection space where distances are relevant for the model. The projection space is defined
    by the `projection` method.
    """
    @property
    def search_method_class(self):
        """
        This property defines the search method class to use for the search. In this case, it is the KLEORSimMissSearch.
        """
        return KLEORSimMissSearch

class KLEORGlobalSim(KLEORBase):
    """
    The KLEORGlobalSim method search for Semi-Factuals examples by searching for the Nearest Unlike Neighbor (NUN) of
    the query. The NUN is the closest example to the query that has a different prediction than the query. Then, the
    method search for the K-Nearest Neighbors (KNN) of the NUN that have the same prediction as the query.

    In addition, for a SF candidate to be considered, the SF should be closer to the query than the NUN in the
    projection space (i.e. the SF should be 'between' the input and its NUN). This condition is added to the search.

    The search is done in a projection space where distances are relevant for the model. The projection space is defined
    by the `projection` method.
    """
    @property
    def search_method_class(self):
        """
        This property defines the search method class to use for the search. In this case, it is the
        KLEORGlobalSimSearch.
        """
        return KLEORGlobalSimSearch
