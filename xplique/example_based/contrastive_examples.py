"""
Implementation of both counterfactuals and semi factuals methods for classification tasks.

SM CF guided to be implemented (I think): KLEOR at least Sim-Miss and Global-Sim
SM CF free to be implemented: MDN but has to be adapated, Local-Region Model??  
"""
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
    This class allows to search for counterfactuals by searching for the closest sample that do not have the same label.
    It is a naive approach as it follows a greedy approach.
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

        if projection is None:
            projection = Projection(space_projection=lambda inputs: inputs)

        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
        )
        
        self.distance = distance
        self.order = ORDER.ASCENDING

        self.search_method = self.search_method_class(
            cases_dataset=self.cases_dataset,
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
        return FilterKNN


    def filter_fn(self, _, __, targets, cases_targets) -> tf.Tensor:
        """
        Filter function to mask the cases for which the label is different from the predicted
        label on the inputs.
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
    This method will search the counterfactuals with a specific label. This label should be provided by the user in the
    cf_labels_dataset args.
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
        if projection is None:
            projection = Projection(space_projection=lambda inputs: inputs)
        # TODO: add a warning here if it is a custom projection that requires using targets as it might mismatch with the explain

        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
        )
        
        self.distance = distance
        self.order = ORDER.ASCENDING

        self.search_method = self.search_method_class(
            cases_dataset=self.cases_dataset,
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
        return FilterKNN


    def filter_fn(self, _, __, cf_targets, cases_targets) -> tf.Tensor:
        """
        Filter function to mask the cases for which the label is different from the label(s) expected for the
        counterfactuals.

        Parameters
        ----------
        cf_targets
            TODO
        cases_targets
            TODO
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
        Compute examples to explain the inputs.
        It project inputs with `self.projection` in the search space
        and find examples with `self.search_method`.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        cf_targets
            TODO: change the description here

        Returns
        -------
        return_dict
            Dictionary with listed elements in `self.returns`.
            If only one element is present it returns the element.
            The elements that can be returned are:
            examples, weights, distances, indices, and labels.
        """
        # TODO make an assert on the cf_targets
        return super().explain(inputs, cf_targets)


class KLEORBase(BaseExampleMethod):
    """
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

        if projection is None:
            projection = Projection(space_projection=lambda inputs: inputs)

        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
        )
        
        self.distance = distance
        self.order = ORDER.ASCENDING

        self.search_method = self.search_method_class(
            cases_dataset=self.cases_dataset,
            targets_dataset=self.targets_dataset,
            k=self.k,
            search_returns=self._search_returns,
            batch_size=self.batch_size,
            distance=self.distance,
        )

    @property
    def returns(self) -> Union[List[str], str]:
        """Getter for the returns parameter."""
        return self._returns

    @returns.setter
    def returns(self, returns: Union[List[str], str]):
        """
        """
        default = "examples"
        self._returns = _sanitize_returns(returns, self._returns_possibilities, default)
        self._search_returns = ["indices", "distances"]

        if isinstance(self._returns, list) and ("nuns" in self._returns):
            self._search_returns.append("nuns_indices")
        elif isinstance(self._returns, list) and ("nuns_indices" in self._returns):
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
        """
        return_dict = super().format_search_output(search_output, inputs, targets)
        if "nuns" in self.returns:
            return_dict["nuns"] = dataset_gather(self.cases_dataset, search_output["nuns_indices"])
        if "nuns_indices" in self.returns:
            return_dict["nuns_indices"] = search_output["nuns_indices"]
        if "dist_to_nuns" in self.returns:
            return_dict["dist_to_nuns"] = search_output["dist_to_nuns"]
        return return_dict


class KLEORGlobalSim(KLEORBase):
    @property
    def search_method_class(self):
        return KLEORGlobalSimSearch


class KLEORSimMiss(KLEORBase):
    @property
    def search_method_class(self):
        return KLEORSimMissSearch
