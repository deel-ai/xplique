"""
Implementation of both counterfactuals and semi factuals methods for classification tasks.

SM CF guided to be implemented (I think): KLEOR at least Sim-Miss and Global-Sim
SM CF free to be implemented: MDN but has to be adapated, Local-Region Model??  
"""
import numpy as np
import tensorflow as tf

from ..types import Callable, List, Optional, Union, Dict
from ..commons import sanitize_inputs_targets

from .base_example_method import BaseExampleMethod
from .search_methods import ORDER, FilterKNN, KLEORSimMiss, KLEORGlobalSim
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
        search_method = FilterKNN

        if projection is None:
            projection = Projection(space_projection=lambda inputs: inputs)

        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            search_method=search_method,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
            distance=distance,
            filter_fn=self.filter_fn,
            order = ORDER.ASCENDING
        )


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
        search_method = FilterKNN

        if projection is None:
            projection = Projection(space_projection=lambda inputs: inputs)
        # TODO: add a warning here if it is a custom projection that requires using targets as it might mismatch with the explain

        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            search_method=search_method,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
            distance=distance,
            filter_fn=self.filter_fn,
            order = ORDER.ASCENDING
        )

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

class KLEOR(BaseExampleMethod):
    """
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
        strategy: str = "sim_miss",
    ):
        
        self.k = k
        self.set_returns(case_returns)

        if strategy == "global_sim":
            search_method = KLEORGlobalSim
        elif strategy == "sim_miss":
            search_method = KLEORSimMiss
        else:
            raise ValueError("strategy should be either 'global_sim' or 'sim_miss'.")

        if projection is None:
            projection = Projection(space_projection=lambda inputs: inputs)

        # set attributes
        batch_size = super()._initialize_cases_dataset(
            cases_dataset, labels_dataset, targets_dataset, batch_size
        )

        assert hasattr(projection, "__call__"), "projection should be a callable."

        # check projection type
        if isinstance(projection, Projection):
            self.projection = projection
        elif hasattr(projection, "__call__"):
            self.projection = Projection(get_weights=None, space_projection=projection)
        else:
            raise AttributeError(
                "projection should be a `Projection` or a `Callable`, not a"
                + f"{type(projection)}"
            )

        # project dataset
        projected_cases_dataset = self.projection.project_dataset(self.cases_dataset,
                                                                  self.targets_dataset)

        # set `search_returns` if not provided and overwrite it otherwise
        if isinstance(case_returns, list) and ("nuns" in case_returns):
            search_method_returns = ["indices", "distances", "nuns"]
        else:
            search_method_returns = ["indices", "distances"]

        # initiate search_method
        self.search_method = search_method(
            cases_dataset=projected_cases_dataset,
            targets_dataset=self.targets_dataset,
            k=k,
            search_returns=search_method_returns,
            batch_size=batch_size,
            distance=distance,
        )

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
        possibilities = ["examples", "weights", "distances", "labels", "include_inputs", "nuns"]
        default = "examples"
        self.returns = _sanitize_returns(returns, possibilities, default)

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
            if isinstance(return_dict, dict):
                return_dict["nuns"] = search_output["nuns"]
            else:
                # find the other only key
                other_key = [k for k in self.returns if k != "nuns"][0]
                return_dict = {
                    other_key: return_dict,
                    "nuns": search_output["nuns"]
                }
        return return_dict
