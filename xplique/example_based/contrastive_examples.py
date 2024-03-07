"""
Implementation of both counterfactuals and semi factuals methods for classification tasks.
"""
import numpy as np
import tensorflow as tf

from ..types import Callable, List, Optional, Union

from .base_example_method import BaseExampleMethod
from .search_methods import BaseSearchMethod, KNN, ORDER, FilterKNN
from .projections import Projection

class NaiveSemiFactuals(BaseExampleMethod):
    """
    Define a naive version of semi factuals search. That for a given sample
    it will return the farthest sample which have the same label.
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
            order = ORDER.DESCENDING
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
        mask = tf.equal(tf.expand_dims(predicted_labels, axis=1), label_targets) #(n, bs)
        return mask

class PredictedLabelAwareSemiFactuals():
    def __init__(self) -> None:
        raise NotImplementedError

class NaiveCounterFactuals(BaseExampleMethod):
    def __init__():
        raise NotImplementedError
