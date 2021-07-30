"""
Representativity & Consistency metric
"""

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

from ..commons import numpy_sanitize
from ..types import Callable, Optional, Union, Tuple, Dict


class MeGe:
    """
    Used to calculate metrics for representativity and consistency of explanations.
    MeGe (representativity) gives you an overview of the generalization of your explanations:
    how much seeing one explanation informs you about the others.
    ReCo (consistency) is the consistency score of the explanations, it informs about the
    confidence of the explanations, how much two models explanations will not contradict
    each other.

    Ref. Fel, Vigouroux & al., How good is your explanation? Algorithmic stability measures to
    assess the quality of explanations for deep neural networks (2020).
    https://arxiv.org/abs/2009.04521

    Parameters
    ----------
    learning_algorithm
        Function that will be called with (x_train, y_train, x_test, y_test) and that should
        return a Keras Model.
    inputs
        Input samples under study.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    k_splits
        Number of splits to estimate the metrics (usually between 4 and 6 are enough).
    """
    # pylint: disable=R0902

    def __init__(self,
                 learning_algorithm: Callable,
                 inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
                 batch_size: Optional[int] = 64,
                 k_splits: int = 4):
        self.inputs, self.targets = numpy_sanitize(inputs, targets)
        self.batch_size = batch_size
        self.k_splits = k_splits

        # start by splitting the dataset in K splits, and train K model on K-1 split
        assert len(inputs) % k_splits == 0, "`k_splits` must divide the length of the dataset."

        self.split_len = len(inputs) // k_splits
        self.x_splits = np.array([
            self.inputs[i * self.split_len:(i + 1) * self.split_len] for i in range(k_splits)])
        self.y_splits = np.array([
            self.targets[i * self.split_len:(i + 1) * self.split_len] for i in range(k_splits)])

        self.models = []
        for i in range(k_splits):
            # model_i is trained with the split k \ k_i (without k_i)
            ids_to_keep = [j for j in range(k_splits) if j != i]
            x_train = self.x_splits[ids_to_keep].reshape((-1, *self.inputs.shape[1:]))
            y_train = self.y_splits[ids_to_keep].reshape((-1, *self.targets.shape[1:]))

            model = learning_algorithm(x_train, y_train, self.x_splits[i], self.y_splits[i])
            self.models.append(model)

    def evaluate(self,
                 explainer_class: Callable,
                 explainer_params: Optional[Dict] = None) -> Tuple[float, float]:
        # pylint: disable=C0103,R1702
        """
        Evaluate the MeGe score.

        Parameters
        ----------
        explainer_class
            Explainer Attribution class to use.
        explainer_params
            Explainer dict parameters to use to create each explainer.

        Returns
        -------
        mege
            The Mean representativity score.
        reco
            The Relative Consistency score.
        """
        explainer_params = explainer_params or {}

        # for each model, we will need his explanation and his predictions
        predictions = np.array([
            np.argmax(model.predict(self.inputs, batch_size=self.batch_size), -1)
            for model in self.models
        ])
        explanations = np.array([
            explainer_class(model, **explainer_params)(self.inputs, self.targets)
            for model in self.models
        ])

        s_eq, s_ne = self._pairwise_distances(predictions, explanations)
        s_total = np.concatenate([s_eq, s_ne], axis=0)

        mege = 1.0 / (1.0 + np.mean(s_eq))

        reco = 0.0
        step = (len(s_total) // 20) + 1 # skip some element to speed up computation
        for gamma in np.sort(s_total)[::step]:
            tp = np.sum(s_eq < gamma)
            tn = np.sum(s_ne > gamma)
            fp = np.sum(s_ne < gamma)
            fn = np.sum(s_eq > gamma)

            tpr = tp / (tp + fn + 1e-10)
            tnr = tn / (tn + fp + 1e-10)

            reco_gamma = tpr + tnr - 1.0
            reco = np.max([reco, reco_gamma])

        return mege, reco

    def _pairwise_distances(self,
                            predictions: np.array,
                            explanations: np.array) -> Tuple[np.array, np.array]:
        """
        Compute all the pairwise distance between the models explanations.
        Use those distances to build two sets: s_equal for the distance between explanations of
        the same predictions, and s_not_equal for the distances between explanations of
        different predictions.

        Parameters
        ----------
        predictions
            Models predictions (one for each explanation).
        explanations
            Models explanations (one for each prediction).

        Returns
        -------
        s_eq
            The set of distances when the predictions are the same.
        s_ne
            The set of distances when the predictions are different.
        """
        # pylint: disable=C0103,R1702
        # compute pairwise distances and add them into correct distribution
        # s_eq is S^= (eq.3) and s_ne is S^{!=} (eq.4)
        s_eq, s_ne = [], []

        for i in range(self.k_splits):
            for j in range(i+1, self.k_splits):
                # get all the explanations for the pairs (model_i, model_j)
                # with i != j
                e_i = explanations[i]
                e_j = explanations[j]

                for n, e_i_n in enumerate(e_i):
                    # compute only the distance between one model
                    # that has seen the data and the other who don't
                    # to ensure that, the split_id must be i or j
                    # that would mean that one the model has use this split to test
                    split_id = n // self.split_len
                    compute_distance = split_id in [i, j]

                    if compute_distance:
                        pred_i = predictions[i, n]
                        pred_j = predictions[j, n]

                        dist = self._spearman_distance(e_i_n, e_j[n])

                        # we need at least one good answer
                        if pred_i == np.argmax(self.targets[n], -1) or \
                           pred_j == np.argmax(self.targets[n], -1):
                            # both are correct, add to S=
                            if pred_i == pred_j:
                                s_eq.append(dist)
                            # one is correct, add to S!=
                            else:
                                s_ne.append(dist)

        # ignore possible nan values (e.g grad-cam is full of zero)
        s_eq = np.array(s_eq)[~np.isnan(s_eq)]
        s_ne = np.array(s_ne)[~np.isnan(s_ne)]

        return s_eq, s_ne

    @staticmethod
    def _spearman_distance(explanation_a: Union[tf.Tensor, np.array],
                           explanation_b: Union[tf.Tensor, np.array]) -> float:
        """
        Compute the spearman distance between two explanations.

        Parameters
        ----------
        explanation_a
            First explanation as tensor or numpy array.
        explanation_b
            Second explanation as tensor or numpy array.

        Returns
        -------
        dist
            The spearman distance defined as (1 - |spearman_corr(., .)|)**0.5
        """
        explanation_a = MeGe._sanitize_explanation(explanation_a)
        explanation_b = MeGe._sanitize_explanation(explanation_b)

        rho, _ = spearmanr(explanation_a, explanation_b)
        dist = np.sqrt(1.0 - np.abs(rho))
        return dist


    @staticmethod
    def _sanitize_explanation(explanation: Union[tf.Tensor, np.array]) -> np.array:
        """
        Sanitize explanations to ensure we can properly compute the spearman distance.

        Parameters
        ----------
        explanation
            Explanation to sanitize.

        Returns
        -------
        explanation
            Explanation sanitized.
        """
        if len(explanation.shape) > 2:
            explanation = np.mean(explanation, -1)
        explanation = np.nan_to_num(explanation, 0.0).flatten()

        return explanation
