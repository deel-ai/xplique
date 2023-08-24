"""
Module related to Kernel SHAP method
"""

import tensorflow as tf
import numpy as np
from sklearn import linear_model

from .lime import Lime
from ..commons import Tasks
from ..types import Callable, Union, Optional, OperatorSignature

class KernelShap(Lime):
    """
    By setting appropriately the perturbation function, the similarity kernel and the interpretable
    model in the LIME framework we can theoretically obtain the Shapley Values more efficiently.
    Therefore, KernelShap is a method based on LIME with specific attributes.

    More information regarding this method and proof of equivalence can be found in the
    original paper here:
    https://arxiv.org/abs/1705.07874

    Parameters
    ----------
    model
        The model from which we want to obtain explanations.
    batch_size
        Number of perturbed samples to process at once, mandatory when nb_samples is huge.
        Notice, it is different compare to WhiteBox explainers which batch the inputs.
        Here inputs are process one by one.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    map_to_interpret_space
        Function which group features of an input corresponding to the same interpretable
        feature (e.g super-pixel).
        It allows to transpose from (resp. to) the original input space to (resp. from)
        the interpretable space.
    nb_samples
        The number of perturbed samples you want to generate for each input sample.
        Default to 800.
    ref_value
        It defines reference value which replaces each feature when the corresponding
        interpretable feature is set to 0.
        It should be provided as: a ndarray of shape (1) if there is no channels in your input
        and (C,) otherwise.
        The default ref value is set to (0.5,0.5,0.5) for inputs with 3 channels (corresponding
        to a grey pixel when inputs are normalized by 255) and to 0 otherwise.
    """
    def __init__(self,
                 model: Callable,
                 batch_size: int = 64,
                 operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
                 map_to_interpret_space: Optional[Callable] = None,
                 nb_samples: int = 800,
                 ref_value: Optional[np.ndarray] = None):

        Lime.__init__(
            self,
            model,
            batch_size,
            operator,
            interpretable_model = linear_model.LinearRegression(),
            similarity_kernel = KernelShap._kernel_shap_similarity_kernel,
            pertub_func = KernelShap._kernel_shap_pertub_func,
            ref_value = ref_value,
            map_to_interpret_space = map_to_interpret_space,
            nb_samples = nb_samples,
            )

    # No need to redifine the explain method (herited from Lime)

    @staticmethod
    @tf.function
    def _kernel_shap_pertub_func(nb_features: Union[int, tf.Tensor],
                                 nb_samples: int) -> tf.Tensor:
        """
        The perturbed instances are sampled that way:
         - We choose a number of selected features k, considering the distribution
                p(k) = (nb_features - 1) / (k * (nb_features - k))
            where nb_features is the total number of features in the interpretable space
         - Then we randomly select a binary vector with k ones, all the possible sample
           are equally likely. It is done by generating a random vector with values drawn
           from a normal distribution and keeping the top k elements which then will be 1
           and other values are 0.
         Since there are nb_features choose k vectors with k ones, this weighted sampling
         is equivalent to applying the Shapley kernel for the sample weight, defined as:
            k(nb_features, k) = (nb_features - 1)/(k*(nb_features - k)*(nb_features choose k))
        This trick is the one used in the Captum library: https://github.com/pytorch/captum
        """

        nb_features = tf.squeeze(nb_features)
        probs_nb_selected_feature = KernelShap._get_probs_nb_selected_feature(
            tf.cast(nb_features, dtype=tf.int32)
        )
        nb_selected_features = tf.random.categorical(tf.math.log([probs_nb_selected_feature]),
                                                     nb_samples,
                                                     dtype=tf.int32)
        nb_selected_features = tf.reshape(nb_selected_features, [nb_samples])
        nb_selected_features = tf.one_hot(nb_selected_features, nb_features, dtype=tf.int32)

        rand_vals = tf.random.normal([nb_samples, nb_features])
        idx_sorted_values = tf.argsort(rand_vals, axis=1, direction='DESCENDING')

        threshold_idx = idx_sorted_values * nb_selected_features
        threshold_idx = tf.reduce_sum(threshold_idx, axis=1)

        threshold = rand_vals * tf.one_hot(threshold_idx, nb_features)
        threshold = tf.reduce_sum(threshold, axis=1)
        threshold = tf.expand_dims(threshold, axis=1)
        threshold = tf.repeat(threshold, repeats=nb_features, axis=1)

        interpret_samples = tf.greater(rand_vals, threshold)
        interpret_samples = tf.cast(interpret_samples, dtype=tf.int32)

        return interpret_samples

    @staticmethod
    @tf.function
    def _get_probs_nb_selected_feature(num_features: Union[int, tf.Tensor]) -> tf.Tensor:
        """
        Compute the distribution:
            p(k) = (nb_features - 1) / (k * (nb_features - k))
        """
        list_features_indexes = tf.range(1, num_features)
        denom = tf.multiply(list_features_indexes, (num_features - list_features_indexes))
        num = num_features - 1
        probs = tf.divide(num, denom)
        probs = tf.concat([[0.0], probs], 0)
        return tf.cast(probs, dtype=tf.float32)

    @staticmethod
    def _kernel_shap_similarity_kernel(
        original_input,
        interpret_samples,
        perturbed_samples
    ) -> tf.Tensor:
    # pylint: disable=unused-argument
        """
        This method compute the similarity between interpretable perturbed samples and
        the original input (i.e a tf.ones(num_features)). The trick used for computation
        reason is to instead of using the original similarity kernel to pick random perturbed
        instances of interpretable sample in order to follow a certain probability rule, we
        let the pertub function create the pertub interpretable sample directly following
        this probability. Therefore, from the pertub function we can pick them with
        equal probability. See the `_kernel_shap_pertub_func` for more details.
        """

        similarities = tf.ones(len(interpret_samples), dtype=tf.float32)

        return similarities
