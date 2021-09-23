"""
Module related to Kernel SHAP method
"""

import tensorflow as tf
import numpy as np
from sklearn import linear_model

from .lime import Lime
from ..types import Callable, Union, Optional

class KernelShap(Lime):
    """
    By setting appropriately the pertubation function, the similarity kernel and the interpretable
    model in the LIME framework we can theoretically obtain the Shapley Values more efficiently.
    Therefore, KernelShap is a method based on LIME with specific attributes.

    More information regarding this method and proof of equivalence can be found in the
    original paper here:
    https://arxiv.org/abs/1705.07874
    """
    def __init__(self,
                 model: Callable,
                 batch_size: int = 64,
                 map_to_interpret_space: Optional[Callable] = None,
                 nb_samples: int = 800,
                 ref_value: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        model
            Model that you want to explain.

        batch_size
            The batch size to predict the pertubed samples targets value.
            Default to 64.

        map_to_interpret_space
            Function which group an input features which correspond to the same interpretable
            feature (e.g super-pixel).
            It allows to transpose from (resp. to) the original input space to (resp. from)
            the interpretable space.
            The default mapping is:
                - the quickshift segmentation algorithm for inputs with (N, W, H, C) shape,
                we assume here such shape is used to represent (W, H, C) images.
                - the felzenszwalb segmentation algorithm for inputs with (N, W, H) shape,
                we assume here such shape is used to represent (W, H) images.
                - an identity mapping if inputs has shape (N, W), we assume here your inputs
                are tabular data.

            To use your own custom map function you should use the following scheme:

            def custom_map_to_interpret_space(inputs: tf.tensor (N, W (, H, C) )) ->
            tf.tensor (N, W (, H)):
                **some grouping techniques**
                return mappings

            For instance you can use the scikit-image (as we did for the quickshift algorithm)
            library to defines super pixels on your images..

        nb_samples
            The number of pertubed samples you want to generate for each input sample.
            Default to 800.

        ref_values
            It defines reference value which replaces each feature when the corresponding
            interpretable feature is set to 0.
            It should be provided as: a ndarray of shape (1) if there is no channels in your input
            and (C,) otherwise

            The default ref value is set to (0.5,0.5,0.5) for inputs with 3 channels (corresponding
            to a grey pixel when inputs are normalized by 255) and to 0 otherwise.
        """
        Lime.__init__(
            self,
            model,
            batch_size,
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
        The pertubed instances are sampled that way:
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
        pertubed_samples
    ) -> tf.Tensor:
    # pylint: disable=unused-argument
        """
        This method compute the similarity between interpretable pertubed samples and
        the original input (i.e a tf.ones(num_features)). The trick used for computation
        reason is to instead of using the original similarity kernel to pick random pertubed
        instances of interpretable sample in order to follow a certain probability rule, we
        let the pertub function create the pertub interpretable sample directly following
        this probability. Therefore, from the pertub function we can pick them with
        equal probability. See the `_kernel_shap_pertub_func` for more details.
        """

        similarities = tf.ones(len(interpret_samples), dtype=tf.float32)

        return similarities
