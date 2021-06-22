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
    Kernel SHAP is a method that uses the LIME framework to compute Shapley Values. Setting
    the loss function, weighting kernel and regularization terms appropriately in the LIME
    framework allows theoretically obtaining Shapley Values more efficiently than directly
    computing Shapley Values.

    More information regarding this method and proof of equivalence can be found in the
    original paper here:
    https://arxiv.org/abs/1705.07874
    """
    def __init__(self,
                 model: Callable,
                 batch_size: int = 1,
                 map_to_interpret_space: Optional[Callable] = None,
                 nb_samples: int = 800,
                 batch_pertubed_samples: Optional[int] = 64,
                 ref_values: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        model
            Model that you want to explain.

        map_to_interpret_space
            Function which group an input features which correspond to the same interpretable
            feature (e.g super-pixel).
            It allows to transpose from (resp. to) the original input space to (resp. from)
            the interpretable space.
            The default mapping is the identity mapping which is quickly a poor mapping.

            To use your own custom map function you should use the following scheme:

            def custom_map_to_interpret_space(inputs: tf.tensor (N, W, H, C)) ->
            tf.tensor (N, W, H):
                **some grouping techniques**
                return mappings

            For instance you can use the scikit-image library to defines super pixels on your
            images.

        nb_samples
            The number of pertubed samples you want to generate for each input sample.
            Default to 800.

        batch_pertubed_samples
            The batch size to predict the pertubed samples labels value.
            Default to 64.

        ref_values
            It defines reference value which replaces each feature when the corresponding
            interpretable feature is set to 0.
            It should be provided as: a ndarray (C,)

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
            ref_values = ref_values,
            map_to_interpret_space = map_to_interpret_space,
            nb_samples = nb_samples,
            batch_pertubed_samples = batch_pertubed_samples
            )

    # No need to redifine the explain method (herited from Lime)

    @staticmethod
    @tf.function
    def _kernel_shap_similarity_kernel(
        _ , __, interpret_sample: tf.Tensor
    ) -> tf.Tensor:
        """
        This method compute the similarity between an interpretable pertubed sample and
        the original input (i.e a tf.ones(num_features)).
        """

        num_selected_features = tf.cast(
            tf.reduce_sum(interpret_sample),
            dtype=tf.int32)
        num_features = len(interpret_sample)

        if (tf.equal(num_selected_features, tf.constant(0))
            or tf.equal(num_selected_features,num_features)):
            # Theoretically, in that case the weight should be
            # infinite. However, we will consider it is sufficient to
            # set this weight to 1000000 (all other weights are 1).
            return tf.constant(1000000.0, dtype=tf.float32)

        return tf.constant(1.0, dtype=tf.float32)

    @staticmethod
    @tf.function
    def _kernel_shap_pertub_func(num_features: Union[int, tf.Tensor],
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
         is equivalent to applying the Shapley kernel for the sample weight,
         defined as:
            k(nb_features, k) = (nb_features - 1)/(k*(nb_features - k)*(nb_features choose k))
        This trick is the one used in the Captum library: https://github.com/pytorch/captum
        """
        probs_nb_selected_feature = KernelShap._get_probs_nb_selected_feature(
            tf.cast(num_features,dtype=tf.int32))
        nb_selected_features = tf.random.categorical(tf.math.log([probs_nb_selected_feature]),
                                                     nb_samples,
                                                     dtype=tf.int32)
        nb_selected_features = tf.reshape(nb_selected_features,[nb_samples])
        interpret_samples = []

        for i in range(nb_samples):
            rand_vals = tf.random.normal([num_features])
            threshold = tf.math.top_k(rand_vals, k=(nb_selected_features[i]+1))
            threshold = tf.reduce_min(threshold.values)
            interpret_sample = tf.greater(rand_vals, threshold)
            interpret_sample = tf.cast(interpret_sample, dtype=tf.int32)
            interpret_samples.append(interpret_sample)
        interpret_samples = tf.stack(interpret_samples, axis=0)

        return interpret_samples

    @staticmethod
    @tf.function
    def _get_probs_nb_selected_feature(num_features: Union[int, tf.Tensor]) -> tf.Tensor:
        """
        Compute the distribution:
            p(k) = (nb_features - 1) / (k * (nb_features - k))
        """
        list_features_indexes = tf.range(1,num_features)
        denom = tf.multiply(list_features_indexes,(num_features - list_features_indexes))
        num = num_features - 1
        probs = tf.divide(num,denom)
        probs = tf.concat([[0.0],probs], 0)
        return tf.cast(probs, dtype=tf.float32)
