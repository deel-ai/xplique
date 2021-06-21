"""
Module related to Kernel SHAP method
"""

import tensorflow as tf

from sklearn import linear_model

from .lime import Lime

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
                 model,
                 batch_size: int = 1,
                 map_to_interpret_space = None,
                 nb_samples: int = 800,
                 batch_pertubed_samples = 64,
                 ref_values = None):
        """
        Parameters
        ----------
        model : tf.keras.Model
            Model that you want to explain.

        map_to_interpret_space : callable, optional
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

        nb_samples: int
            The number of pertubed samples you want to generate for each input sample.
            Default to 50.

        batch_pertubed_samples: int
            The batch size to predict the pertubed samples labels value.
            Default to None (i.e predictions of all the pertubed samples one shot).

        ref_values : ndarray
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
        _ , __, interpret_sample
    ):
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
            # weight should be theoretically infinite when
            # num_selected_features = 0 or num_features
            # enforcing that trained linear model must satisfy
            # end-point criteria. In practice, it is sufficient to
            # make this weight substantially larger so setting this
            # weight to 1000000 (all other weights are 1).
            return tf.constant(1000000.0, dtype=tf.float32)

        return tf.constant(1.0, dtype=tf.float32)

    @staticmethod
    @tf.function
    def _kernel_shap_pertub_func(num_features, nb_samples):
        """
        Perturbations are sampled by the following process:
         - Choose k (number of selected features), based on the distribution
                p(k) = (M - 1) / (k * (M - k))
            where M is the total number of features in the interpretable space
         - Randomly select a binary vector with k ones, each sample is equally
            likely. This is done by generating a random vector of normal
            values and thresholding based on the top k elements.
         Since there are M choose k vectors with k ones, this weighted sampling
         is equivalent to applying the Shapley kernel for the sample weight,
         defined as:
                k(M, k) = (M - 1) / (k * (M - k) * (M choose k))
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
    def _get_probs_nb_selected_feature(num_features):
        """
        Compute the distribution:
            p(k) = (num_features - 1) / (k * (num_features - k))
        """
        list_features_indexes = tf.range(1,num_features)
        denom = tf.multiply(list_features_indexes,(num_features - list_features_indexes))
        num = num_features - 1
        probs = tf.divide(num,denom)
        probs = tf.concat([[0.0],probs], 0)
        return tf.cast(probs, dtype=tf.float32)
