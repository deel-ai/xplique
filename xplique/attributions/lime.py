"""
Module related to LIME method
"""

import warnings

import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import cosine_similarity #pylint:  disable=E0611
from sklearn import linear_model
from skimage.segmentation import quickshift, felzenszwalb

from .base import BlackBoxExplainer, sanitize_input_output
from ..commons import Tasks
from ..types import Callable, Union, Optional, Any, OperatorSignature

class Lime(BlackBoxExplainer):
    """
    Used to compute the LIME method.

    Ref. Ribeiro & al., "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
    https://arxiv.org/abs/1602.04938

    Notes
    -----

    Note that the quality of an explanation relies strongly on your choice of the interpretable
    model, the similarity kernel and the map function mapping a sample into an interpretable space.
    The similarity kernel will define how close the perturbed samples are from the original sample
    you want to explain.
    For instance, if you have large images (e.g 299x299x3) a similarity kernel with the argument
    kernel width of 1 will compute similarities really close to 0. Consequently the interpretable
    model will not train. In order to make it work you have to use (for example) a larger kernel
    width.
    Moreover, depending on the similarities vector some interpretable model will fit
    better than other (e.g Ridge on large colored image might perform better than Lasso).
    Finally, your map function will defines how many features your linear model has to learn.
    Basically, an identity mapping (map each pixel as a single feature), on large image means there
    is as many features as there are pixels (e.g 299x299x3->89401 features) which can lead to poor
    explanations.

    N.B: This module was built to be deployed on GPU to be fully efficient. Considering the number
    of samples and number of inputs you want to process it might even be necessary.

    Parameters
    ----------
    model
        The model from which we want to obtain explanations
    batch_size
        Number of perturbed samples to process at once, mandatory when nb_samples is huge.
        Notice, it is different compare to WhiteBox explainers which batch the inputs.
        Here inputs are process one by one.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    interpretable_model
        Model object to train interpretable model.
        See the documentation for more information.
    similarity_kernel
        Function which considering an input, perturbed instances of these input and
        the interpretable version of those perturbed samples compute the similarities between
        the input and the perturbed samples.
        See the documentation for more information.
    pertub_func
        Function which generate perturbed interpretable samples in the interpretation space from
        the number of interpretable features (e.g nb of super pixel) and the number of perturbed
        samples you want per original input.
        See the documentation for more information.
    ref_value
        It defines reference value which replaces each feature when the corresponding
        interpretable feature is set to 0.
        It should be provided as: a ndarray of shape (1) if there is no channels in your input
        and (C,) otherwise

        The default ref value is set to (0.5,0.5,0.5) for inputs with 3 channels (corresponding
        to a grey pixel when inputs are normalized by 255) and to 0 otherwise.
    map_to_interpret_space
        Function which group features of an input corresponding to the same interpretable
        feature (e.g super-pixel).
        It allows to transpose from (resp. to) the original input space to (resp. from)
        the interpretable space.
        See the documentation for more information.
    nb_samples
        The number of perturbed samples you want to generate for each input sample.
        Default to 150.
    prob
        The probability argument for the default pertub function.
    distance_mode
        The distance mode used in the default similarity kernel, you can choose either
        "euclidean" or "cosine" (will compute cosine similarity).
        Default value set to "euclidean".
    kernel_width
        Width of your kernel. It is important to make it evolving depending on your inputs size
        otherwise you will get all similarity close to 0 leading to poor performance or NaN
        values.
        Default to 45 (i.e adapted for RGB images).
    """
    def __init__(
        self,
        model: Callable,
        batch_size: Optional[int] = None,
        operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
        interpretable_model: Any = linear_model.Ridge(alpha=2),
        similarity_kernel: Optional[Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]] = None,
        pertub_func: Optional[Callable[[Union[int, tf.Tensor],int], tf.Tensor]] = None,
        map_to_interpret_space: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        ref_value: Optional[np.ndarray] = None,
        nb_samples: int = 150,
        distance_mode: str = "euclidean",
        kernel_width: float = 45.0,
        prob: float = 0.5
        ): # pylint: disable=R0913

        if not all(hasattr(interpretable_model, attr) for attr in ['fit', 'predict']):
            raise ValueError(
                "The interpretable model is invalid. It should have a fit and a predict"
                " method. It should also have a coef_ attribute (the interpretable "
                "explanation) once fit is called."
                )

        if similarity_kernel is None:
            similarity_kernel = Lime._get_exp_kernel_func(distance_mode, kernel_width)

        if pertub_func is None:
            pertub_func = Lime._get_default_pertub_function(prob)

        if (nb_samples>=500) and (batch_size is None):
            warnings.warn(
                "You set a number of perturbed samples per input >= 500 and "
                "batch_size is set to None"
                "This mean that you will ask your model to handle more than 500"
                " perturbed samples per input at once."
                "This can lead to OOM issue. To avoid it you can set the"
                " batch_size argument."
            )

        super().__init__(model, batch_size, operator)

        self.map_to_interpret_space = map_to_interpret_space
        self.interpretable_model = interpretable_model
        self.similarity_kernel = similarity_kernel
        self.pertub_func = pertub_func
        self.ref_value = ref_value
        self.nb_samples = nb_samples

        self.batch_size = self.batch_size or self.nb_samples

    def _set_shape_dependant_parameters(self,
                                        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]):
        """
        Set default values for parameters dependant on the data type, thus the inputs shape.
        I.e. `ref_value` and `map_to_interpret_space`.

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, H, W, C).
            More information in the documentation.
        """

        if self.ref_value is not None:
            if len(inputs.shape) == 4:
                assert(self.ref_value.shape[0] == inputs.shape[-1]),\
                    "The dimension of ref_values must match inputs (C, )"
            self.ref_value = tf.cast(self.ref_value, tf.float32)
        else:
            if len(inputs.shape) in [2, 3]:  # Tabular data or time series
                self.ref_value = tf.zeros(1)
            elif len(inputs.shape) == 4:  # Image
                if inputs.shape[-1] == 3:  # RGB image
                    # grey pixel
                    self.ref_value = tf.fill(inputs.shape[-1], 0.5)
                elif inputs.shape[-1] == 1:  # Black and white image
                    self.ref_value = tf.zeros(inputs.shape[-1])

        if self.map_to_interpret_space is None:
            if len(inputs.shape) == 2:  # Tabular data
                self.map_to_interpret_space = Lime._default_tab_map_to_interpret_space
            elif len(inputs.shape) == 3:  # Time series
                self.map_to_interpret_space = Lime._default_time_series_map_to_interpret_space
            elif len(inputs.shape) == 4:  # Image
                if inputs.shape[-1] == 3:  # RGB image
                    self.map_to_interpret_space = Lime._default_image_map_to_interpret_space
                elif inputs.shape[-1] == 1:  # Black and white image
                    self.map_to_interpret_space = Lime._default_2dimage_map_to_interpret_space

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        This method attributes the output of the model with given targets
        to the inputs of the model using the approach described above,
        training an interpretable model and returning a representation of the
        interpretable model.

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, H, W, C).
            More information in the documentation.
        targets
            Tensor or Array. One-hot encoding of the model's output from which an explanation
            is desired. One encoding per input and only one output at a time. Therefore,
            the expected shape is (N, output_size).
            More information in the documentation.

        Returns
        -------
        explanations
            Interpretable coefficients, same shape as the inputs, except for the channels.
            Coefficients of the interpretable model. Those coefficients having the size of the
            interpretable space will be given the same value to coefficient which were grouped
            together (e.g belonging to the same super-pixel).
        """
        self._set_shape_dependant_parameters(inputs)
        explanations = []

        for inp, target in tf.data.Dataset.from_tensor_slices(
            (inputs, targets)
        ):
            # get the mapping of the current input
            mapping = self.map_to_interpret_space(inp)
            # get the number of interpretable feature
            num_features = tf.reduce_max(mapping) + tf.ones(1, dtype=tf.int32)
            if tf.greater(num_features, 10000):
                warnings.warn(
                    "The current input got a number of interpretable features > 10000. "
                    "This can be very slow or lead to OOM issues when fitting the interpretable "
                    "model. You should consider using a map function which select less features."
                )

            # get perturbed interpretable samples of the input
            interpret_samples = self.pertub_func(num_features, self.nb_samples)

            # get the perturbed targets value and the similarities value
            perturbed_targets = []
            similarities = []
            for int_samples in tf.data.Dataset.from_tensor_slices(
                interpret_samples
            ).batch(self.batch_size):

                masks = Lime._get_masks(int_samples, mapping)
                perturbed_samples = Lime._apply_masks(inp, masks, self.ref_value)

                augmented_target = tf.expand_dims(target, axis=0)
                augmented_target = tf.repeat(augmented_target, len(perturbed_samples), axis=0)

                batch_perturbed_targets = self.inference_function(self.model,
                                                                  perturbed_samples,
                                                                  augmented_target)

                perturbed_targets.append(batch_perturbed_targets)

                batch_similarities = self.similarity_kernel(inp, int_samples, perturbed_samples)
                similarities.append(batch_similarities)

            perturbed_targets = tf.concat(perturbed_targets, axis=0)
            similarities = tf.concat(similarities, axis=0)

            # train the interpretable model
            explain_model = self.interpretable_model

            explain_model.fit(
                interpret_samples.numpy(),
                perturbed_targets.numpy(),
                sample_weight=similarities.numpy()
            )

            explanation = explain_model.coef_
            # cast the interpretable explanation
            explanation = tf.cast(explanation, dtype=tf.float32)

            # broadcast explanations to match the original inputs shapes
            # except for channels
            explanation = Lime._broadcast_explanation(explanation, mapping)

            explanations.append(explanation)

        explanations = tf.stack(explanations, axis=0)

        if len(explanations.shape) == 3 and len(inputs.shape) == 4:
            explanations = tf.expand_dims(explanations, axis=-1)

        return explanations

    @staticmethod
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.int32)
        )
    )
    def _get_masks(interpret_samples: tf.Tensor, mapping: tf.Tensor) -> tf.Tensor:
        """
        This method translate the generated samples in the interpretable space of an input into
        masks to apply to the original input to obtain samples in the original input space.

        Parameters
        ----------
        interpret_samples
            Tensor of shape (nb_samples, num_features)
            Intrepretable samples of an input, with:
                nb_samples number of samples
                num_features the dimension of the interpretable space.
        mapping
            Tensor
            The mapping of the original input from which we drawn interpretable samples.
            Its size is equal to width (and height) of the original input

        Returns
        -------
        masks
            The masks corresponding to each interpretable samples
        """
        tf_masks = tf.gather(interpret_samples, indices=mapping, axis=1)
        return tf_masks

    @staticmethod
    @tf.function
    def _apply_masks(original_input: tf.Tensor,
                     sample_masks: tf.Tensor,
                     ref_value: tf.Tensor) -> tf.Tensor:
        """
        This method apply masks obtained from the perturbed interpretable samples to the
        original input (i.e we get perturbed samples in the original space).

        Parameters
        ----------
        original_input
            The input we want to explain
        sample_masks
            The masks we obtained from the perturbed instances in the interpretable space
        ref_value
            The reference value which replaces each feature when the corresponding
            interpretable feature is set to 0

        Returns
        -------
        perturbed_samples
            The perturbed samples corresponding to the masks applied to the original input
        """
        pert_samples = tf.expand_dims(original_input, axis=0)
        pert_samples = tf.repeat(pert_samples, repeats=sample_masks.shape[0], axis=0)

        # if there is channels we need to expand masks dimension
        if len(original_input.shape)==3:

            sample_masks = tf.expand_dims(sample_masks, axis=-1)
            sample_masks = tf.repeat(sample_masks, repeats=original_input.shape[-1], axis=-1)

            pert_samples = pert_samples * tf.cast(sample_masks, tf.float32)
            ref_val = tf.reshape(ref_value, (1,1,1,original_input.shape[-1]))

        else:

            pert_samples = pert_samples * tf.cast(sample_masks, tf.float32)
            ref_val = tf.reshape(ref_value, (1, *(1,) * len(original_input.shape)))

        pert_samples += (tf.ones((sample_masks.shape)) - tf.cast(sample_masks, tf.float32))*ref_val

        return pert_samples

    @staticmethod
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.int32)
        )
    )
    def _broadcast_explanation(explanation: tf.Tensor, mapping: tf.Tensor) -> tf.Tensor:
        """
        This method allows to broadcast explanations from the interpretable space to the
        corresponding super pixels

        Parameters
        ----------
        explanation
            Explanation value for each super pixel
        mapping
            The mapping of the original input from which we drawn interpretable samples
            (i.e features index).

        Returns
        -------
        broadcast_explanation
            The explanation of the current input considered
        """

        broadcast_explanation = tf.gather(explanation, indices=mapping, axis=0)
        return broadcast_explanation

    @staticmethod
    def _get_default_pertub_function(
        prob: float = 0.5
        ) -> Callable[[Union[int, tf.Tensor],int], tf.Tensor]:
        """
        This method allows you to get a pertub function with the corresponding prob
        argument.
        """

        prob = tf.cast(prob, dtype=tf.float32)
        @tf.function
        def _default_pertub_function(num_features: Union[int, tf.Tensor],
                                     nb_samples: int) -> tf.Tensor:
            """
            This method generate nb_samples tensor belonging to {0,1}^num_features.
            The prob argument is the probability to have a 1.

            Parameters
            ----------
            num_features
                The number of interpretable features (e.g super pixel).
            nb_samples
                The number of perturbed interpretable samples we want
            prob:
                It defines the probability to draw a 1

            Returns
            -------
            interpretable_perturbed_samples
                Tensor of shape (nb_samples, num_features)
            """

            probs = tf.ones(num_features, tf.float32) * tf.cast(prob, tf.float32)
            uniform_sampling = tf.random.uniform(shape=[nb_samples, tf.squeeze(num_features)],
                                                dtype=tf.float32,
                                                minval=0,
                                                maxval=1)
            sample = tf.greater(probs, uniform_sampling)
            sample = tf.cast(sample, dtype=tf.int32)
            return sample

        return _default_pertub_function

    @staticmethod
    def _get_exp_kernel_func(
        distance_mode: str = "euclidean", kernel_width: float = 1.0
    ) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        This method allow to get the function which compute:
            exp(-D(original_input,perturbed_sample)^2/kernel_width^2)
        Where D is the distance defined by distance mode.

        Parameters
        ----------
        distance_mode
            Can be either euclidian or cosine
        kernel_width
            The size of the kernel

        Returns
        -------
        similarity_kernel
            This callable should return distances between inputs and its perturbed samples
            (either in original space or in the interpretable space).
        """
        kernel_width = tf.cast(kernel_width,dtype=tf.float32)

        if distance_mode=="euclidean":
            @tf.function(
                input_signature = (
                    tf.TensorSpec(shape=None, dtype=tf.float32),
                    tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                    tf.TensorSpec(shape=None, dtype=tf.float32)
                )
            )
            def _euclidean_similarity_kernel(
                original_input,
                interp_samples,
                perturbed_samples
            ) -> tf.Tensor:
            # pylint: disable=unused-argument

                augmented_input = tf.expand_dims(original_input, axis=0)
                augmented_input = tf.repeat(augmented_input, repeats=len(interp_samples), axis=0)

                flatten_inputs = tf.reshape(augmented_input, [len(interp_samples),-1])
                flatten_samples = tf.reshape(perturbed_samples, [len(interp_samples),-1])

                distances = tf.norm(flatten_inputs - flatten_samples, ord='euclidean', axis=1)

                similarities = tf.exp(-1.0 * (distances**2) / (kernel_width**2))

                return similarities

            return _euclidean_similarity_kernel

        if distance_mode=="cosine":
            @tf.function(
                input_signature = (
                    tf.TensorSpec(shape=None, dtype=tf.float32),
                    tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                    tf.TensorSpec(shape=None, dtype=tf.float32)
                )
            )
            def _cosine_similarity_kernel(
                original_input,
                interp_samples,
                perturbed_samples
            ) -> tf.Tensor:
            # pylint: disable=unused-argument

                augmented_input = tf.expand_dims(original_input, axis=0)
                augmented_input = tf.repeat(augmented_input, repeats=len(interp_samples), axis=0)

                flatten_inputs = tf.reshape(augmented_input, [len(interp_samples),-1])
                flatten_samples = tf.reshape(perturbed_samples, [len(interp_samples),-1])

                distances = 1.0 - cosine_similarity(flatten_inputs, flatten_samples, axis=1)
                similarities = tf.exp(-1.0 * (distances**2) / (kernel_width**2))

                return similarities

            return _cosine_similarity_kernel

        raise ValueError("distance_mode must be either cosine or euclidean.")

    @staticmethod
    def _default_image_map_to_interpret_space(inp: tf.Tensor) -> tf.Tensor:
        """
        This method compute the quickshift segmentation.

        Parameters
        ----------
        inputs
            A single Input sample

        Returns
        -------
        mappings
            Mappings which map each pixel to the corresponding segment
        """
        mapping = quickshift(inp.numpy().astype('double'), ratio=0.5, kernel_size=2)
        mapping = tf.cast(mapping, tf.int32)

        return mapping

    @staticmethod
    def _default_2dimage_map_to_interpret_space(inp: tf.Tensor) -> tf.Tensor:
        """
        This method compute the felzenszwalb segmentation.

        Parameters
        ----------
        inputs
            A single Input sample

        Returns
        -------
        mappings
            Mappings which map each pixel to the corresponding segment
        """
        mapping = felzenszwalb(inp.numpy().astype('double'))
        mapping = tf.cast(mapping, tf.int32)

        return mapping

    @staticmethod
    def _default_tab_map_to_interpret_space(inp: tf.Tensor) -> tf.Tensor:
        """
        This method compute a similarity mapping i.e each features is independent.

        Parameters
        ----------
        inp
            A single Input sample

        Returns
        -------
        mappings
            Mappings which map each pixel to the corresponding segment
        """
        mapping = tf.range(len(inp))

        return mapping

    @staticmethod
    def _default_time_series_map_to_interpret_space(inp: tf.Tensor) -> tf.Tensor:
        """
        This method compute a similarity mapping i.e each features is independent.

        Parameters
        ----------
        inp
            A single Input sample

        Returns
        -------
        mappings
            Mappings which map each pixel to the corresponding segment
        """
        mapping = tf.range(inp.shape[0] * inp.shape[1])
        mapping = tf.reshape(mapping, (inp.shape[0], inp.shape[1]))
        return mapping
