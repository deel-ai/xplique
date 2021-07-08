"""
Module related to LIME method
"""

import warnings

import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import cosine_similarity #pylint:  disable=E0611
from sklearn import linear_model
from skimage.segmentation import quickshift

from .base import BlackBoxExplainer, sanitize_input_output
from ..commons import batch_predictions_one_hot
from ..types import Callable, Union, Optional, Any

class Lime(BlackBoxExplainer):
    """
    Used to compute the LIME method.

    Ref. Ribeiro & al., "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
    https://arxiv.org/abs/1602.04938

    Note that the quality of an explanation relies strongly on your choice of the interpretable
    model, the similarity kernel and the map function mapping a sample into an interpretable space.
    The similarity kernel will define how close the pertubed samples are from the original sample
    you want to explain.
    For instance, if you have large images (e.g 299x299x3) the default similarity kernel with the
    default kernel width will compute similarities really close to 0 consequently the interpretable
    model will not train. In order to makes it work you have to use (for example) a larger kernel
    width.
    Moreover, depending on the similarities vector you obtain some interpretable model will fit
    better than other (e.g Ridge on large colored image might perform better than Lasso).
    Finally, your map function will defines how many features your linear model has to learn.
    Basically, the default mapping is an identity mapping, so for large image it means there is
    as many features as there are pixels (e.g 299x299x3->89401 features) which can lead to poor
    explanations.

    N.B: This module was built to be deployed on GPU to be fully efficient. Considering the number
    of samples and number of inputs you want to process it might even be necessary.
    """
    def __init__(
        self,
        model: Callable,
        batch_size: int = 1,
        interpretable_model: Any = linear_model.Ridge(alpha=2),
        similarity_kernel: Optional[Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]] = None,
        pertub_func: Optional[Callable[[Union[int, tf.Tensor],int], tf.Tensor]] = None,
        map_to_interpret_space: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        ref_values: Optional[np.ndarray] = None,
        nb_samples: int = 150,
        batch_pertubed_samples: Optional[int] = None,
        distance_mode: str = "euclidean",
        kernel_width: float = 45.0,
        prob: float = 0.5
        ): # pylint: disable=R0913
        """
        Parameters
        ----------
        model
            Model that you want to explain.

        batch_size
            Number of samples to explain at once, if None compute all at once.

        interpretable_model
            Model object to train interpretable model.
            A Model object provides a `fit` method to train the model,
            containing three array:

            - interpretable_inputs: ndarray (2D nb_samples x num_interp_features),
            - expected_outputs: ndarray (1D nb_samples),
            - weights: ndarray (1D nb_samples)

            The model object should also provide a `predict` method and have a coef_ attributes
            (the interpretable explanation) at least once fit is called.
            As interpretable model you can use linear models from scikit-learn.
            Note that here nb_samples doesn't indicates the length of inputs but the number of
            pertubed samples we want to generate for each input.

        similarity_kernel
            Function which considering an input (duplicated to match the number of pertubed
            instances created), pertubed instances of thoses samples and the interpretable
            version of those pertubed samples compute the similarities.
            The similarities can be computed in the original input space or in the interpretable
            space.
            You can provide a custom function. Note that to use a custom function, you have to
            follow the following scheme:

            def custom_similarity(
                duplicate_input , pertubed_samples, interpret_samples
            ) -> tf.tensor (shape=(nb_samples,), dtype = tf.float32):
                ** some tf actions **
                return similarities

            where:
                duplicate_input, pertubed_samples are tf.tensor (nb_samples, W, H, C)
                interpret_samples is a tf.tensor (nb_samples, num_interp_features)
            If it is possible you can add the tf.function decorator.

            The default similarity kernel use the euclidian distance between the original input and
            sample in the input space.

        pertub_function
            Function which generate pertubed interpretable samples in the interpretation space from
            the number of interpretable features (e.g nb of super pixel) and the number of pertubed
            samples you want per original sample.
            The generated interp_samples belong to {0,1}^num_features. Where 1 indicates that we
            keep the corresponding feature (e.g super pixel) in the mapping.
            To use your own custom pertub function you should use the following scheme:

            @tf.function
            def custom_pertub_function(num_features, nb_samples) ->
            tf.tensor (shape=(nb_samples, num_interp_features), dtype=tf.int32):
                ** some tf actions**
                return pertubed_sample

            The default pertub function provided keep a feature (e.g super pixel) with a
            probability 0.5.
            If you want to change it, defines your own prob value when initiating the explainer.

        ref_values
            It defines reference value which replaces each feature when the corresponding
            interpretable feature is set to 0.
            It should be provided as: a ndarray (C,)

            The default ref value is set to (0.5,0.5,0.5) for inputs with 3 channels (corresponding
            to a grey pixel when inputs are normalized by 255) and to 0 otherwise.

        map_to_interpret_space
            Function which group an input features which correspond to the same interpretable
            feature (e.g super-pixel).
            It allows to transpose from (resp. to) the original input space to (resp. from)
            the interpretable space.
            The default mapping is the quickshift segmentation algorithm.

            To use your own custom map function you should use the following scheme:

            def custom_map_to_interpret_space(inputs: tf.tensor (N, W, H, C)) ->
            tf.tensor (N, W, H):
                **some grouping techniques**
                return mappings

            For instance you can use the scikit-image (as we did for the quickshift algorithm)
            library to defines super pixels on your images.

        nb_samples
            The number of pertubed samples you want to generate for each input sample.
            Default to 150.

        batch_pertubed_samples
            The batch size to predict the pertubed samples targets value.
            Default to None (i.e predictions of all the pertubed samples one shot).

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
            Default to 1.
        """

        if similarity_kernel is None:
            similarity_kernel = Lime._get_exp_kernel_func(distance_mode, kernel_width)

        if pertub_func is None:
            pertub_func = Lime._get_default_pertub_function(prob)

        if map_to_interpret_space is None:
            map_to_interpret_space = Lime._default_map_to_interpret_space

        if (nb_samples>=500) and (batch_pertubed_samples is None):
            warnings.warn(
                "You set a number of pertubed samples per input >= 500 and "
                "batch_pertubed_samples is set to None"
                "This mean that you will ask your model to make more than 500 predictions"
                " one shot."
                "This can lead to OOM issue. To avoid it you can set the"
                " batch_pertubed_samples."
            )

        if not all(hasattr(interpretable_model, attr) for attr in ['fit', 'predict']):
            raise ValueError(
                "The interpretable model is invalid. It should have a fit and a predict"
                " method. It should also have a coef_ attribute (the interpretable "
                "explanation) once fit is called."
                )

        super().__init__(model, batch_size)

        self.map_to_interpret_space = map_to_interpret_space
        self.interpretable_model = interpretable_model
        self.similarity_kernel = similarity_kernel
        self.pertub_func = pertub_func
        self.ref_values = ref_values
        self.nb_samples = nb_samples
        self.batch_pertubed_samples = batch_pertubed_samples

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> np.ndarray:
        """
        This method attributes the output of the model with given targets
        to the inputs of the model using the approach described above,
        training an interpretable model and returning a representation of the
        interpretable model.

        Parameters
        ----------
        inputs
            Tensor or numpy array of shape (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.

        targets
            Tensor or numpy array of shape (N, L)
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each input.s.

        Returns
        -------
        explanations
            Numpy array of shape: (N, W, H)
            Coefficients of the interpretable model. Those coefficients having the size of the
            interpretable space will be given the same value to coefficient which were grouped
            together (e.g belonging to the same super-pixel).
        """

        if self.ref_values is None:
            if inputs.shape[-1] == 3:
                # grey pixel
                ref_values = tf.ones(inputs.shape[-1])*0.5
            else:
                ref_values = tf.zeros(inputs.shape[-1])
        else:
            assert(
                self.ref_values.shape[0] == inputs.shape[-1]
            ),"The dimension of ref_values must match inputs (C, )"
            ref_values = tf.cast(self.ref_values, tf.float32)

        # use the map function to get a mapping per input to the interpretable space
        mappings = self.map_to_interpret_space(inputs)
        batch_size = self.batch_size or len(inputs)

        return Lime._compute(self.model,
                            batch_size,
                            inputs,
                            targets,
                            self.interpretable_model,
                            self.similarity_kernel,
                            self.pertub_func,
                            ref_values,
                            mappings,
                            self.nb_samples,
                            self.batch_pertubed_samples
                            )

    @staticmethod
    def _compute(model: Callable,
                batch_size: int,
                inputs: tf.Tensor,
                targets: tf.Tensor,
                interpretable_model: Callable,
                similarity_kernel: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
                pertub_func: Callable[[Union[int, tf.Tensor],int], tf.Tensor],
                ref_values: tf.Tensor,
                mappings: tf.Tensor,
                nb_samples: int,
                batch_pertubed_samples: int
                ) -> tf.Tensor:
                # pylint: disable=R0913
        """
        This method attributes the output of the model with given targets
        to the inputs of the model using the approach described above,
        training an interpretable model and returning a representation of the
        interpretable model.

        Parameters
        ----------
        model
            Model to explain.

        inputs
            Tensor of shape (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.

        targets
            Tensor of shape (N, L)
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

        interpretable_model
            Model object to train interpretable model.
            A Model object provides a `fit` method to train the model,
            containing three array:

            - interpretable_inputs: ndarray (2D nb_samples x num_interp_features),
            - expected_outputs: ndarray (1D nb_samples),
            - weights: ndarray (1D nb_samples)

            The model object should also provide a `predict` method and have a coef_ attributes
            (the interpretable explanation).
            As interpretable model you can use linear models from scikit-learn.
            Note that here nb_samples doesn't indicates the length of inputs but the number of
            pertubed samples we want to generate for each input.

        similarity_kernel
            Function which considering an input (duplicated to match the number of pertubed
            instances created), pertubed instances of thoses samples and the interpretable
            version of those pertubed samples compute the similarities.
            The similarities can be computed in the original input space or in the interpretable
            space.
            You can provide a custom function. Note that to use a custom function, you have to
            follow the following scheme:

            def custom_similarity(
                inputs , pertubed_samples, interpret_samples
            ) -> tf.tensor (shape=(batch_size,), dtype = tf.float32):
                ** some tf actions **
                return similarities

            where:
                inputs, pertubed_samples are tf.tensor (batch_size, W, H, C)
                interpret_samples is a tf.tensor (batch_size, num_interp_features)
            If it is possible you can add the tf.function decorator.

            The default similarity kernel use the euclidian distance between the original input and
            sample in the input space.

        pertub_function
            Function which generate pertubed interpretable samples in the interpretation space from
            the number of interpretable features (e.g nb of super pixel) and the number of pertubed
            samples you want per original sample.
            The generated interp_samples belong to {0,1}^num_features. Where 1 indicates that we
            keep the corresponding feature (e.g super pixel) in the mapping.
            The pertub function should use the following scheme:

            @tf.function
            def custom_pertub_function(num_features, nb_samples) ->
            tf.tensor (shape=(nb_samples, num_interp_features), dtype=tf.int32):
                ** some tf actions**
                return pertubed_sample

        ref_values
            It defines reference value which replaces each feature when the corresponding
            interpretable feature is set to 0.
            It should be provided as: a tf.Tensor (C,)

        mappings
            Tensor of shape (N, W, H)
            It is grouping features which correspond to the same interpretable feature (super-pixel)
            It allows to transpose from (resp. to) the original input space to (resp. from) the
            interpretable space.

            Values accross all tensors should be integers in the range 0 to num_interp_features - 1

        nb_samples
            The number of pertubed samples you want to generate for each input sample.

        batch_pertubed_samples
            The batch size to predict the pertubed samples labels value.

        Returns
        -------
        explanations
            Tensor of shape: (N, W, H)
            Coefficients of the interpretable model. Those coefficients having the size of the
            interpretable space will be given the same value to coefficient which were grouped
            together (e.g belonging to the same super-pixel).
        """
        explanations = []

        # get the number of interpretable features for each inputs
        num_features = tf.reduce_max(tf.reduce_max(mappings, axis=1),axis=1)
        num_features += tf.ones(len(mappings),dtype=tf.int32)

        if tf.greater(tf.cast(tf.reduce_max(num_features),tf.float32),1e4):
            warnings.warn(
                "One or several inputs got a number of interpretable features > 10000. "
                "This can be very slow or lead to OOM issues when fitting the interpretable"
                "model. You should consider using a map function which select less features."
            )

        # augment the target vector to match (N, nb_samples, L)
        augmented_targets = tf.expand_dims(targets, axis=1)
        augmented_targets = tf.repeat(augmented_targets, repeats=nb_samples, axis=1)

        # add a prefetch variable for numerous inputs
        nb_prefetch = 0
        if len(inputs)//batch_size > 2:
            nb_prefetch = 2

        # batch inputs, mappings, augmented targets and num_features
        for b_inp, b_targets, b_mappings, b_num_features in tf.data.Dataset.from_tensor_slices(
            (inputs, augmented_targets, mappings, num_features)
        ).batch(batch_size).prefetch(nb_prefetch):

            # get the pertubed samples (interpretable and in the original space)
            interpret_samples, pertubed_samples = tf.map_fn(
                fn= lambda inp: Lime._generate_sample(
                    inp[0],
                    pertub_func,
                    inp[1],
                    inp[2],
                    nb_samples,
                    ref_values
                ),
                elems=(b_inp, b_mappings, b_num_features),
                fn_output_signature=(
                    tf.RaggedTensorSpec(shape=[None,None],dtype=tf.int32),
                    tf.float32
                )
            )

            # get the targets of pertubed_samples
            pertubed_targets = tf.map_fn(
                fn= lambda inp: batch_predictions_one_hot(
                    model,
                    inp[0],
                    inp[1],
                    batch_pertubed_samples
                ),
                elems=(pertubed_samples, b_targets),
                fn_output_signature=tf.float32
            )

            # augment inputs to match the number of pertubed instances
            augmented_b_inp = tf.expand_dims(b_inp, axis = 1)
            augmented_b_inp = tf.repeat(augmented_b_inp, repeats=nb_samples, axis=1)

            # compute similarities
            similarities = tf.map_fn(
                fn= lambda inp: similarity_kernel(inp[0],
                                                  inp[1],
                                                  inp[2]
                ),
                elems=(augmented_b_inp, pertubed_samples, interpret_samples),
                fn_output_signature=tf.float32
            )

            # train the interpretable model
            for int_samples, pertubed_target, samples_weight in tf.data.Dataset.from_tensor_slices(
                    (interpret_samples,pertubed_targets,similarities)):

                explain_model = interpretable_model

                explain_model.fit(
                    int_samples.numpy(),
                    pertubed_target.numpy(),
                    sample_weight=samples_weight.numpy()
                )

                explanation = explain_model.coef_
                # add the interpretable explanation
                explanation = tf.cast(explanation, dtype=tf.float32)
                explanations.append(explanation)

        explanations = tf.ragged.stack(explanations, axis=0)
        # broadcast explanations to match the original inputs shapes
        complete_explanations = tf.map_fn(
            fn= lambda inp: Lime._broadcast_explanation(inp[0],inp[1]),
            elems=(explanations,mappings),
            fn_output_signature=tf.float32
        )

        return complete_explanations

    @staticmethod
    def _default_map_to_interpret_space(inputs: tf.Tensor) -> tf.Tensor:
        """
        This method compute the quickshift segmentation.

        Parameters
        ----------
        inputs
            Tensor of shape (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.

        Returns
        -------
        mappings
            Tensor of shape (N, W, H)
            Mappings which map each pixel to the corresponding segment
        """
        mappings = []
        for inp in inputs:
            mapping = quickshift(inp.numpy().astype('double'), ratio=0.5, kernel_size=2)
            mapping = tf.cast(mapping, tf.int32)
            mappings.append(mapping)
        mappings = tf.stack(mappings, axis=0)
        return mappings

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
                The number of pertubed interpretable samples we want
            prob:
                It defines the probability to draw a 1

            Returns
            -------
            interpretable_pertubed_samples
                Tensor of shape (nb_samples, num_features)
            """
            probs = tf.ones(tf.expand_dims(num_features,axis=0),tf.float32)*tf.cast(prob,tf.float32)
            uniform_sampling = tf.random.uniform(shape=[nb_samples,num_features],
                                                dtype=tf.float32,
                                                minval=0,
                                                maxval=1)
            sample = tf.greater(probs, uniform_sampling)
            sample = tf.cast(sample, dtype=tf.int32)
            return sample

        return _default_pertub_function

    @staticmethod
    @tf.function
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
            Tensor of shape (W, H)
            The mapping of the original input from which we drawn interpretable samples.
            Its size is equal to width and height of the original input

        Returns
        -------
        masks
            Tensor of shape (nb_samples, W, H)
            The masks corresponding to each interpretable samples
        """
        tf_masks = tf.gather(interpret_samples,indices=mapping,axis=1)
        return tf_masks

    @staticmethod
    @tf.function
    def _apply_masks(original_input: tf.Tensor,
                     sample_masks: tf.Tensor,
                     ref_value: tf.Tensor) -> tf.Tensor:
        """
        This method apply masks obtained from the pertubed interpretable samples to the
        original input (i.e we get pertubed samples in the original space).

        Parameters
        ----------
        original_input
            Tensor of shape (W, H, C)
            The input we want to explain
        sample_masks
            Tensor of shape (nb_samples, W, H)
            The masks we obtained from the pertubed instances in the interpretable space
        ref_value
            Tensor of shape (C,)
            The reference value which replaces each feature when the corresponding
            interpretable feature is set to 0

        Returns
        -------
        pertubed_samples
            Tensor of shape (nb_samples, W, H, C)
            The pertubed samples corresponding to the masks applied to the original input
        """
        pert_samples = tf.expand_dims(original_input, axis=0)
        pert_samples = tf.repeat(pert_samples, repeats=len(sample_masks), axis=0)

        sample_masks = tf.expand_dims(sample_masks, axis=-1)
        sample_masks = tf.repeat(sample_masks, repeats=original_input.shape[-1], axis=-1)

        pert_samples = pert_samples * tf.cast(sample_masks, tf.float32)
        ref_val = tf.reshape(ref_value, (1,1,1,original_input.shape[-1]))
        pert_samples += (tf.ones((sample_masks.shape)) - tf.cast(sample_masks, tf.float32))*ref_val

        return pert_samples


    @staticmethod
    @tf.function
    def _generate_sample(original_input: tf.Tensor,
                        pertub_func: Callable[[Union[int, tf.Tensor],int], tf.Tensor],
                        mapping: tf.Tensor,
                        num_features: Union[int, tf.Tensor],
                        nb_samples: int,
                        ref_value: tf.Tensor) -> tf.Tensor:
        """
        This method generate nb_samples pertubed instance of the current input in the
        interpretable space.
        Then it computes the pertubed instances into the input space.

        Parameters
        ----------
        original_input
            Tensor of shape (W, H, C)
            The input we want to explain
        pertub_func
            Function which generate a pertubed sample in the interpretation space from an
            interpretable input.
        mapping
            Tensor of shape (W, H)
            The mapping of the original input from which we drawn interpretable samples.
            Its size is equal to width and height of the current input
        num_features
            The dimension size of the interpretable space
        nb_samples
            The number of pertubed instances we want of current input
        ref_value
            The reference value which replaces each feature when the corresponding
            interpretable feature is set to 0

        Returns
        -------
        interpret_samples
            Ragged Tensor of shape (nb_samples, num_features)
            Intrepretable samples of the original input.
        pertubed_samples
            Tensor of shape (nb_samples, W, H, C)
            The samples corresponding to the masks applied to the original input
        """

        interpret_samples = pertub_func(num_features, nb_samples)

        masks = Lime._get_masks(interpret_samples, mapping)
        pertubed_samples = Lime._apply_masks(original_input, masks, ref_value)

        # since all inputs might have a nb of features different, we need to
        # specify that interpret samples is a Ragged Tensor, then it allows
        # to map this function even if from one batch to another its length
        # vary
        interpret_samples = tf.RaggedTensor.from_tensor(interpret_samples)

        return interpret_samples, pertubed_samples

    @staticmethod
    def _get_exp_kernel_func(
        distance_mode: str = "euclidean", kernel_width: float = 1.0
    ) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        This method allow to get the function which compute:
            exp(-D(original_input,pertubed_sample)^2/kernel_width^2)
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
            This callable should return distances between inputs and its pertubed samples
            (either in original space or in the interpretable space).

        """
        kernel_width = tf.cast(kernel_width,dtype=tf.float32)

        if distance_mode=="euclidean":
            @tf.function(
                input_signature=(
                    tf.TensorSpec(shape=[None,None,None,None], dtype=tf.float32),
                    tf.TensorSpec(shape=[None,None,None,None], dtype=tf.float32),
                    tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int32)
                )
            )
            def _euclidean_similarity_kernel(original_inputs: tf.Tensor,
                                             pertubed_samples: tf.Tensor,
                                             interpret_samples: tf.RaggedTensor
                                            ) -> tf.Tensor:
                                            # pylint: disable=unused-argument
                nb_samples = len(original_inputs)
                flatten_inputs = tf.reshape(original_inputs, [nb_samples,-1])
                flatten_samples = tf.reshape(pertubed_samples, [nb_samples,-1])

                distances = tf.norm(flatten_inputs - flatten_samples, ord='euclidean', axis=1)

                similarities = tf.exp(-1.0 * (distances**2) / (kernel_width**2))
                return similarities

            return _euclidean_similarity_kernel

        if distance_mode=="cosine":
            @tf.function(
                input_signature=(
                    tf.TensorSpec(shape=[None,None,None,None], dtype=tf.float32),
                    tf.TensorSpec(shape=[None,None,None,None], dtype=tf.float32),
                    tf.RaggedTensorSpec(shape=[None,None], dtype=tf.int32)
                )
            )
            def _cosine_similarity_kernel(original_inputs: tf.Tensor,
                                          pertubed_samples: tf.Tensor,
                                          interpret_samples: tf.RaggedTensor
                                         ) -> tf.Tensor:
                                         # pylint: disable=unused-argument

                nb_samples = len(original_inputs)
                flatten_inputs = tf.reshape(original_inputs, [nb_samples,-1])
                flatten_samples = tf.reshape(pertubed_samples, [nb_samples,-1])

                distances = 1.0 - cosine_similarity(flatten_inputs, flatten_samples, axis=1)
                similarities = tf.exp(-1.0 * (distances**2) / (kernel_width**2))
                return similarities

            return _cosine_similarity_kernel

        raise ValueError("distance_mode must be either cosine or euclidean.")

    @staticmethod
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None], dtype=tf.float32),
            tf.TensorSpec(shape=[None,None], dtype=tf.int32)
        )
    )
    def _broadcast_explanation(explanation: tf.Tensor, mapping: tf.Tensor) -> tf.Tensor:
        """
        This method allows to broadcast explanations from the interpretable space to the
        corresponding super pixels

        Parameters
        ----------
        explanation
            Tensor of shape (num_features)
            Explanation value for each super pixel
        mapping
            Tensor of shape (W, H)
            The mapping of the original input from which we drawn interpretable samples
            (i.e super-pixels positions).
            Its size is equal to width and height of the original input

        Returns
        -------
        broadcast_explanation
            Tensor of shape (W, H)
            The explanation of the current input considered

        """

        broadcast_explanation = tf.gather(explanation, indices=mapping, axis=0)
        return broadcast_explanation
