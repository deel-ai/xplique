"""
Framework-agnostic CRAFT implementation for holistic model explanations.
"""
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.exceptions import NotFittedError

from xplique.attributions.global_sensitivity_analysis.sobol_attribution_method import (
    SobolAttributionMethod,
)
from xplique.attributions.gradient_input import GradientInput
from xplique.commons.prediction_types import StructuredPrediction
from xplique.plots.image import _clip_percentile

from .craft import Factorization, Sensitivity
from .factorizer import SklearnNMFFactorizer
from .latent_extractor import EncodedData, LatentData, LatentExtractor


def show_ax(img, ax, **kwargs):
    """
    Display an image on a matplotlib axis with normalization.

    Converts channel-first images to channel-last format, normalizes pixel
    values to [0, 1] range, and displays without axis labels.

    Parameters
    ----------
    img
        Image array to display, either in channel-first (C, H, W) or
        channel-last (H, W, C) format
    ax
        Matplotlib axis object on which to display the image
    **kwargs
        Additional keyword arguments passed to ax.imshow()
    """
    img = np.array(img)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    ax.imshow(img, **kwargs)
    ax.axis("off")


class HolisticCraft:
    """
    Framework-agnostic CRAFT implementation for holistic model explanations.

    This base class provides concept-based explanations for various model types
    (object detection, classification, etc.) by extracting and analyzing intermediate
    activations using Non-negative Matrix Factorization (NMF). It supports both
    TensorFlow and PyTorch through framework-specific subclasses.

    The workflow involves:
    1. Extracting latent activations from a detection model
    2. Factorizing activations into interpretable concepts using NMF
    3. Computing concept importance using gradient-based attribution methods
    4. Visualizing concepts as spatial heatmaps overlaid on input images

    Parameters
    ----------
    latent_extractor
        Extractor that splits the model into encoder (input to activations) and
        decoder (activations to predictions) for concept extraction
    number_of_concepts
        Number of concepts to extract via NMF decomposition
    device
        Device specification for tensor operations (framework-specific)
    factorizer
        Optional factorizer instance implementing the ConceptFactorizer protocol.
        If None, creates a default SklearnNMFFactorizer with alpha_W=1e-2 and
        max_iter=200

    Attributes
    ----------
    latent_extractor
        The latent extractor instance
    number_of_concepts
        Number of concepts extracted
    batch_size
        Batch size inherited from latent_extractor
    factorization
        Factorization object containing NMF results, populated after fit()
    factorizer
        Factorizer instance used for concept extraction
    device
        Device for tensor operations
    cmaps
        List of colormaps for visualization
    """

    def __init__(
        self,
        latent_extractor: LatentExtractor,
        number_of_concepts: int = 20,
        device: str = None,
        factorizer: Optional[Any] = None
    ):
        self.latent_extractor = latent_extractor
        self.number_of_concepts = number_of_concepts
        self.batch_size = latent_extractor.batch_size
        self.factorization = None
        self.device = device

        # Use provided factorizer or create default NMF factorizer
        if factorizer is None:
            self.factorizer = SklearnNMFFactorizer(
                n_components=number_of_concepts,
                alpha_W=1e-2,
                max_iter=200
            )
        else:
            self.factorizer = factorizer

        # Setup visualization colormaps
        self.cmaps = [
            Sensitivity._get_alpha_cmap(cmap) for cmap in plt.get_cmap("tab10").colors
        ]

    def check_if_fitted(self):
        """Checks if the factorization model has been fitted to input data.

        Raises
        ------
        NotFittedError
            If the factorization model has not been fitted to input data.
        """
        if not self.factorizer.is_fitted:
            raise NotFittedError(
                "The factorization model has not been fitted to input data yet."
            )

    def fit(
        self, inputs, class_id: int = 0
    ):
        """
        Fit NMF to extract concepts from latent activations.

        Processes input images through the latent extractor to obtain intermediate
        activations, then applies Non-negative Matrix Factorization to discover
        interpretable concepts. The concepts are spatial patterns in activation
        space that recur across different images and locations.

        Activations are converted to NumPy immediately to minimize device memory usage,
        which is especially important for large datasets and GPU processing.

        Parameters
        ----------
        inputs
            Input images to extract concepts from, as framework tensors or arrays
        class_id
            Target class ID for object detection (used in factorization metadata)

        """
        # pass the data through the 1st part of the model
        latent_data_list = self.latent_extractor.input_to_latent_batched(inputs)

        # get the activations and concatenate as numpy arrays to minimize device memory usage
        # Converting to numpy immediately frees device memory after each batch
        activations = np.concatenate(
            [latent_data.get_activations(as_numpy=True) for latent_data in latent_data_list],
            axis=0
        )

        needs_reshape = len(activations.shape) > 2 # (N,H,W,C) or (N,Tokens,C)
        if needs_reshape:
            activations_original_shape = activations.shape[:-1]
            # Activations are already in numpy format, reshape for factorization
            activations = np.reshape(activations, (-1, activations.shape[-1]))

        # Check if factorizer requires positive activations
        if self.factorizer.requires_positive_activations and np.any(activations < 0):
            raise ValueError(
                "Factorizer requires non-negative activations but received negative values."
            )

        # Apply factorizer to the activations
        concept_bank_w, coeffs_u = self.factorizer.fit(activations)
        concept_bank_w = concept_bank_w.astype(np.float32)

        # Reshape coefficients back to spatial dimensions
        if needs_reshape:
            coeffs_u = coeffs_u.reshape(*activations_original_shape, -1)

        self.factorization = Factorization(
            class_id=class_id, reducer=self.factorizer, concept_bank_w=concept_bank_w,
            crops_u=None, coeffs_u=coeffs_u
        )

    def transform(self, inputs=None, resize=None) -> np.ndarray:
        """Transform inputs to concept coefficients.

        This method encodes the inputs and returns only the concept coefficients
        as a concatenated numpy array, discarding the latent data.

        If inputs is None, returns the stored coefficients from fit() if available
        (useful for ConvexNMF which can only encode training data).

        Parameters
        ----------
        inputs
            Input images to transform. If None, returns stored coefficients from fit().
        resize
            Target size for resizing images

        Returns
        -------
        coeffs_u
            Concept coefficients for the inputs (or stored coefficients if inputs=None)

        Raises
        ------
        ValueError
            If inputs is None but no stored coefficients are available
        """
        # If no inputs provided, return stored coefficients from fit()
        if inputs is None:
            self.check_if_fitted()
            if self.factorization.coeffs_u is None:
                raise ValueError(
                    "No stored coefficients available, and no inputs given."
                )
            return self.factorization.coeffs_u

        # encode, but only return coeffs_u as a single tensor
        encoded_data = self.encode(inputs, resize)
        # extract coeffs_u using named attribute access for clarity
        coeffs_u = np.concatenate([enc.coeffs_u for enc in encoded_data], axis=0)
        return coeffs_u

    def transform_latent(self, latent_data: LatentData) -> np.ndarray:
        """
        Transform latent data to concept coefficients.

        Projects latent activations onto the learned concept space using the
        fitted NMF model. This non-differentiable transform is faster than
        transform_latent_differentiable() but cannot be used for gradient-based
        methods.

        Parameters
        ----------
        latent_data
            Single image's latent representation containing activations

        Returns
        -------
        coeffs_u
            Concept coefficients, shape (H, W, n_concepts)

        Raises
        ------
        ValueError
            If latent_data is not a single LatentData instance
        NotFittedError
            If fit() has not been called yet
        """
        if not isinstance(latent_data, LatentData):
            raise ValueError(
                f"transform_latent() only accepts a single LatentData as input, "
                f"got {type(latent_data)}")
        self.check_if_fitted()

        activations = latent_data.get_activations(as_numpy=True)
        needs_reshape = len(activations.shape) > 2 # (N,H,W,C) or (N,Tokens,C)
        if needs_reshape:
            activations_original_shape = activations.shape[:-1]
            activations = np.reshape(activations, (-1, activations.shape[-1]))

        # Encode activations to coefficients using the factorizer
        coeffs_u = self.factorizer.encode(activations)
        if needs_reshape:
            coeffs_u = np.reshape(coeffs_u, (*activations_original_shape, -1))
        return coeffs_u

    def transform_latent_differentiable(self, latent_data: LatentData) -> Any:
        """
        Transform latent data to concept coefficients with gradient preservation.

        Uses differentiable least squares to project activations onto concepts,
        maintaining the computational graph for gradient-based attribution methods.
        Must be implemented by framework-specific subclasses.

        Parameters
        ----------
        latent_data
            Single image's latent representation containing activations

        Returns
        -------
        coeffs_u
            Concept coefficients as framework tensor with gradients

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError(
            "transform_latent_differentiable must be implemented by framework-specific subclass")

    def encode(
        self, inputs: Union[np.ndarray, Any],
        resize: Optional[Tuple[int, int]] = None,
        differentiable: bool = False
    ) -> List[EncodedData]:
        """Encode inputs to latent data and concept coefficients.

        Parameters
        ----------
        inputs
            Input images to encode
        resize
            Target size for resizing images
        differentiable
            If True, preserves gradients for backpropagation using differentiable
            least squares. If False (default), uses standard NMF transform which
            is faster but does not preserve gradients.

        Returns
        -------
        encoded_data
            List of EncodedData named tuples, each containing:
            - latent_data: LatentData object with intermediate activations
            - coeffs_u: Concept coefficients (numpy array or tensor with gradients)

            When differentiable=False, coeffs_u are numpy arrays.
            When differentiable=True, coeffs_u are framework tensors
            (torch.Tensor or tf.Tensor) with gradients preserved.
        """
        latent_data_list = self.latent_extractor.input_to_latent_batched(
            inputs, resize, keep_gradients=differentiable
        )
        encoded_data = []
        for latent_data in latent_data_list:
            if differentiable:
                coeffs_u = self.transform_latent_differentiable(latent_data)
            else:
                coeffs_u = self.transform_latent(latent_data)
            encoded_data.append(EncodedData(latent_data, coeffs_u))
        return encoded_data

    def decode(self, latent_data: LatentData,
               coeffs_u: Union[np.ndarray, Any]) -> StructuredPrediction:
        """Decode concept coefficients back to predictions.

        This method accepts a single LatentData and returns a prediction tensor
        that implements the StructuredPrediction protocol (either MultiBoxTensor for
        object detection or ClassifierTensor for classification).

        The latent_extractor.latent_to_logit() method returns predictions that
        are already formatted by the output_formatter:
        - A list with 1 element (PyTorch formatters)
        - A single tensor directly (TensorFlow formatters with batch_size=1)

        The formatter guarantees the output implements StructuredPrediction protocol.
        This method only handles unwrapping single-element lists.

        Parameters
        ----------
        latent_data
            Single image's latent representation (not batched)
        coeffs_u
            Concept coefficients for reconstruction

        Returns
        -------
        predictions
            Predictions implementing StructuredPrediction protocol (has filter()
            and to_batched_tensor() methods). Concrete types are MultiBoxTensor
            for object detection or ClassifierTensor for classification.

        Raises
        ------
        ValueError
            If latent_data is not a single LatentData instance, or if
            latent_to_logit returns a list with != 1 elements
        """
        if not isinstance(latent_data, LatentData):
            raise ValueError("decode() only accepts a single LatentData as input")

        self.check_if_fitted()

        # Convert coeffs_u to framework tensor if needed
        if isinstance(coeffs_u, np.ndarray):
            coeffs_u = self._to_tensor(coeffs_u, dtype=self._framework_module.float32)

        # Reconstruct activations from concepts
        concept_bank_tensor = self._to_tensor(
            self.factorization.concept_bank_w,
            dtype=self._framework_module.float32)
        activations = coeffs_u @ concept_bank_tensor

        # Set activations and decode through model
        latent_data.set_activations(activations)
        result = self.latent_extractor.latent_to_logit(latent_data)

        # latent_to_logit may return either:
        # - A list with 1 element (e.g., PyTorch formatters always return lists)
        # - A single tensor directly (e.g., TensorFlow with batch_size=1)
        # Extract single prediction if in list form
        if isinstance(result, list):
            if len(result) != 1:
                raise ValueError(
                    f"Expected single-element list for single LatentData, "
                    f"got {len(result)} elements")
            result = result[0]

        return result

    def compute_explanation_per_concept(
            self,
            images: np.ndarray,
            explainer_partial: Callable,
            class_id: Optional[int] = None,
            confidence: Optional[float] = None) -> np.ndarray:
        """
        Compute explanations per concept using the provided explainer partial.

        Wraps the concept decoder in a framework specific wrapper for compatibility
        with Xplique attribution methods.

        For each image, creates a concept decoder and uses the specified attribution
        method to compute how much each concept contributes to the filtered detections.

        Parameters
        ----------
        images
            Input images as numpy arrays
        explainer_partial
            Partial function that creates an attribution explainer when given
            model and batch_size arguments
        class_id
            Target class ID for filtering detections
        confidence
            Confidence threshold for filtering detections

        Returns
        -------
        explanations
            Concatenated explanations for all images, shape (N, H, W, n_concepts)
        """
        explanation_list = []

        with self.latent_extractor.temporary_force_batch_size(1):
            # Encode images to get latent data and concept coefficients
            # The list is composed of 1 EncodedData per image because
            # object detection models can return various number of
            # detection boxes per image
            encoded_data_list = self.encode(images)

            for i, enc in enumerate(encoded_data_list):
                decoded_result = self.decode(enc.latent_data, enc.coeffs_u)
                filtered_result = decoded_result.filter(class_id=class_id, confidence=confidence)
                if len(filtered_result) == 0:  # No detection
                    explanation = np.zeros(
                        (1, enc.coeffs_u.shape[1], enc.coeffs_u.shape[2], enc.coeffs_u.shape[3]))
                    print(
                        f"No detection for image {i}, returning zero explanation "
                        f"of shape {explanation.shape}")
                else:
                    targets = self._to_numpy(filtered_result.to_batched_tensor())
                    decoder = self.make_concept_decoder(enc.latent_data)
                    explainer = explainer_partial(model=decoder, batch_size=1)
                    explanation = explainer.explain(enc.coeffs_u, targets)
                    explanation = explanation.numpy()
                explanation_list.append(explanation)
        return np.concatenate(explanation_list, axis=0)

    def make_concept_decoder(self, latent_data: LatentData) -> Any:
        """Creates a concept decoder for gradient-based attribution.

        The decoder is bound to a specific image's latent representation and
        accepts concept coefficients as input. Suitable for computing gradients
        with respect to concepts using attribution methods like GradientInput.

        Parameters
        ----------
        latent_data
            Image-specific latent representation

        Returns
        -------
        decoder
            ConceptDecoder instance with signature: (coeffs_u) -> predictions
        """
        raise NotImplementedError(
            "make_concept_decoder must be implemented by framework-specific subclass")

    def display_images_per_concept(
        self,
        images: np.ndarray,
        coeffs_u: Optional[np.ndarray] = None,
        filter_percentile: int = 80,
        clip_percentile: int = 5,
        order: Optional[List[int]] = None,
    ) -> Figure:
        """
        Display concept heatmaps overlaid on images.

        Creates a grid visualization with one row per image and one column per
        concept. Each cell shows the input image with a heatmap overlay indicating
        where that concept is activated.

        Parameters
        ----------
        images
            Input images to visualize (array of shape (N, H, W, C) for tensorflow
            or (N, C, H, W) for pytorch)
        coeffs_u
            Optional pre-computed coefficients, shape (N, H, W, C) or (N, Tokens, C).
            If None, coefficients will be computed via transform(images).
        filter_percentile
            Percentile used to filter the concept heatmap
            (only show concept if excess N-th percentile). Defaults to 80.
        clip_percentile
             Percentile value to use if clipping is needed when drawing the concept,
            e.g a value of 1 will perform a clipping between percentile 1 and 99.
            This parameter allows to avoid outliers in case of too extreme values.
            Default to 5.
        order
            Optional list of concept IDs to specify display order. If None,
            concepts are shown in sequential order

        Returns
        -------
        fig
            matplotlib figure with len(images) rows and number_of_concepts columns
        """
        nb_cols = self.number_of_concepts
        nb_rows = len(images)

        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(25, 2 * nb_rows))
        axs = np.atleast_2d(axs)  # fix issue when nb_rows == 1

        if order is None:
            concepts_id = list(range(self.number_of_concepts))
        else:
            concepts_id = order

        for i, c_i in enumerate(concepts_id):
            axs[0, i].set_title(f"concept #{c_i}", fontsize=10)

        # encode images
        if coeffs_u is None:
            coeffs_u = self.transform(images)
            # coeffs_u shape is (N, H, W, C) or (N, Tokens, C)

        if len(coeffs_u.shape) == 3:
            # Reshape (N, Tokens, C) to (N, H, W, C)
            num_images, num_tokens, num_concepts = coeffs_u.shape
            height = width = int(np.sqrt(num_tokens))  # Only valid if Tokens is a perfect square
            if height * width != num_tokens:
                raise ValueError(
                    f"Cannot reshape coeffs_u of shape {coeffs_u.shape} to (N, H, W, C) "
                    f"because Tokens is not a perfect square.")
            coeffs_u = coeffs_u.reshape(num_images, height, width, num_concepts)

        # convert images to be displayed
        if self.framework == 'torch':
            # channel first (N, C, H, W) -> channel last (N, H, W, C)
            images = images.permute(0, 2, 3, 1)

        images = self._to_numpy(images)
        dsize = (images.shape[2], images.shape[1])  # cv2 expects (width, height)

        # Display a heatmap per concept, per image
        for i, c_i in enumerate(concepts_id):
            for image_id, image in enumerate(images):
                # Display the image
                show_ax(image, ax=axs[image_id, i])

                # only show concept if excess N-th percentile
                concept_heatmap = coeffs_u[image_id, :, :, c_i]
                sigma = np.percentile(concept_heatmap.flatten(), filter_percentile)
                heatmap = concept_heatmap * (concept_heatmap > sigma)

                # resize the heatmap before clipping
                heatmap = cv2.resize(
                    heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
                )
                heatmap = _clip_percentile(heatmap, clip_percentile)

                # Display the heatmap overlay
                cmap_idx = c_i % len(self.cmaps)
                show_ax(
                    heatmap, cmap=self.cmaps[::-1][cmap_idx], alpha=0.5, ax=axs[image_id, i]
                )
        return fig

    def display_top_images_per_concept(
        self,
        images: Union[np.ndarray, List[Any]],
        nb_top_images: int = 3,
        filter_percentile: int = 80,
        clip_percentile: int = 5,
        order: Optional[List[int]] = None,
        coeffs_u: Optional[np.ndarray] = None,
    ) -> Figure:
        """Display top N images per concept ranked by average activation.

        Parameters
        ----------
        images
            Input images (as framework tensors or numpy arrays)
        nb_top_images
            Number of top images to display per concept (default: 3)
        filter_percentile
            Percentile threshold for filtering heatmaps (default: 80)
        clip_percentile
            Percentile for clipping heatmap values (default: 5)
        order
            Optional list of concept IDs to specify display order
        coeffs_u
            Optional pre-computed concept coefficients. If None, will call
            self.transform(images) to compute them. Use this to pass the
            coefficients stored in factorization.coeffs_u after fit().

        Returns
        -------
        fig
            matplotlib figure with nb_top_images rows and number_of_concepts columns
        """
        nb_cols = self.number_of_concepts
        nb_rows = nb_top_images

        # Determine image dimensions
        image_shape = images[0].shape if hasattr(images[0], 'shape') else np.array(images[0]).shape
        if image_shape[0] == 3:  # channels first (PyTorch)
            dsize = image_shape[1:3]
            framework = 'torch'
        else:  # channels last (TensorFlow)
            dsize = image_shape[0:2]
            framework = 'tf'
        dsize = (dsize[1], dsize[0])  # cv2 expects (width, height)

        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(25, 2 * nb_rows))
        axs = np.atleast_2d(axs)

        if order is None:
            concepts_id = list(range(self.number_of_concepts))
        else:
            concepts_id = order

        for i, c_i in enumerate(concepts_id):
            axs[0, i].set_title(f"concept #{c_i}", fontsize=10)

        # Get concept activations
        if coeffs_u is None:
            coeffs_u = self.transform(images)
        for i, c_i in enumerate(concepts_id):
            # Find top images for this concept by ranking mean activation
            top_image_for_concept_i = np.argsort(np.mean(coeffs_u, (1, 2))[:, c_i])[::-1]

            for j, image_id in enumerate(top_image_for_concept_i[:nb_top_images]):
                # Convert image to numpy based on framework
                if framework == 'torch':
                    img_np = self._to_numpy(images[image_id].squeeze().permute(1, 2, 0))
                else:  # tensorflow
                    img_np = self._to_numpy(images[image_id])

                show_ax(img_np, ax=axs[j, i])
                concept_heatmap = coeffs_u[image_id, :, :, c_i]

                # only show concept if exceeds N-th percentile
                sigma = np.percentile(concept_heatmap.flatten(), filter_percentile)
                heatmap = concept_heatmap * (concept_heatmap > sigma)

                # resize the heatmap before clipping
                heatmap = cv2.resize(
                    heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC
                )
                heatmap = _clip_percentile(heatmap, clip_percentile)

                cmap_idx = c_i % len(self.cmaps)
                show_ax(heatmap, cmap=self.cmaps[::-1][cmap_idx], alpha=0.5, ax=axs[j, i])

        return fig

    def estimate_importance(
        self,
        images: Union[np.ndarray, List[Any]],
        operator: Callable,
        class_id: int,
        method: str = "gradient_input",
        confidence: float = 0.9,
        spatial_reducer: Optional[str] = "mean",
        abs_before_reduce: bool = True,
        aggregation_reducer: Optional[str] = "mean",
        **method_kwargs: Any
    ) -> np.ndarray:
        """
        Estimate concept importance using the specified attribution method.

        Parameters
        ----------
        images
            Input images to analyze
        operator
            Function to extract target values from predictions
        class_id
            Target class ID for filtering detections
        confidence
            Confidence threshold for filtering detections (default: 0.9)
        method
            Attribution method: "gradient_input" or "sobol" (default: "gradient_input")
        spatial_reducer
            Reducer to use on the spatial dimension of the raw explanations.
            Explanation has shape (num_images, height, width, num_concepts) and will be reduced
            to (num_images, num_concepts) before final aggregation.
            Either "min", "mean", "max", "sum", or `None` to ignore. Default is "mean".
        abs_before_reduce
            Whether to take the absolute value of the explanations before spatial reduction
            (default: True)
        aggregation_reducer
            Reducer to use on the image dimension after spatial reduction.
            Either "min", "mean", "max", "sum", or `None` to ignore. Default is "mean".
            Transform spatial explanations from shape (num_images, num_concepts) to the final
            importances (num_concepts,).
        **method_kwargs
            Additional keyword arguments for the attribution method, such as 'grid_size'
            and 'nb_design' for the Sobol method

        Returns
        -------
        importances
            Importance scores for each concept, shape (n_concepts,)
        """
        if method == "gradient_input":
            explainer_partial = partial(
                GradientInput,
                operator=operator,
                harmonize=False,
                **method_kwargs
            )
        elif method == "sobol":
            # set default values for Sobol-specific parameters if not provided
            method_kwargs.setdefault("grid_size", 8)
            method_kwargs.setdefault("nb_design", 32)
            method_kwargs.setdefault("perturbation_function", "amplitude")
            explainer_partial = partial(
                SobolAttributionMethod,
                nb_channels=self.number_of_concepts,
                operator=operator,
                **method_kwargs
            )
        else:
            raise ValueError(f"Unknown attribution method: {method}")

        return self._estimate_importance(
            images,
            class_id,
            confidence,
            explainer_partial,
            spatial_reducer=spatial_reducer,
            abs_before_reduce=abs_before_reduce,
            aggregation_reducer=aggregation_reducer,
        )

    def _estimate_importance(self,
                            images: Union[np.ndarray, List[Any]],
                            class_id: int,
                            confidence: float,
                            explainer_partial: Callable,
                            spatial_reducer: Optional[str] = 'max',
                            abs_before_reduce: bool = True,
                            aggregation_reducer: Optional[str] = 'mean') -> np.ndarray:
        """
        Internal method for computing concept importance.

        Uses the provided explainer to compute per-concept explanations, then
        aggregates across spatial dimensions and images to produce final
        importance scores.

        Parameters
        ----------
        images
            Input images to analyze
        class_id
            Target class ID for filtering detections
        confidence
            Confidence threshold for filtering detections
        explainer_partial
            Partial function that creates an attribution explainer
        spatial_reducer
            String, name of the reducer to use on the spatial dimension of the raw explanations.
            Explanation has shape (num_images, height, width, num_concepts) and will be reduced
            to (num_images, num_concepts) before final aggregation.
            Either "min", "mean", "max", "sum", "median" or `None` to ignore. Default is "max".
        abs_before_reduce
            Whether to take the absolute value of the explanations before spatial reduction
        aggregation_reducer
            String, name of the reducer to use on the image dimension after spatial reduction.
            Either "min", "mean", "max", "sum", "median" or `None` to ignore. Default is "mean".
            Transform spatial explanations from shape (num_images, num_concepts) to the final
            importances (num_concepts,).

        Returns
        -------
        importances
            Mean absolute importance per concept
        """

        # explainer_partial is a partial function that creates an explainer with
        # all the necessary arguments except the model and batch_size which will
        # be provided in compute_explanation_per_concept

        explanation = self.compute_explanation_per_concept(
            images, explainer_partial, class_id, confidence)

        reducers = {
            'min': np.min,
            'max': np.max,
            'sum': np.sum,
            'mean': np.mean,
            'median': np.median
        }
        if abs_before_reduce:
            explanation = np.abs(explanation)
        if spatial_reducer is not None:
            explanation = reducers[spatial_reducer](explanation, axis=(1, 2))
        if aggregation_reducer is not None:
            importances = reducers[aggregation_reducer](explanation, axis=0)
        else:
            importances = explanation

        return importances
