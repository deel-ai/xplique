"""
Framework-agnostic CRAFT implementation for holistic model explanations.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.exceptions import NotFittedError

from xplique.attributions.base import WhiteBoxExplainer
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
    kwargs
        Additional keyword arguments passed to ax.imshow()
    """
    img = np.array(img, dtype=np.float32)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    ax.imshow(img, **kwargs)
    ax.axis("off")


class PartialExplainer:
    """
    Wrapper for explainer classes to enable deferred instantiation.

    This class stores an explainer class and its configuration kwargs, allowing
    the explainer to be instantiated later when the model and batch_size become
    available during concept importance estimation or concept localization.

    Parameters
    ----------
    explainer_class
        The explainer class to instantiate (e.g., GradientInput, SobolAttributionMethod).
        Must be callable and accept 'model' and 'batch_size' as keyword arguments.
    kwargs
        Configuration arguments for the explainer (e.g., operator, reducer, grid_size).
        Should NOT include 'model' or 'batch_size' as these will be provided during
        instantiation.

    Raises
    ------
    ValueError
        If 'model' or 'batch_size' are provided in kwargs, since these are reserved for
        later instantiation.
    """

    def __init__(self, explainer_class, **kwargs):
        # Validate that model and batch_size are not provided
        if "model" in kwargs or "batch_size" in kwargs:
            raise ValueError(
                "PartialExplainer should not receive 'model' or 'batch_size' arguments. "
                "These will be provided automatically during explanation."
            )

        self.explainer_class = explainer_class
        self.kwargs = kwargs

    def __call__(self, model, batch_size):
        """
        Instantiate the explainer with the provided model and batch_size.

        Parameters
        ----------
        model
            The model to explain
        batch_size
            Batch size for processing

        Returns
        -------
        explainer
            Instantiated explainer object
        """
        return self.explainer_class(model=model, batch_size=batch_size, **self.kwargs)


class ConceptLocalizer:
    """Map input samples to one scalar score per learned concept.

    Parameters
    ----------
    parent_craft
        Fitted :class:`HolisticCraft` instance used to extract coefficients.
    concept_reducer
        Reduction applied over non-batch, non-concept coefficient dimensions.
        Supported strings are ``"mean"``, ``"sum"``, and ``"max"``. A callable
        may be supplied for custom reduction semantics. Callable reducers receive
        the complete coefficient array with shape
        ``(batch_size, *spatial_dimensions, number_of_concepts)`` and must return
        an array with shape ``(batch_size, number_of_concepts)``. Coefficients are
        reduced as-is, so signed factorizers retain their sign; use a callable
        reducer such as a mean absolute value when magnitude is desired.

    Returns
    -------
    concept_scores
        Float32 concept scores with shape
        ``(batch_size, number_of_concepts)``.

    Notes
    -----
    The parent CRAFT instance must be fitted. Localization repeatedly evaluates
    ``factorizer.encode()`` on perturbed inputs, so the factorizer must support
    out-of-sample encoding.

    Raises
    ------
    ValueError
        If the reducer, concept coefficients, or reduced scores are invalid.
    RuntimeError
        If the factorizer cannot encode unseen activations.
    """

    parent_craft: "HolisticCraft"
    concept_reducer: Union[str, Callable]

    def __init__(
        self,
        parent_craft: "HolisticCraft",
        concept_reducer: Union[str, Callable] = "mean",
    ):
        self.parent_craft = parent_craft
        self.concept_reducer = concept_reducer
        self._validate_reducer()

    def _validate_reducer(self) -> None:
        if isinstance(self.concept_reducer, str):
            allowed_reducers = {"mean", "sum", "max"}
            if self.concept_reducer not in allowed_reducers:
                raise ValueError(
                    "concept_reducer must be one of {'mean', 'sum', 'max'} "
                    "or a callable returning shape (batch_size, number_of_concepts)."
                )
        elif not callable(self.concept_reducer):
            raise ValueError(
                "concept_reducer must be one of {'mean', 'sum', 'max'} "
                "or a callable returning shape (batch_size, number_of_concepts)."
            )

    def _reduce_coefficients(self, coeffs_u: np.ndarray) -> np.ndarray:
        """Reduce concept coefficients to one scalar score per concept and sample."""
        coeffs_u = np.asarray(coeffs_u)
        if coeffs_u.ndim < 2:
            raise ValueError(
                "Concept coefficients must have at least 2 dimensions: "
                "(batch_size, number_of_concepts)."
            )
        if any(size == 0 for size in coeffs_u.shape):
            raise ValueError(
                f"Concept coefficients must not contain empty dimensions, got {coeffs_u.shape}."
            )
        if coeffs_u.shape[-1] != self.parent_craft.number_of_concepts:
            raise ValueError(
                f"Concept coefficients contain {coeffs_u.shape[-1]} concepts, expected "
                f"{self.parent_craft.number_of_concepts}."
            )

        spatial_axes = tuple(range(1, coeffs_u.ndim - 1))
        if not spatial_axes:
            scores = coeffs_u
        elif isinstance(self.concept_reducer, str):
            reducers = {
                "mean": np.mean,
                "sum": np.sum,
                "max": np.max,
            }
            scores = reducers[self.concept_reducer](coeffs_u, axis=spatial_axes)
        else:
            scores = self.concept_reducer(coeffs_u)

        scores = np.asarray(scores)
        expected_shape = (coeffs_u.shape[0], self.parent_craft.number_of_concepts)
        if scores.shape != expected_shape:
            raise ValueError(
                f"Reduced concept scores must have shape {expected_shape}, got {scores.shape}."
            )

        scores = scores.astype(np.float32, copy=False)
        if not np.all(np.isfinite(scores)):
            raise ValueError("Reduced concept scores must contain only finite values.")
        return scores

    def _compute_scores(self, inputs: Any) -> np.ndarray:
        """Encode inputs and reduce their concept coefficients."""
        try:
            coeffs_u = self.parent_craft.transform(inputs)
        except NotImplementedError as error:
            raise RuntimeError(
                "The selected factorizer cannot encode unseen activations. "
                "Input-to-concept localization requires factorizer.encode() "
                "because attribution methods evaluate perturbed inputs."
            ) from error
        return self._reduce_coefficients(coeffs_u)

    def __call__(self, inputs: Any) -> np.ndarray:
        """Return reduced concept scores for a batch of inputs."""
        return self._compute_scores(inputs)


class HolisticCraft(ABC):
    """
    Framework-agnostic CRAFT implementation for holistic model explanations.

    This base class provides concept-based explanations for various model types
    (object detection, classification, etc.) by extracting and analyzing intermediate
    activations using Non-negative Matrix Factorization (NMF). It supports both
    TensorFlow and PyTorch through framework-specific subclasses.

    Ref. Fel et al.,  CRAFT Concept Recursive Activation FacTorization (2023).
    https://arxiv.org/abs/2211.10154
    Ref. Fel et al., A Holistic Approach to Unifying Automatic Concept Extraction
    and Concept Importance Estimation (2023).
    https://arxiv.org/pdf/2306.07304

    The workflow involves:
    1. Extracting latent activations from a computer vision model
    2. Factorizing activations into interpretable concepts using NMF
    3. Computing concept importance by attributing task predictions to concepts
    4. Visualizing coefficient activation maps overlaid on input images
    5. Optionally localizing concept scores to input regions with black-box attribution

    Concept activation, concept importance, and concept localization answer different
    questions. Coefficients describe how strongly a concept is present, importance
    describes how much it contributes to a task prediction, and localization describes
    which input regions drive a selected concept score.

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
        factorizer: Optional[Any] = None,
    ):
        self.latent_extractor = latent_extractor
        self.number_of_concepts = number_of_concepts
        self.batch_size = latent_extractor.batch_size
        self.factorization = None
        self.device = device

        # Use provided factorizer or create default NMF factorizer
        if factorizer is None:
            self.factorizer = SklearnNMFFactorizer(
                n_components=number_of_concepts, alpha_W=1e-2, max_iter=200
            )
        else:
            self.factorizer = factorizer

        # Setup visualization colormaps
        self.cmaps = [Sensitivity._get_alpha_cmap(cmap) for cmap in plt.get_cmap("tab10").colors]

    def check_if_fitted(self):
        """Checks if the factorization model has been fitted to input data.

        Raises
        ------
        NotFittedError
            If the factorization model has not been fitted to input data.
        """
        if not self.factorizer.is_fitted or self.factorization is None:
            raise NotFittedError("The factorization model has not been fitted to input data yet.")

    def fit(self, inputs, class_id: int = 0):
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
        # Pass the data through the 1st part of the model, convert each batch to
        # numpy immediately to free device memory before the next batch
        activations_list = [
            latent_data.get_activations(as_numpy=True)
            for latent_data in self.latent_extractor.input_to_latent_generator(inputs)
        ]
        if not activations_list:
            raise ValueError("No activations extracted from inputs.")
        activations = np.concatenate(activations_list, axis=0)

        needs_reshape = len(activations.shape) > 2  # (N,H,W,C) or (N,Tokens,C)
        if needs_reshape:
            activations_original_shape = activations.shape[:-1]
            spatial_positions = int(np.prod(activations_original_shape[1:]))
            if spatial_positions <= 1:
                warnings.warn(
                    f"Latent activations have only {spatial_positions} spatial position(s) "
                    f"(shape {activations.shape}). CRAFT needs multiple spatial positions to "
                    f"extract meaningful concepts. Check your split_layer, it may be set "
                    f"too close to the end of the network (e.g. after a global pooling layer).",
                    UserWarning,
                    stacklevel=3,
                )
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
            inputs=None,
            class_id=class_id,
            crops=None,
            reducer=self.factorizer,
            concept_bank_w=concept_bank_w,
            crops_u=None,
            coeffs_u=coeffs_u,
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
                raise ValueError("No stored coefficients available, and no inputs given.")
            return self.factorization.coeffs_u

        # encode, but only return coeffs_u as a single tensor
        encoded_data = self.encode(inputs, resize)
        if not encoded_data:
            raise ValueError("No activations extracted from inputs.")
        # extract coeffs_u using named attribute access for clarity
        coeffs_u = np.concatenate([enc.coeffs_u for enc in encoded_data], axis=0)
        return coeffs_u

    def latent_to_concept(self, latent_data: LatentData) -> np.ndarray:
        """
        Transform latent data to concept coefficients.

        Projects latent activations onto the learned concept space using the
        fitted NMF model. This non-differentiable transform is faster than
        latent_to_concept_differentiable() but cannot be used for gradient-based
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
                f"latent_to_concept() only accepts a single LatentData as input, "
                f"got {type(latent_data)}"
            )
        self.check_if_fitted()

        activations = latent_data.get_activations(as_numpy=True)
        needs_reshape = len(activations.shape) > 2  # (N,H,W,C) or (N,Tokens,C)
        if needs_reshape:
            activations_original_shape = activations.shape[:-1]
            activations = np.reshape(activations, (-1, activations.shape[-1]))

        # Encode activations to coefficients using the factorizer
        coeffs_u = self.factorizer.encode(activations)
        if needs_reshape:
            coeffs_u = np.reshape(coeffs_u, (*activations_original_shape, -1))
        return coeffs_u

    @abstractmethod
    def latent_to_concept_differentiable(self, latent_data: LatentData) -> Any:
        """
        Transform latent data to concept coefficients with gradient preservation.

        Uses a differentiable non-negative optimization procedure to project
        activations onto concepts while maintaining the computational graph for
        gradient-based attribution methods. Must be implemented by
        framework-specific subclasses.

        Parameters
        ----------
        latent_data
            Single image's latent representation containing activations

        Returns
        -------
        coeffs_u
            Concept coefficients as framework tensor with gradients
        """
        raise NotImplementedError

    @abstractmethod
    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert a framework-specific tensor to a numpy array.

        Parameters
        ----------
        tensor
            Framework tensor (torch.Tensor or tf.Tensor) or numpy array

        Returns
        -------
        array
            Numpy array
        """
        raise NotImplementedError

    @abstractmethod
    def _to_tensor(self, array: np.ndarray, dtype: Any = None) -> Any:
        """Convert a numpy array to a framework-specific tensor.

        Parameters
        ----------
        array
            Numpy array to convert
        dtype
            Target framework dtype (e.g., torch.float32 or tf.float32)

        Returns
        -------
        tensor
            Framework-specific tensor
        """
        raise NotImplementedError

    def encode(
        self,
        inputs: Union[np.ndarray, Any],
        resize: Optional[Tuple[int, int]] = None,
        differentiable: bool = False,
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
            non-negative optimization. If False (default), uses standard NMF
            transform which is faster but does not preserve gradients.

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
        encoded_data = []
        for latent_data in self.latent_extractor.input_to_latent_generator(
            inputs, resize, keep_gradients=differentiable
        ):
            if differentiable:
                coeffs_u = self.latent_to_concept_differentiable(latent_data)
            else:
                coeffs_u = self.latent_to_concept(latent_data)
            encoded_data.append(EncodedData(latent_data, coeffs_u))
        return encoded_data

    def decode(
        self, latent_data: LatentData, coeffs_u: Union[np.ndarray, Any]
    ) -> StructuredPrediction:
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

        result = self._decode_coefficients(latent_data, coeffs_u)

        # Public decoding returns one structured prediction. Internal attribution
        # decoding uses _decode_coefficients() directly to handle perturbation batches.
        if isinstance(result, list):
            if len(result) != 1:
                raise ValueError(
                    f"Expected single-element list for single LatentData, "
                    f"got {len(result)} elements"
                )
            result = result[0]

        return result

    def _decode_coefficients(
        self, latent_data: LatentData, coeffs_u: Union[np.ndarray, Any]
    ) -> Any:
        """Reconstruct latent activations and return raw formatted predictions."""
        self.check_if_fitted()

        # Convert coeffs_u to framework tensor if needed
        if isinstance(coeffs_u, np.ndarray):
            coeffs_u = self._to_tensor(coeffs_u, dtype=self._framework_module.float32)

        # Reconstruct activations from concepts
        concept_bank_tensor = self._to_tensor(
            self.factorization.concept_bank_w, dtype=self._framework_module.float32
        )
        activations = coeffs_u @ concept_bank_tensor

        # Set activations and decode through model
        latent_data.set_activations(activations)
        return self.latent_extractor.latent_to_logit(latent_data)

    def compute_explanation_per_concept(
        self,
        images: np.ndarray,
        partial_explainer: PartialExplainer,
        class_id: Optional[int] = None,
        confidence: Optional[float] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Compute explanations per concept using the provided explainer.

        Wraps the concept decoder in a framework specific wrapper for compatibility
        with Xplique attribution methods.

        For each image, creates a concept decoder and uses the specified attribution
        method to compute how much each concept contributes to the filtered detections.

        Parameters
        ----------
        images
            Input images as numpy arrays
        partial_explainer
            PartialExplainer instance that creates an attribution explainer when called
            with model and batch_size arguments
        class_id
            Target class ID for filtering detections
        confidence
            Confidence threshold for filtering detections
        verbose
            If True, prints progress information during processing

        Returns
        -------
        explanations
            Concatenated explanations for all images, shape (N, H, W, n_concepts)

        Raises
        ------
        TypeError
            If partial_explainer is not a PartialExplainer instance
        """
        if not isinstance(partial_explainer, PartialExplainer):
            raise TypeError(
                f"partial_explainer must be a PartialExplainer instance,"
                f" got {type(partial_explainer).__name__}.\n"
                f"Wrap your explainer class using PartialExplainer, e.g., "
                f"PartialExplainer(GradientInput, operator=my_operator)"
            )

        explanation_list = []

        # Targets and decoder metadata are prepared per image. Each explainer can
        # still batch coefficient perturbations using the configured batch size.
        with self.latent_extractor.temporary_force_batch_size(1):
            # Encode images to get latent data and concept coefficients
            # The list is composed of 1 EncodedData per image because
            # object detection models can return various number of
            # detection boxes per image
            encoded_data_list = self.encode(images)
            if not encoded_data_list:
                raise ValueError("No latent data extracted from inputs.")

            total_images = len(encoded_data_list)
            for i, enc in enumerate(encoded_data_list):
                if verbose:
                    print(f"\rProcessing image {i + 1}/{total_images}...", end="", flush=True)
                # Pass 1 (no gradients): plain forward pass to build attribution targets.
                decoded_result = self.decode(enc.latent_data, enc.coeffs_u)
                filtered_result = decoded_result.filter(class_id=class_id, confidence=confidence)
                expected_explanation_shape = tuple(enc.coeffs_u.shape)
                is_empty = (
                    bool(filtered_result.is_empty)
                    if hasattr(filtered_result, "is_empty")
                    else len(filtered_result) == 0
                )
                if is_empty:  # No detection
                    explanation = np.zeros(expected_explanation_shape)
                    if verbose:
                        print(
                            f"\nNo detection for image {i}, returning zero explanation "
                            f"of shape {explanation.shape}"
                        )
                else:
                    targets = self._to_numpy(
                        filtered_result.to_attribution_target(class_id).to_batched_tensor()
                    )
                    decoder = self.make_concept_decoder(enc.latent_data)
                    explainer_instance = partial_explainer(
                        model=decoder, batch_size=self.batch_size
                    )

                    # Pass 2 (differentiable): explainer calls ConceptDecoder internally to compute
                    # gradients. Explain the importance of each concept w.r.t the targets.
                    explanation = explainer_instance.explain(enc.coeffs_u, targets)
                    explanation = self._to_numpy(explanation)
                    if explanation.shape != expected_explanation_shape:
                        raise ValueError(
                            f"Explanation shape {explanation.shape} does not match expected shape "
                            f"{expected_explanation_shape} for image {i}. Check that the explainer "
                            f"and concept decoder are correctly implemented."
                        )
                explanation_list.append(explanation)
            if verbose:
                # Print newline after all images are processed
                print()
        return np.concatenate(explanation_list, axis=0)

    @abstractmethod
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
        raise NotImplementedError

    def make_concept_localizer(
        self,
        concept_reducer: Union[str, Callable] = "mean",
    ) -> ConceptLocalizer:
        """Create a callable mapping input images to one score per concept.

        Parameters
        ----------
        concept_reducer
            Reduction used to turn each concept coefficient map into a scalar
            concept score. Supported strings are ``"mean"``, ``"sum"``, and
            ``"max"``. A callable must return shape
            ``(batch_size, number_of_concepts)``.

        Returns
        -------
        localizer
            Callable suitable for Xplique black-box explainers, returning
            float32 scores with shape ``(batch_size, number_of_concepts)``.

        Raises
        ------
        ValueError
            If ``concept_reducer`` is invalid.
        """
        return ConceptLocalizer(self, concept_reducer)

    def _validate_concept_ids(
        self,
        concept_ids: Optional[List[int]],
        parameter_name: str = "concept_ids",
    ) -> List[int]:
        """Validate concept id selections and return them as a list."""
        if concept_ids is None:
            return list(range(self.number_of_concepts))

        try:
            concept_ids = list(concept_ids)
        except TypeError as error:
            raise ValueError(f"{parameter_name} must be an iterable of concept IDs") from error

        if not concept_ids:
            raise ValueError(f"{parameter_name} must contain at least one concept ID")
        if len(concept_ids) > self.number_of_concepts:
            raise ValueError(f"{parameter_name} cannot contain more IDs than number_of_concepts")
        if any(
            not isinstance(concept_id, (int, np.integer))
            or isinstance(concept_id, bool)
            or not 0 <= concept_id < self.number_of_concepts
            for concept_id in concept_ids
        ):
            raise ValueError(
                f"{parameter_name} concept IDs must be integers between 0 and "
                f"{self.number_of_concepts - 1}"
            )
        if len(set(concept_ids)) != len(concept_ids):
            raise ValueError(f"{parameter_name} cannot contain duplicate concept IDs")

        return [int(concept_id) for concept_id in concept_ids]

    def _to_numpy_image_batch(self, images: Union[np.ndarray, List[Any]]) -> np.ndarray:
        """Convert image inputs to a validated rank-4 NumPy batch."""
        if isinstance(images, list):
            image_batch = []
            for img in images:
                image = np.asarray(self._to_numpy(img))
                if image.ndim == 4 and image.shape[0] == 1:
                    image = image[0]
                if image.ndim != 3:
                    expected_shape = (
                        "(C, H, W) or (1, C, H, W)"
                        if self.framework == "torch"
                        else "(H, W, C) or (1, H, W, C)"
                    )
                    raise ValueError(f"images list entries must have shape {expected_shape}.")
                image_batch.append(image)
            if not image_batch:
                raise ValueError("images list must contain at least one image.")
            images_np = np.stack(image_batch, axis=0)
        else:
            images_np = self._to_numpy(images)
            images_np = np.asarray(images_np)

        if images_np.ndim == 3:
            images_np = np.expand_dims(images_np, axis=0)

        if images_np.ndim != 4:
            expected_shape = "(N, C, H, W)" if self.framework == "torch" else "(N, H, W, C)"
            raise ValueError(f"images must be a non-empty image batch with shape {expected_shape}.")
        if images_np.shape[0] == 0 or min(images_np.shape[1:]) <= 0:
            raise ValueError(
                "images must contain at least one image with positive spatial dimensions."
            )

        return images_np

    def _normalize_image_batch_to_nhwc(self, images: Union[np.ndarray, List[Any]]) -> np.ndarray:
        """Convert a framework-native image batch to NHWC float32."""
        images_np = self._to_numpy_image_batch(images)

        if self.framework == "torch":
            images_np = np.moveaxis(images_np, 1, -1)

        return images_np.astype(np.float32, copy=False)

    def compute_concept_attributions(
        self,
        images,
        partial_explainer: PartialExplainer,
        concept_ids: Optional[List[int]] = None,
        concept_reducer: Union[str, Callable] = "mean",
    ) -> np.ndarray:
        """Compute input-space localization maps for selected learned concepts.

        The fitted encoder and factorizer are exposed as a multi-output callable
        whose output is one scalar score per concept. A standard black-box
        attribution method then attributes each selected score to the input.

        Parameters
        ----------
        images
            Non-empty batch of images. TensorFlow inputs use ``(N, H, W, C)``;
            PyTorch inputs use ``(N, C, H, W)``. A single image or a list of
            images in the corresponding framework layout is also accepted.
        partial_explainer
            Deferred black-box Xplique explainer configuration. Omit ``operator``
            or leave it as ``None``; one-hot concept targets select the score directly.
        concept_ids
            Optional iterable of unique concept IDs to localize. Uncomputed
            result channels are filled entirely with ``NaN``. If omitted, all
            concepts are localized.
        concept_reducer
            Reduction from coefficient maps to scalar concept scores.
        Returns
        -------
        concept_maps
            Float32 maps with shape ``(N, H, W, number_of_concepts)``.
            Channel ``k`` always corresponds to concept ``k``.

        Notes
        -----
        Localization evaluates the fitted factorizer on perturbed inputs, so the
        factorizer must support out-of-sample ``encode()``. The localizer keeps
        signed concept scores unchanged; use a custom reducer when concept
        magnitude rather than signed activation is intended. White-box explainers
        are rejected when ``explainer_class`` is a class. Callable factories are
        allowed and are responsible for producing a compatible black-box explainer.

        Raises
        ------
        NotFittedError
            If CRAFT has not been fitted.
        TypeError
            If ``partial_explainer`` is not a :class:`PartialExplainer`.
        ValueError
            If inputs, concept IDs, reducer, explainer type, or explainer output
            shape is invalid.
        RuntimeError
            If the factorizer cannot encode perturbed, unseen activations.
        """
        self.check_if_fitted()

        if not isinstance(partial_explainer, PartialExplainer):
            raise TypeError(
                f"partial_explainer must be a PartialExplainer instance, got "
                f"{type(partial_explainer).__name__}."
            )

        if partial_explainer.kwargs.get("operator") is not None:
            raise ValueError(
                "Concept localization uses one-hot concept targets and does not accept a "
                "custom operator. Omit the operator argument."
            )

        explainer_class = partial_explainer.explainer_class
        # Class-based white-box explainers can be rejected explicitly. Callable factories
        # are intentionally left to their own validation because their class is unknown.
        if isinstance(explainer_class, type) and issubclass(explainer_class, WhiteBoxExplainer):
            raise ValueError(
                "Input-to-concept localization currently supports black-box attribution "
                "methods only. The ConceptLocalizer uses factorizer.encode(), which is "
                "not guaranteed to be differentiable. Use Rise, SobolAttributionMethod, "
                "Occlusion, Lime, KernelShap, or another compatible black-box explainer."
            )

        selected_concepts = self._validate_concept_ids(concept_ids, parameter_name="concept_ids")
        attribution_inputs = self._normalize_image_batch_to_nhwc(images)
        localizer = self.make_concept_localizer(concept_reducer)

        explainer_instance = partial_explainer(model=localizer, batch_size=self.batch_size)

        num_images, height, width, _ = attribution_inputs.shape
        concept_maps = np.full(
            (num_images, height, width, self.number_of_concepts),
            np.nan,
            dtype=np.float32,
        )

        for concept_id in selected_concepts:
            targets = np.zeros((num_images, self.number_of_concepts), dtype=np.float32)
            targets[:, concept_id] = 1.0

            single_concept_map = explainer_instance.explain(attribution_inputs, targets)
            single_concept_map = self._to_numpy(single_concept_map)
            if single_concept_map.ndim == 4:
                if single_concept_map.shape[-1] != 1:
                    raise ValueError(
                        "Concept attribution maps must have one channel when returned as "
                        f"rank-4 tensors, got {single_concept_map.shape}."
                    )
                single_concept_map = single_concept_map[..., 0]
            elif single_concept_map.ndim != 3:
                raise ValueError(
                    "Concept attribution maps must have shape (N, H, W) or (N, H, W, 1), "
                    f"got {single_concept_map.shape}."
                )

            expected_shape = (num_images, height, width)
            if single_concept_map.shape != expected_shape:
                raise ValueError(
                    "Concept attribution map shape must match attribution inputs spatial "
                    f"shape {expected_shape}, got {single_concept_map.shape}."
                )
            if not np.all(np.isfinite(single_concept_map)):
                raise ValueError(
                    f"Concept attribution map for concept {concept_id} must contain "
                    "only finite values."
                )

            concept_maps[..., concept_id] = single_concept_map.astype(np.float32, copy=False)

        return concept_maps

    def _prepare_concept_maps(
        self,
        concept_maps: np.ndarray,
    ) -> Tuple[np.ndarray, List[int]]:
        """Validate concept maps and identify which concept channels are available."""
        concept_maps = np.asarray(concept_maps, dtype=np.float32)

        if concept_maps.ndim != 4:
            raise ValueError("concept_maps must have shape (N, H, W, n_concepts).")
        if any(size == 0 for size in concept_maps.shape):
            raise ValueError(
                f"concept_maps must not contain empty dimensions, got {concept_maps.shape}."
            )
        if concept_maps.shape[-1] != self.number_of_concepts:
            raise ValueError(
                f"concept_maps contains {concept_maps.shape[-1]} concepts, expected "
                f"{self.number_of_concepts}."
            )

        channel_is_uncomputed = np.all(np.isnan(concept_maps), axis=(0, 1, 2))
        channel_is_computed = np.all(np.isfinite(concept_maps), axis=(0, 1, 2))
        if not np.all(channel_is_uncomputed | channel_is_computed):
            raise ValueError("Each concept map channel must be entirely finite or entirely NaN.")

        available_concepts = np.flatnonzero(channel_is_computed).tolist()
        return concept_maps, available_concepts

    def _resolve_concept_map_source(
        self,
        concept_maps: np.ndarray,
        images_np: np.ndarray,
        requested_concepts: Optional[List[int]],
    ) -> Tuple[np.ndarray, List[int]]:
        """Validate display maps and resolve the concept columns to display."""
        concept_maps, available_concepts = self._prepare_concept_maps(concept_maps)
        if concept_maps.shape[0] != images_np.shape[0]:
            raise ValueError(
                "concept_maps and images must contain the same number of samples, "
                f"got {concept_maps.shape[0]} and {images_np.shape[0]}."
            )

        if requested_concepts is None:
            if not available_concepts:
                raise ValueError("No computed concept maps are available to display.")
            concepts_id = available_concepts
        else:
            unavailable = [
                concept_id
                for concept_id in requested_concepts
                if concept_id not in available_concepts
            ]
            if unavailable:
                raise ValueError(
                    f"Requested concept maps are not available for concept IDs: {unavailable}."
                )
            concepts_id = requested_concepts

        return concept_maps, concepts_id

    def _prepare_display_concept_inputs(
        self,
        images: Union[np.ndarray, List[Any]],
        coeffs_u: Optional[np.ndarray],
        order: Optional[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Normalise images and coefficients for display methods.

        Computes concept coefficients when not provided, reshapes token-based
        coefficients to spatial form, converts images to HWC numpy arrays, and
        resolves the ordered list of concept IDs.

        Parameters
        ----------
        images
            Input images as a batch tensor or list of tensors/arrays
        coeffs_u
            Pre-computed concept coefficients, or None to compute via transform()
        order
            Optional list of concept IDs. If None, uses sequential order.

        Returns
        -------
        images_np
            Images as HWC numpy arrays, shape (N, H, W, C)
        coeffs_u
            Concept coefficients, shape (N, H, W, n_concepts)
        concepts_id
            Ordered list of concept IDs to display
        """
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
                    f"because Tokens is not a perfect square."
                )
            coeffs_u = coeffs_u.reshape(num_images, height, width, num_concepts)
        elif len(coeffs_u.shape) != 4:
            raise ValueError(
                "coeffs_u must have shape (N, H, W, n_concepts) or (N, tokens, n_concepts)"
            )

        if coeffs_u.shape[-1] != self.number_of_concepts:
            raise ValueError(
                f"coeffs_u contains {coeffs_u.shape[-1]} concepts, expected "
                f"{self.number_of_concepts}"
            )

        images_np = self._normalize_image_batch_to_nhwc(images)
        concepts_id = self._validate_concept_ids(order, parameter_name="order")

        return images_np, coeffs_u, concepts_id

    def display_concept_heatmap(
        self,
        image: np.ndarray,
        concept_heatmap: np.ndarray,
        concept_idx: int,
        ax: Any,
        filter_percentile: int = 80,
        clip_percentile: int = 5,
    ) -> None:
        """Overlay a single concept heatmap on a single image.

        Displays the image on the given axis, then overlays the concept heatmap
        after filtering low-magnitude activations and clipping outlier values.
        The heatmap may be either a latent concept activation map or an
        input-space attribution map. Absolute magnitudes are displayed, so signed
        attribution direction is intentionally not represented.

        Parameters
        ----------
        image
            Single image as HWC numpy array, shape (H, W, C)
        concept_heatmap
            Concept heatmap, shape (H', W') or (H', W', 1)
        concept_idx
            Index of the concept, used to select the colormap
        ax
            Matplotlib axis on which to draw
        filter_percentile
            Percentile used to filter the concept heatmap
            (only show concept if excess N-th percentile). Defaults to 80.
        clip_percentile
            Percentile value to use if clipping is needed when drawing the concept,
            e.g a value of 1 will perform a clipping between percentile 1 and 99.
            This parameter allows to avoid outliers in case of too extreme values.
            Default to 5.
        """
        concept_heatmap = np.asarray(concept_heatmap, dtype=np.float32)
        if concept_heatmap.ndim == 3 and concept_heatmap.shape[-1] == 1:
            concept_heatmap = concept_heatmap[..., 0]
        elif concept_heatmap.ndim != 2:
            raise ValueError(
                f"concept_heatmap must have shape (H, W) or (H, W, 1), got {concept_heatmap.shape}."
            )

        if not np.all(np.isfinite(concept_heatmap)):
            raise ValueError("concept_heatmap must contain only finite values.")

        concept_heatmap = np.abs(concept_heatmap)

        # Display the image
        show_ax(image, ax=ax)

        # only show concept if excess N-th percentile
        sigma = np.percentile(concept_heatmap, filter_percentile)
        heatmap = concept_heatmap * (concept_heatmap > sigma)

        # resize the heatmap before clipping when needed
        if heatmap.shape[:2] != image.shape[:2]:
            dsize = (image.shape[1], image.shape[0])  # cv2 expects (width, height)
            heatmap = cv2.resize(heatmap[:, :, None], dsize=dsize, interpolation=cv2.INTER_CUBIC)
            heatmap = np.maximum(heatmap, 0.0)
        else:
            heatmap = heatmap[:, :, None]
        heatmap = _clip_percentile(heatmap, clip_percentile)

        # Display the heatmap overlay
        cmap_idx = concept_idx % len(self.cmaps)
        show_ax(heatmap, cmap=self.cmaps[::-1][cmap_idx], alpha=0.5, ax=ax)

    def display_images_per_concept(
        self,
        images: np.ndarray,
        coeffs_u: Optional[np.ndarray] = None,
        filter_percentile: int = 80,
        clip_percentile: int = 5,
        order: Optional[List[int]] = None,
        concept_maps: Optional[np.ndarray] = None,
    ) -> Figure:
        """
        Display concept heatmaps overlaid on images.

        Creates a grid visualization with one row per image and one column per
        concept. Each cell shows the input image with a heatmap overlay indicating
        where that concept is activated.

        Parameters
        ----------
        images
            Input images to visualize (array of shape (N, H, W, C) for tensorflow or (N, C, H, W)
            for pytorch)
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
        concept_maps
            Optional input-space concept attribution maps of shape
            (N, H, W, n_concepts). When provided, maps are used for overlays
            while `coeffs_u` keeps its original meaning. Each concept channel
            must be either entirely finite or entirely ``NaN``. If ``order`` is
            None, only finite channels are displayed. Signed maps are displayed
            by absolute magnitude.

        Returns
        -------
        fig
            matplotlib figure with len(images) rows and number_of_concepts columns
        """
        if concept_maps is None:
            images_np, coeffs_u, concepts_id = self._prepare_display_concept_inputs(
                images, coeffs_u, order
            )
            heatmap_source = coeffs_u
        else:
            images_np = self._normalize_image_batch_to_nhwc(images)
            requested_concepts = (
                None if order is None else self._validate_concept_ids(order, parameter_name="order")
            )
            heatmap_source, concepts_id = self._resolve_concept_map_source(
                concept_maps, images_np, requested_concepts
            )

        nb_cols = len(concepts_id)
        nb_rows = len(images_np)

        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(2 * nb_cols, 2 * nb_rows))
        axs = np.asarray(axs).reshape(nb_rows, nb_cols)

        for i, c_i in enumerate(concepts_id):
            axs[0, i].set_title(f"concept #{c_i}", fontsize=10)

        # Display a heatmap per concept, per image
        for i, c_i in enumerate(concepts_id):
            for image_id, image in enumerate(images_np):
                self.display_concept_heatmap(
                    image=image,
                    concept_heatmap=heatmap_source[image_id, :, :, c_i],
                    concept_idx=c_i,
                    ax=axs[image_id, i],
                    filter_percentile=filter_percentile,
                    clip_percentile=clip_percentile,
                )
        return fig

    def get_topk_images_per_concept(
        self,
        coeffs_u: np.ndarray,
        topk: int = 3,
    ) -> np.ndarray:
        """Return the indices of the top images for each concept, ranked by mean activation.

        Parameters
        ----------
        coeffs_u
            Concept coefficients, shape (N, H, W, n_concepts)
        topk
            Number of top images to return per concept (default: 3)

        Returns
        -------
        top_image_ids
            Array of shape (n_concepts, topk) containing the indices of the top images
            for each concept, ranked by descending mean activation
        """
        # Compute mean activation per image per concept: (N, n_concepts)
        mean_activations = np.mean(coeffs_u, axis=(1, 2))

        # For each concept, find the top-k image indices by descending mean activation
        top_image_ids = np.argsort(mean_activations, axis=0)[::-1, :][:topk, :].T
        # top_image_ids shape: (n_concepts, topk)
        return top_image_ids

    def display_top_images_per_concept(
        self,
        images: Union[np.ndarray, List[Any]],
        topk: int = 3,
        filter_percentile: int = 80,
        clip_percentile: int = 5,
        order: Optional[List[int]] = None,
        coeffs_u: Optional[np.ndarray] = None,
        concept_maps: Optional[np.ndarray] = None,
    ) -> Figure:
        """Display top N images per concept ranked by average activation.

        Parameters
        ----------
        images
            Input images (as framework tensors or numpy arrays)
        topk
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
        concept_maps
            Optional input-space concept attribution maps of shape
            (N, H, W, n_concepts) used only for display overlays. Top-image
            ranking remains based on mean concept coefficients. Each concept
            channel must be either entirely finite or entirely ``NaN``. If
            ``order`` is None, only finite channels are displayed. Signed maps
            are displayed by absolute magnitude.

        Returns
        -------
        fig
            matplotlib figure with topk rows and number_of_concepts columns
        """
        images_np, coeffs_u, concepts_id = self._prepare_display_concept_inputs(
            images, coeffs_u, order
        )

        heatmap_source = coeffs_u
        if concept_maps is not None:
            requested_concepts = None if order is None else concepts_id
            heatmap_source, concepts_id = self._resolve_concept_map_source(
                concept_maps, images_np, requested_concepts
            )

        nb_rows = topk
        nb_cols = len(concepts_id)
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(2 * nb_cols, 2 * nb_rows))
        axs = np.asarray(axs).reshape(nb_rows, nb_cols)

        for i, c_i in enumerate(concepts_id):
            axs[0, i].set_title(f"concept #{c_i}", fontsize=10)

        # Get top image indices for all concepts at once: (n_concepts, topk)
        topk_images_ids = self.get_topk_images_per_concept(coeffs_u, topk)

        for i, c_i in enumerate(concepts_id):
            for j, image_id in enumerate(topk_images_ids[c_i]):
                self.display_concept_heatmap(
                    image=images_np[image_id],
                    concept_heatmap=heatmap_source[image_id, :, :, c_i],
                    concept_idx=c_i,
                    ax=axs[j, i],
                    filter_percentile=filter_percentile,
                    clip_percentile=clip_percentile,
                )

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
        verbose: bool = False,
        **method_kwargs: Any,
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
        verbose
            If True, prints progress information during processing
        **method_kwargs
            Additional keyword arguments for the attribution method, such as 'grid_size'
            and 'nb_design' for the Sobol method

        Returns
        -------
        importances
            Importance scores for each concept, shape (n_concepts,)
        """
        if method == "gradient_input":
            method_kwargs.setdefault("operator", operator)
            method_kwargs.setdefault("reducer", None)
            explainer = PartialExplainer(GradientInput, **method_kwargs)
        elif method == "sobol":
            # set default values for Sobol-specific parameters if not provided
            method_kwargs.setdefault("grid_size", 8)
            method_kwargs.setdefault("nb_design", 32)
            method_kwargs.setdefault("perturbation_function", "amplitude")
            method_kwargs.setdefault("nb_channels", self.number_of_concepts)
            method_kwargs.setdefault("operator", operator)
            explainer = PartialExplainer(SobolAttributionMethod, **method_kwargs)
        else:
            raise ValueError(f"Unknown attribution method: {method}")

        # explainer is a PartialExplainer that creates an explainer with
        # all the necessary arguments except the model and bach_size which
        # will be provided in compute_explanation_per_concept when the
        # concept decoder is created
        explanation = self.compute_explanation_per_concept(
            images, explainer, class_id, confidence, verbose
        )
        return self.reduce_to_importance(
            explanation,
            spatial_reducer=spatial_reducer,
            abs_before_reduce=abs_before_reduce,
            aggregation_reducer=aggregation_reducer,
        )

    def reduce_to_importance(
        self,
        explanation: np.ndarray,
        spatial_reducer: Optional[str] = "max",
        abs_before_reduce: bool = True,
        aggregation_reducer: Optional[str] = "mean",
    ) -> np.ndarray:
        """
        Reduce pre-computed concept explanations to global importance scores.

        Parameters
        ----------
        explanation
            Per-concept explanations, shape (N, H, W, n_concepts), as returned by
            :meth:`compute_explanation_per_concept`.
        spatial_reducer
            Reducer applied over the spatial dimensions (H, W) to collapse each image
            to a per-concept score vector. Either "min", "mean", "max", "sum", "median"
            or `None` to skip. Default is "max".
        abs_before_reduce
            Whether to take the absolute value of explanations before spatial reduction.
            Default is True.
        aggregation_reducer
            Reducer applied over the image dimension after spatial reduction to produce
            a single score per concept. Either "min", "mean", "max", "sum", "median"
            or `None` to skip (returns per-image scores). Default is "mean".

        Returns
        -------
        importances
            Importance scores for each concept, shape (n_concepts,).
        """
        reducers = {
            "min": np.min,
            "max": np.max,
            "sum": np.sum,
            "mean": np.mean,
            "median": np.median,
        }
        if abs_before_reduce:
            explanation = np.abs(explanation)
        spatial_axes = tuple(range(1, explanation.ndim - 1))
        if spatial_reducer is not None and spatial_axes:
            explanation = reducers[spatial_reducer](explanation, axis=spatial_axes)
        if aggregation_reducer is not None:
            importances = reducers[aggregation_reducer](explanation, axis=0)
        else:
            importances = explanation
        return importances

    def reduce_to_prevalence(self, explanation: np.ndarray) -> np.ndarray:
        """
        Compute concept prevalence from pre-computed explanations.

        A concept is prevalent when it is frequently the most important one across
        images, i.e. it dominates the most samples (argmax of per-image importance).

        Ref. Fel et al., A Holistic Approach to Unifying Automatic Concept Extraction
        and Concept Importance Estimation (2023).
        https://arxiv.org/pdf/2306.07304

        Parameters
        ----------
        explanation
            Per-concept explanations, shape (N, H, W, n_concepts), as returned by
            :meth:`compute_explanation_per_concept`.

        Returns
        -------
        prevalence
            Fraction of images for which each concept is dominant, shape (n_concepts,).
            Values sum to 1.
        """
        spatial_axes = tuple(range(1, explanation.ndim - 1))
        per_image = (
            np.mean(explanation, axis=spatial_axes) if spatial_axes else explanation
        )  # (N, n_concepts)
        dominant = np.argmax(per_image, axis=-1)  # (N,)
        prevalence = np.zeros(self.number_of_concepts)
        for c in range(self.number_of_concepts):
            prevalence[c] = np.sum(dominant == c) / len(dominant)
        return prevalence

    def reduce_to_reliability(self, explanation: np.ndarray, accuracy: np.ndarray) -> np.ndarray:
        """
        Compute concept reliability from pre-computed explanations and per-image accuracy.

        A concept is reliable when images for which it is the most important concept
        also tend to be correctly predicted. Reliability is the mean accuracy of the
        group of images sharing the same dominant concept.

        Ref. Fel et al., A Holistic Approach to Unifying Automatic Concept Extraction
        and Concept Importance Estimation (2023).
        https://arxiv.org/pdf/2306.07304

        Parameters
        ----------
        explanation
            Per-concept explanations, shape (N, H, W, n_concepts), as returned by
            :meth:`compute_explanation_per_concept`.
        accuracy
            Per-image accuracy scores, shape (N,). For classification: 0.0 or 1.0.
            For object detection: per-image IoU, AP, or any scalar correctness metric
            computed externally by the user.

        Returns
        -------
        reliability
            Mean accuracy per dominant-concept group, shape (n_concepts,).
            Concepts with no dominant image get a reliability of 0.0.
        """
        accuracy = np.asarray(accuracy)
        spatial_axes = tuple(range(1, explanation.ndim - 1))
        per_image = (
            np.mean(explanation, axis=spatial_axes) if spatial_axes else explanation
        )  # (N, n_concepts)
        dominant = np.argmax(per_image, axis=-1)  # (N,)
        reliability = np.zeros(self.number_of_concepts)
        for c in range(self.number_of_concepts):
            mask = dominant == c
            reliability[c] = np.mean(accuracy[mask]) if mask.any() else 0.0
        return reliability


class ConceptDecoder:
    """
    Concept decoder module.

    Converts concept coefficients back to object detection predictions by
    reconstructing activations and passing them through the decoder network.

    Attributes
    ----------
    parent_craft
        HolisticCraft instance used to decode predictions
    latent_data
        Image-specific latent representation to use for decoding
    """

    parent_craft: HolisticCraft
    latent_data: LatentData

    def set_latent_data(self, latent_data: LatentData) -> None:
        """
        Update the latent data for this decoder.

        Parameters
        ----------
        latent_data
            New latent representation to use
        """
        self.latent_data = latent_data

    def _decode(self, coeffs_u):
        """
        Decode concept coefficients to predictions

        Parameters
        ----------
        coeffs_u
            Batched concept coefficients.

        Returns
        -------
        logits
            Predictions as a dense batched tensor. Object detections are zero-padded
            to the largest number of boxes in the batch.
        """
        predictions = self.parent_craft._decode_coefficients(self.latent_data, coeffs_u)
        return self._predictions_to_tensor(predictions)

    def _predictions_to_tensor(self, predictions):
        """Convert framework-specific structured predictions to a dense tensor."""
        raise NotImplementedError
