"""TensorFlow-specific latent extractor for object detection models."""

from typing import Callable, Generator, List, Optional, Tuple, Union

import tensorflow as tf

from xplique.utils_functions.object_detection.base.box_formatter import (
    BaseBoxFormatter,
)
from xplique.utils_functions.object_detection.tf.box_model_wrapper import (
    _pad_and_stack_box_predictions,
)
from xplique.utils_functions.object_detection.tf.multi_box_tensor import TfMultiBoxTensor
from xplique.utils_functions.output_as_list_mixin import OutputAsListMixin

from ..latent_extractor import LatentData, LatentExtractor


class TfLatentExtractor(OutputAsListMixin, LatentExtractor):
    """
    TensorFlow-specific latent extractor for object detection models.

    This class provides TensorFlow-specific implementations for extracting intermediate
    activations from object detection models and decoding them back to predictions.
    It handles batching, resizing, and output formatting for TensorFlow models.

    Parameters
    ----------
    model
        Complete TensorFlow model (for reference, not directly used)
    input_to_latent_model
        TensorFlow model/function that maps inputs to latent activations
    latent_to_logit_model
        TensorFlow model/function that maps latent activations to predictions
    latent_data_class
        Class to use for storing latent data (default: LatentData)
    output_formatter
        Formatter to convert raw model outputs to standardized box format
    batch_size
        Number of samples to process at once

    Attributes
    ----------
    output_as_list
        Whether to return outputs as list of MultiBoxTensor (True) or
        as stacked tensor (False)
    """

    def __init__(
        self,
        model: Callable,
        input_to_latent_model: Callable,
        latent_to_logit_model: Callable,
        latent_data_class=LatentData,
        output_formatter: Optional[BaseBoxFormatter] = None,
        batch_size: int = 8,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        super().__init__(
            model,
            input_to_latent_model,
            latent_to_logit_model,
            latent_data_class,
            output_formatter,
            batch_size,
        )
        self.output_as_list = True

    @staticmethod
    def _prepare_inputs(inputs: tf.Tensor) -> tf.Tensor:
        """Validate image inputs and add a batch dimension to one image."""
        inputs = tf.convert_to_tensor(inputs)
        if inputs.shape.rank not in (3, 4):
            raise ValueError(
                "inputs must have rank 3 (a single image) or rank 4 (a batch of images)"
            )
        if inputs.shape.num_elements() == 0:
            raise ValueError("inputs must contain at least one value")
        if inputs.shape.rank == 3:
            inputs = tf.expand_dims(inputs, axis=0)
        return inputs

    def forward(self, samples: tf.Tensor) -> Union[List["TfMultiBoxTensor"], tf.Tensor]:
        """
        Process samples through the complete model pipeline.

        Encodes inputs to latent representations, decodes to predictions, and
        optionally formats outputs. Return format depends on output_as_list flag.

        Parameters
        ----------
        samples
            Input images as TensorFlow tensors

        Returns
        -------
        outputs
            If output_as_list=True: List of MultiBoxTensor (one per image)
            If output_as_list=False: Zero-padded tensor of shape (N, max_num_boxes, features)
        """
        latent_data = self.input_to_latent(samples)
        outputs = self.latent_to_logit_model(latent_data)
        if self.output_formatter:
            outputs = self.output_formatter(outputs)
            if not self.output_as_list:
                if isinstance(outputs, (list, tuple)):
                    outputs = _pad_and_stack_box_predictions(outputs)
                elif hasattr(outputs, "to_batched_tensor"):
                    outputs = outputs.to_batched_tensor()
                else:
                    outputs = tf.expand_dims(outputs, axis=0)
        return outputs

    def input_to_latent(self, inputs: tf.Tensor) -> LatentData:
        """
        Extract latent representations from input images.

        Encodes single or batched input images into their intermediate latent
        representations. Automatically handles 3D inputs by adding batch dimension.

        Parameters
        ----------
        inputs
            Input image(s) as TensorFlow tensor. Shape: (H, W, C) or (N, H, W, C)

        Returns
        -------
        latent_data
            Extracted latent activations wrapped in LatentData container
        """
        inputs = self._prepare_inputs(inputs)
        latent_data = self.input_to_latent_model(inputs)
        return latent_data

    def input_to_latent_generator(
        self,
        inputs: tf.Tensor,
        resize: Optional[Tuple[int, int]] = None,
        keep_gradients: bool = False,
    ) -> Generator[LatentData, None, None]:
        # pylint: disable=unused-argument
        """
        Generator that yields latent representations for batched inputs.

        Internal generator method that splits inputs into batches, optionally resizes,
        encodes to latent space, and yields results incrementally. Efficiently handles
        large datasets by processing one batch at a time.

        Parameters
        ----------
        inputs
            Input images as TensorFlow tensor. Shape: (N, H, W, C)
        resize
            Target size (height, width) for resizing images before encoding.
            If None, uses original image sizes.
        keep_gradients
            Whether to keep gradients during processing (for gradient-based methods)

        Yields
        ------
        latent_data
            LatentData object containing encoded activations for current batch
        """
        inputs = self._prepare_inputs(inputs)
        batch_count = inputs.shape[0]
        if batch_count is None:
            raise ValueError("inputs must have a known batch dimension")

        for i in range(0, batch_count, self.batch_size):
            i_end = min(i + self.batch_size, batch_count)
            batch = inputs[i:i_end]

            if resize:
                batch = tf.image.resize(batch, size=resize)

            latent_data = self.input_to_latent_model(batch)
            del batch
            yield latent_data

    def latent_to_logit(self, latent_data: LatentData) -> Union[List[TfMultiBoxTensor], tf.Tensor]:
        """
        Decode latent representations into object detection predictions.

        Transforms latent activations back through the decoder portion of the model
        to produce bounding box predictions, class scores, and labels. Optionally
        applies output formatting to standardize the prediction format.

        Parameters
        ----------
        latent_data
            Latent activations to decode, wrapped in LatentData container

        Returns
        -------
        output
            Object detection predictions. If output_formatter is set, returns
            list of MultiBoxTensor objects with standardized box format.
            Otherwise, returns raw model outputs.
        """
        output = self.latent_to_logit_model(latent_data)
        if self.output_formatter:
            output = self.output_formatter(output)
        return output
