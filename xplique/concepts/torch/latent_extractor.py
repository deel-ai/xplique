"""
PyTorch-specific latent extractor implementations for object detection models.
"""

from abc import abstractmethod
from contextlib import nullcontext
from typing import Callable, Generator, List, Optional, Union

import torch

from xplique.utils_functions.object_detection.base.box_formatter import (
    BaseBoxFormatter,
)
from xplique.utils_functions.object_detection.torch.box_model_wrapper import (
    _pad_and_stack_box_predictions,
)
from xplique.utils_functions.object_detection.torch.multi_box_tensor import TorchMultiBoxTensor
from xplique.utils_functions.output_as_list_mixin import OutputAsListMixin

from ..latent_extractor import LatentData, LatentExtractor


class TorchLatentData(LatentData):
    """
    Base class for PyTorch-based latent representations.

    This abstract class provides a common interface for storing intermediate
    activations and positional encodings from PyTorch object detection models.
    Subclasses must implement the detach method for gradient management.

    Attributes
    ----------
    features
        List of feature tensors from the model.
    pos
        List of positional encoding tensors.
    """

    def __init__(self, features: List, pos: List[torch.Tensor]):
        """
        Initialize PyTorch latent data with features and positional encodings.

        Parameters
        ----------
        features
            List of feature tensors from the model.
        pos
            List of positional encoding tensors.
        """
        self.features = features
        self.pos = pos

    @abstractmethod
    def detach(self) -> "TorchLatentData":
        """
        Detach all tensors from the computation graph.

        This method must be implemented by subclasses to detach features
        and positional encodings, preventing gradient computation.

        Returns
        -------
        latent_data
            Self reference after detaching tensors.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("detach method must be implemented by subclasses")


class TorchLatentExtractor(OutputAsListMixin, LatentExtractor):
    """
    PyTorch-specific latent extractor for object detection models.

    This class provides PyTorch-specific implementations for extracting and processing
    latent representations from object detection models. It handles device management,
    batching, and gradient control for PyTorch tensors.

    Attributes
    ----------
    model
        PyTorch object detection model.
    device
        Device for computation ('cuda' or 'cpu').
    training
        Training mode flag from the model.
    output_as_list
        If True, return outputs as list; if False, stack as tensor.
    """

    def __init__(
        self,
        model: Callable,
        input_to_latent_model: Callable,
        latent_to_logit_model: Callable,
        latent_data_class=LatentData,
        output_formatter: Optional[BaseBoxFormatter] = None,
        batch_size: int = 8,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize PyTorch latent extractor with model and configuration.

        Parameters
        ----------
        model
            PyTorch object detection model.
        input_to_latent_model
            Function (g) that extracts latent representations from inputs.
        latent_to_logit_model
            Function (h) that processes latent data to predictions.
        latent_data_class
            Class for storing latent data. Default is LatentData.
        output_formatter
            Optional formatter for model outputs. Default is None.
        batch_size
            Batch size for processing. Default is 8.
        device
            Device for computation. If None, CUDA is used when available and CPU otherwise.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
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
        self.device = self._resolve_device(device)
        self.model = self.model.to(self.device)
        self.output_as_list = True

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        """Select the default device and reject unavailable CUDA targets early."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            resolved_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError(f"Invalid PyTorch device: {device!r}") from error

        if resolved_device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is not available")
            if (
                resolved_device.index is not None
                and resolved_device.index >= torch.cuda.device_count()
            ):
                raise ValueError(f"CUDA device index {resolved_device.index} is not available")
        return resolved_device

    @staticmethod
    def _prepare_inputs(inputs: torch.Tensor) -> torch.Tensor:
        """Validate image inputs and add a batch dimension to one image."""
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim not in (3, 4):
            raise ValueError(
                "inputs must have rank 3 (a single image) or rank 4 (a batch of images)"
            )
        if inputs.numel() == 0:
            raise ValueError("inputs must contain at least one value")
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        return inputs

    @property
    def training(self) -> bool:
        """Reflect the underlying model's training mode."""
        return self.model.training

    def eval(self) -> "TorchLatentExtractor":
        """
        Set model to evaluation mode.

        Returns
        -------
        self
            Self reference for method chaining.
        """
        self.model.eval()
        return self

    def to(self, device: Union[str, torch.device]) -> "TorchLatentExtractor":
        """
        Move model to specified device.

        Parameters
        ----------
        device
            Target device (e.g., 'cuda', 'cpu').

        Returns
        -------
        self
            Self reference for method chaining.
        """
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        return self

    def zero_grad(self) -> "TorchLatentExtractor":
        """
        Zero out all gradients in the model.

        Returns
        -------
        self
            Self reference for method chaining.
        """
        self.model.zero_grad()
        return self

    def forward(self, samples: torch.Tensor) -> Union[List[TorchMultiBoxTensor], torch.Tensor]:
        """
        Run full forward pass from inputs to predictions.

        Parameters
        ----------
        samples
            Input images as PyTorch tensors.

        Returns
        -------
        outputs
            Model predictions, formatted and optionally stacked based on output_as_list setting.
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
        return outputs

    def input_to_latent(self, inputs: torch.Tensor) -> LatentData:
        """
        Extract latent representations from inputs.

        Parameters
        ----------
        inputs
            Input images as PyTorch tensors (3D or 4D).

        Returns
        -------
        latent_data
            Latent representations extracted by input_to_latent_model.
        """
        inputs = self._prepare_inputs(inputs).to(self.device)
        latent_data = self.input_to_latent_model(inputs)
        return latent_data

    def input_to_latent_generator(
        self, inputs: torch.Tensor, resize: Optional[tuple] = None, keep_gradients: bool = False
    ) -> Generator[LatentData, None, None]:
        """
        Generator that yields latent data batch by batch.

        Parameters
        ----------
        inputs
            Input images as PyTorch tensors.
        resize
            Optional target size for resizing inputs. Default is None.
        keep_gradients
            If True, preserve gradients during processing. Default is False.

        Yields
        ------
        latent_data
            LatentData object for each batch, with automatic memory management.
        """
        inputs = self._prepare_inputs(inputs)

        for i in range(0, inputs.shape[0], self.batch_size):
            i_end = min(i + self.batch_size, inputs.shape[0])
            with nullcontext() if keep_gradients else torch.no_grad():
                batch = inputs[i:i_end].to(self.device)

                if resize:
                    batch = torch.nn.functional.interpolate(
                        batch, size=resize, mode="bilinear", align_corners=False
                    )

                latent_data = self.input_to_latent_model(batch)
            del batch
            yield latent_data

    def latent_to_logit(self, latent_data: LatentData) -> List[TorchMultiBoxTensor]:
        """
        Process latent data to model predictions.

        Parameters
        ----------
        latent_data
            Latent representations to process.

        Returns
        -------
        output
            Model predictions (boxes, scores, labels), optionally formatted.
        """
        output = self.latent_to_logit_model(latent_data)
        if self.output_formatter:
            output = self.output_formatter(output)
        return output
