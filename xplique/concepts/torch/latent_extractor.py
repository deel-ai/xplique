from abc import abstractmethod
from math import ceil
from typing import Callable, Generator, List, Optional, Union

import torch

from xplique.utils_functions.object_detection.base.box_formatter import (
    BaseBoxFormatter,
)
from xplique.utils_functions.object_detection.torch.multi_box_tensor import MultiBoxTensor

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
    def detach(self) -> 'LatentData':
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


class TorchLatentExtractor(LatentExtractor):
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

    def __init__(self, model: Callable,
                 input_to_latent_model: Callable,
                 latent_to_logit_model: Callable,
                 latent_data_class=LatentData,
                 output_formatter: Optional[BaseBoxFormatter] = None,
                 batch_size: int = 8,
                 device: str = 'cuda'):
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
            Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
        """
        super().__init__(
            model,
            input_to_latent_model,
            latent_to_logit_model,
            latent_data_class,
            output_formatter,
            batch_size)
        self.model = self.model.to(device)
        self.device = device
        self.training = self.model.training
        self.output_as_list = True

    def eval(self) -> 'TorchLatentExtractor':
        """
        Set model to evaluation mode.

        Returns
        -------
        self
            Self reference for method chaining.
        """
        self.model.eval()
        return self

    def to(self, device: str) -> 'TorchLatentExtractor':
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
        self.model.to(device)
        return self

    def zero_grad(self) -> 'TorchLatentExtractor':
        """
        Zero out all gradients in the model.

        Returns
        -------
        self
            Self reference for method chaining.
        """
        self.model.zero_grad()
        return self

    def set_output_as_list(self) -> None:
        """
        Configure output format as list.

        Sets output_as_list flag to True, so outputs are returned as lists
        rather than stacked tensors.
        """
        self.output_as_list = True

    def set_output_as_tensor(self) -> None:
        """
        Configure output format as tensor.

        Sets output_as_list flag to False, so outputs are stacked into
        a single tensor when possible.
        """
        self.output_as_list = False

    def forward(self, samples: torch.Tensor) -> Union[List[MultiBoxTensor], torch.Tensor]:
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
        latent_data = self.input_to_latent_model(samples)
        outputs = self.latent_to_logit_model(latent_data)
        if self.output_formatter:
            outputs = self.output_formatter(outputs)
            if not self.output_as_list:
                if isinstance(outputs, (list, tuple)):
                    outputs = torch.stack(outputs, dim=0)
        return outputs

    def forward_batched(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass with automatic batching.

        Parameters
        ----------
        samples
            Input images as PyTorch tensors.

        Returns
        -------
        results
            Concatenated predictions from all batches.
        """
        results = []
        for latent_data in self._input_to_latent_generator(samples):
            output = self.latent_to_logit_model(latent_data)
            if self.output_formatter:
                output = self.output_formatter(output)
            results.append(output)
        results = torch.cat(results, dim=0)
        return results

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
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
        latent_data = self.input_to_latent_model(inputs)
        return latent_data

    def input_to_latent_batched(
            self,
            inputs: torch.Tensor,
            resize: Optional[tuple] = None,
            keep_gradients: bool = False) -> List[LatentData]:
        """
        Extract latent representations with automatic batching.

        Parameters
        ----------
        inputs
            Input images as PyTorch tensors.
        resize
            Optional target size for resizing inputs. Default is None.
        keep_gradients
            If True, preserve gradients during processing. Default is False.

        Returns
        -------
        latent_data_list
            List of LatentData objects from each batch.
        """
        latent_data_list = list(self._input_to_latent_generator(inputs, resize, keep_gradients))
        return latent_data_list

    def _input_to_latent_generator(
            self,
            inputs: torch.Tensor,
            resize: Optional[tuple] = None,
            keep_gradients: bool = False) -> Generator[LatentData, None, None]:
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
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)

        nb_batchs = ceil(len(inputs) / self.batch_size)
        start_ids = [i * self.batch_size for i in range(nb_batchs)]

        if keep_gradients:
            for i in start_ids:
                i_end = min(i + self.batch_size, len(inputs))
                batch = inputs[i:i_end].to(self.device)

                if resize:
                    batch = torch.nn.functional.interpolate(batch, size=resize, mode='bilinear',
                                                            align_corners=False)

                latent_data = self.input_to_latent_model(batch)
                del batch
                torch.cuda.empty_cache()
                yield latent_data
        else:
            with torch.no_grad():
                for i in start_ids:
                    i_end = min(i + self.batch_size, len(inputs))
                    batch = inputs[i:i_end].to(self.device)

                    if resize:
                        batch = torch.nn.functional.interpolate(batch, size=resize, mode='bilinear',
                                                                align_corners=False)

                    latent_data = self.input_to_latent_model(batch)
                    del batch
                    torch.cuda.empty_cache()
                    yield latent_data

    def latent_to_logit(self, latent_data: LatentData) -> List[MultiBoxTensor]:
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

    def latent_to_logit_batched(self, latent_data: LatentData, no_grad: bool = True) -> torch.Tensor:
        """
        Process latent data to predictions with automatic batching.

        Parameters
        ----------
        latent_data
            Latent representations to process.
        no_grad
            If True, disable gradient computation. Default is True.

        Returns
        -------
        result
            Concatenated predictions from all batches.
        """
        if no_grad:
            with torch.no_grad():
                output_list = list(self._latent_to_logit_generator(latent_data))
        else:
            output_list = list(self._latent_to_logit_generator(latent_data))
        result = torch.cat(output_list, dim=0)
        return result

    def _latent_to_logit_generator(self, latent_data: LatentData) -> Generator[List[MultiBoxTensor], None, None]:
        """
        Generator that yields predictions batch by batch.

        Parameters
        ----------
        latent_data
            Latent representations to process.

        Yields
        ------
        boxes_scores_labels
            Model predictions for each batch, with automatic memory management.
        """
        nb_batchs = ceil(len(latent_data) / self.batch_size)
        start_ids = [i * self.batch_size for i in range(nb_batchs)]

        for i in start_ids:
            batch = latent_data[i:i + self.batch_size].to(self.device)
            boxes_scores_labels = self.latent_to_logit_model(batch)
            del batch
            if self.output_formatter:
                boxes_scores_labels = self.output_formatter(boxes_scores_labels)
            yield boxes_scores_labels
