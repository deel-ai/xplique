from typing import Union
import numpy as np
import torch

from xplique.utils_functions.classification.torch import TorchClassifierFormatter
from ..latent_extractor import LatentData, LatentExtractorBuilder
from .latent_extractor import TorchLatentExtractor


class LatentDataLayered(LatentData):
    """
    Stores latent representations (activations) from a layered PyTorch model.

    This class encapsulates intermediate activations from any layered model
    (ResNet, VGG, DenseNet, etc.) used for classification tasks. It stores
    activations from a single intermediate layer of interest.

    Attributes
    ----------
    activations
        Tensor of intermediate activations from the model. Expected shape depends on
        the extraction layer (e.g., (batch, channels, height, width) for conv layers,
        or (batch, features) for fully connected layers).
    """

    def __init__(self, activations: torch.Tensor):
        """
        Initialize layered model latent data with activations.

        Parameters
        ----------
        activations
            Intermediate activations tensor from the model.
        """
        self.activations = activations

    def __len__(self) -> int:
        """
        Return the batch size from the activations.

        Returns
        -------
        batch_size
            Number of samples in the batch.
        """
        return self.activations.shape[0]

    def detach(self) -> None:
        """
        Detach activations tensor from the computation graph.

        This method detaches the activations, preventing gradient computation
        through this tensor.
        """
        self.activations = self.activations.detach()

    def get_activations(self, as_numpy: bool = True, keep_gradients: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract activations as a numpy array or tensor.

        Parameters
        ----------
        as_numpy
            If True, convert tensors to numpy arrays. Default is True.
        keep_gradients
            If True, preserve gradient information. Default is False.

        Returns
        -------
        activations
            Activations as numpy array or PyTorch tensor.
        """
        activations = self.activations

        if not keep_gradients:
            activations = activations.detach()

        is_4d = len(activations.shape) == 4
        if is_4d:
            activations = activations.permute(0, 2, 3, 1)

        if as_numpy:
            activations = activations.cpu().numpy()

        return activations

    def set_activations(self, values: torch.Tensor) -> None:
        """
        Update activations with new values.

        Parameters
        ----------
        values
            New activation tensor (torch.Tensor). If 4D, expected format is
            (N, H, W, C), which will be converted to PyTorch's (N, C, H, W) format.
        """
        is_4d = len(values.shape) == 4
        if is_4d:
            values = values.permute(0, 3, 1, 2)
        self.activations = values

    def to(self, device: torch.device) -> 'LatentDataLayered':
        """
        Move all data to the specified device.

        Parameters
        ----------
        device
            Target device (e.g., torch.device('cuda') or torch.device('cpu')).

        Returns
        -------
        latent_data
            New LatentDataLayered instance with data on the target device.
        """
        return LatentDataLayered(self.activations.to(device))


class LayeredModelExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for generic layered PyTorch models.

    This class provides methods to construct a TorchLatentExtractor for any layered
    model (ResNet, VGG, DenseNet, etc.) by specifying a split layer. It automatically
    splits the model's forward pass into feature extraction (g) and classification (h).
    """

    @classmethod
    def build(
            cls,
            model: torch.nn.Module,
            split_layer: int,
            device: str = 'cuda',
            batch_size: int = 1,
            **kwargs) -> 'TorchLatentExtractor':
        """
        Build a LatentExtractor for a generic layered classifier model.

        This method creates custom g and h functions that split the model's forward pass
        at a specified layer: g extracts features up to and including the split layer,
        and h processes them through the remaining layers to produce predictions.

        Parameters
        ----------
        model
            PyTorch model instance with a sequential structure (via named_modules or
            direct access to layers).
        split_layer
            Integer index of the layer to split at. Supports negative indexing
            (e.g., -1 for the last layer, -2 for the second-to-last). The split
            targets the layer at this index, and h processes the remaining layers.
        device
            Device to run computations on ('cuda' or 'cpu'). Default is 'cuda'.
        batch_size
            Batch size for processing. Default is 1.
        **kwargs
            Additional keyword arguments (ignored, for compatibility).

        Returns
        -------
        latent_extractor
            Configured TorchLatentExtractor instance for the model.

        Raises
        ------
        ValueError
            If split_layer is not found in the model or is an invalid type.
        """
        # Get all model children (sequential layers)
        children_list = list(model.children())

        def g(self, images: torch.Tensor) -> LatentDataLayered:
            """
            Extract activations from the split layer (bottleneck features).

            Parameters
            ----------
            images
                Input images tensor of shape (batch, 3, height, width).

            Returns
            -------
            latent_data
                LatentDataLayered containing split layer activations.
            """
            x = images
            for layer in children_list[:split_layer]:
                x = layer(x)

            # Extract activations at split layer
            activations = x

            return LatentDataLayered(activations)

        def h(self, latent_data: LatentDataLayered) -> torch.Tensor:
            """
            Process latent activations through remaining layers to get logits.

            Parameters
            ----------
            latent_data
                LatentDataLayered containing split layer activations.

            Returns
            -------
            logits
                Classification logits tensor of shape (batch, num_classes).
            """
            x = latent_data.activations

            # Process through remaining layers after split
            for layer in children_list[split_layer:]:
                # Special handling for Sequential containers - check first child
                # This handles cases like VGG's classifier which is a Sequential
                first_child = None
                if isinstance(layer, torch.nn.Sequential):
                    # Get the first child layer to check if it's Linear
                    children_of_seq = list(layer.children())
                    if children_of_seq:
                        first_child = children_of_seq[0]
                elif isinstance(layer, torch.nn.Linear):
                    first_child = layer
                
                # Flatten before FC layers if needed
                if first_child is not None and isinstance(first_child, torch.nn.Linear):
                    if len(x.shape) > 2:
                        x = torch.flatten(x, 1)
                x = layer(x)

            return x

        # Bind methods to the model instance
        model.g = lambda images: g(model, images)
        model.h = lambda latent_data: h(model, latent_data)

        latent_extractor = TorchLatentExtractor(
            model,
            model.g,
            model.h,
            latent_data_class=LatentDataLayered,
            output_formatter=TorchClassifierFormatter(),
            batch_size=batch_size,
            device=device)

        return latent_extractor
