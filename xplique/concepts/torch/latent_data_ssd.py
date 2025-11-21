import types
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision

from xplique.utils_functions.object_detection.torch.box_formatter import (
    TorchvisionBoxFormatter,
)

from ..latent_extractor import LatentData, LatentExtractorBuilder
from .latent_extractor import TorchLatentExtractor


class LatentDataSSD(LatentData):
    """
    LatentData container for SSD models.
    Stores intermediate activations from the SSD backbone (MobileNetV3).
    """

    def __init__(self,
                 images: torchvision.models.detection.image_list.ImageList,
                 original_image_sizes: List,
                 backbone_features: Dict[str, torch.Tensor],
                 extraction_layer: int = -1):
        """
        Initialize LatentDataSSD.

        Parameters
        ----------
        images : ImageList
            Preprocessed images with their sizes
        original_image_sizes : List
            Original image sizes before preprocessing
        backbone_features : Dict[str, torch.Tensor]
            Dictionary of backbone features from MobileNetV3
        extraction_layer : int
            Index of the feature map to use for activations (-1 for last)
        """
        self.images = images
        self.original_image_sizes = original_image_sizes
        self.backbone_features = backbone_features
        self.extraction_layer = extraction_layer

    def __len__(self) -> int:
        """Return batch size."""
        last_key = list(self.backbone_features.keys())[self.extraction_layer]
        return self.backbone_features[last_key].shape[0]

    def detach(self) -> None:
        """Detach all tensors from computation graph."""
        for key, value in self.backbone_features.items():
            self.backbone_features[key] = value.detach()

    def get_activations(self, as_numpy: bool = True, keep_gradients: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract activations for CRAFT processing.

        Parameters
        ----------
        as_numpy : bool
            Convert to numpy array
        keep_gradients : bool
            Keep gradient information

        Returns
        -------
        activations : numpy array or torch.Tensor
            Activations in (N, H, W, C) format
        """
        last_key = list(self.backbone_features.keys())[self.extraction_layer]
        activations = self.backbone_features[last_key]

        if not keep_gradients:
            activations = activations.detach()

        # Convert from (N, C, H, W) to (N, H, W, C)
        is_4d = len(activations.shape) == 4
        if is_4d:
            activations = activations.permute(0, 2, 3, 1)

        if as_numpy:
            activations = activations.cpu().numpy()

        return activations

    def set_activations(self, values: torch.Tensor) -> None:
        """
        Set modified activations back into the latent data.

        Parameters
        ----------
        values : torch.Tensor
            Modified activations (torch.Tensor). If 4D, expected format is
            (N, H, W, C), which will be converted to PyTorch's (N, C, H, W) format.
        """
        is_4d = len(values.shape) == 4
        if is_4d:
            values = values.permute(0, 3, 1, 2)

        last_key = list(self.backbone_features.keys())[self.extraction_layer]
        self.backbone_features[last_key] = values

    def to(self, device: torch.device) -> 'LatentDataSSD':
        """
        Move all data to specified device.

        Parameters
        ----------
        device : torch.device
            Target device

        Returns
        -------
        LatentDataSSD
            New instance on target device
        """
        images = torchvision.models.detection.image_list.ImageList(
            self.images.tensors.to(device),
            self.images.image_sizes
        )
        backbone_features = {}
        for key, value in self.backbone_features.items():
            backbone_features[key] = value.to(device)

        return LatentDataSSD(
            images,
            self.original_image_sizes,
            backbone_features,
            self.extraction_layer)


class SSDExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for SSD models.
    """

    @classmethod
    def build(
            cls,
            model,
            device: str = 'cuda',
            nb_classes: int = 91,
            extraction_layer: int = -1,
            batch_size: int = 1) -> 'TorchLatentExtractor':
        """
        Build a TorchLatentExtractor for SSD models.

        Parameters
        ----------
        model : torch.nn.Module
            SSD model
        device : str
            Device to use ('cuda' or 'cpu')
        nb_classes : int
            Number of classes in the model
        extraction_layer : int
            Index of backbone feature to use for activations
        batch_size : int
            Batch size for processing. Default is 1.
        Returns
        -------
        TorchLatentExtractor
            Configured latent extractor for SSD
        """

        def g(self, images):
            """
            Forward pass up to backbone feature extraction.

            Parameters
            ----------
            images : List[torch.Tensor]
                Input images

            Returns
            -------
            LatentDataSSD
                Latent data containing backbone features
            """
            targets = None

            # Get original image sizes
            original_image_sizes: List[Tuple[int, int]] = []
            for img in images:
                val = img.shape[-2:]
                torch._assert(
                    len(val) == 2,
                    f"expecting the last two dimensions of the torch.Tensor to be H and W instead got {img.shape[-2:]}",
                )
                original_image_sizes.append((val[0], val[1]))

            # Transform the input
            images, targets = self.transform(images, targets)

            # Get features from the backbone
            # For SSD, we extract features from the backbone and extra layers
            backbone_features = self.backbone(images.tensors)

            return LatentDataSSD(
                images,
                original_image_sizes,
                backbone_features,
                extraction_layer=extraction_layer)

        def h(self, latent_data: LatentDataSSD):
            """
            Forward pass from backbone features to final detections.

            Parameters
            ----------
            latent_data : LatentDataSSD
                Latent data containing backbone features

            Returns
            -------
            List[Dict[str, torch.Tensor]]
                List of detection dictionaries
            """
            images = latent_data.images
            original_image_sizes = latent_data.original_image_sizes
            backbone_features = latent_data.backbone_features

            # Convert features dict to list (SSD expects a list)
            if isinstance(backbone_features, dict):
                features = list(backbone_features.values())
            else:
                features = backbone_features

            # Compute SSD head outputs
            head_outputs = self.head(features)

            # Generate anchors
            anchors = self.anchor_generator(images, features)

            # Postprocess detections
            detections: List[Dict[str, torch.Tensor]] = []
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes)

            return detections

        # Attach g and h methods to model
        model.g = types.MethodType(g, model)
        model.h = types.MethodType(h, model)

        # Create formatter and latent extractor
        processed_formatter = TorchvisionBoxFormatter(nb_classes=nb_classes)
        latent_extractor = TorchLatentExtractor(
            model,
            model.g,
            model.h,
            latent_data_class=LatentDataSSD,
            output_formatter=processed_formatter,
            batch_size=batch_size,
            device=device
        )

        return latent_extractor
