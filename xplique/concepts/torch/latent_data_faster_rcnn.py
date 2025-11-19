import types
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import torchvision

from xplique.utils_functions.object_detection.torch.box_formatter import (
    TorchvisionBoxFormatter,
)

from ..latent_extractor import LatentData, LatentExtractorBuilder
from .latent_extractor import TorchLatentExtractor


class LatentDataFasterRcnn(LatentData):
    """
    Stores latent representations (features and metadata) from Faster R-CNN's ResNet backbone or FPN.

    This class encapsulates the multi-scale feature maps from either the ResNet backbone
    or the FPN of a Faster R-CNN model, along with image metadata needed for processing.
    Features are stored as an OrderedDict with keys representing different feature pyramid levels.

    Attributes
    ----------
    images
        ImageList containing preprocessed image tensors and their sizes.
    original_image_sizes
        List of tuples containing original image dimensions before preprocessing.
    features
        OrderedDict of feature maps at different scales.
    extraction_layer
        Index specifying which feature map to use as activations. Default is -1 (last feature).
    """

    def __init__(self,
                 # ImageList contains 'tensors(tensor)' and 'image_sizes(list of int)'
                 images: torchvision.models.detection.image_list.ImageList,
                 original_image_sizes: List,
                 features: OrderedDict,
                 extraction_layer: int = -1):
        """
        Initialize Faster R-CNN latent data with images, sizes, and features.

        Parameters
        ----------
        images
            ImageList containing preprocessed image tensors and sizes.
        original_image_sizes
            List of original image dimensions.
        features
            OrderedDict of feature maps.
        extraction_layer
            Index specifying which feature map to use. Default is -1.
        """
        self.images = images
        self.original_image_sizes = original_image_sizes
        self.features = features
        self.extraction_layer = extraction_layer

    def __len__(self) -> int:
        """
        Return the batch size from the feature maps.

        Returns
        -------
        batch_size
            Number of samples in the batch.
        """
        current_key = list(self.features.keys())[self.extraction_layer]
        return self.features[current_key].shape[0]

    def detach(self) -> None:
        """
        Detach all feature tensors from the computation graph.

        This method detaches all feature maps, preventing gradient computation
        through these tensors.
        """
        for key, value in self.features.items():
            self.features[key] = value.detach()

    def get_activations(self, as_numpy: bool = True, keep_gradients: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract the feature map at the specified index as activations.

        Parameters
        ----------
        as_numpy
            If True, convert tensors to numpy arrays. Default is True.
        keep_gradients
            If True, preserve gradient information. Default is False.

        Returns
        -------
        activations
            Feature map as numpy array or PyTorch tensor. If 4D (N, C, H, W),
            converted to (N, H, W, C) format for compatibility.
        """
        current_key = list(self.features.keys())[self.extraction_layer]
        activations = self.features[current_key]

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
        Update the feature map at the specified index with new activation values.

        Parameters
        ----------
        values
            New feature tensor values. Expected format is (N, H, W, C), which will
            be converted to PyTorch's (N, C, H, W) format.
        """

        # tensorflow/numpy -> torch
        # activations: (N, H, W, C) -> (N, C, H, W)
        is_4d = len(values.shape) == 4
        if is_4d:
            values = values.permute(0, 3, 1, 2)

        current_key = list(self.features.keys())[self.extraction_layer]
        self.features[current_key] = values

    def to(self, device: torch.device) -> 'LatentData':
        """
        Move all data to the specified device.

        Parameters
        ----------
        device
            Target device (e.g., torch.device('cuda') or torch.device('cpu')).

        Returns
        -------
        latent_data
            New LatentDataFasterRcnn instance with data on the target device.
        """
        images = torchvision.models.detection.image_list.ImageList(
            self.images.tensors.to(device),
            self.images.image_sizes
        )
        features = OrderedDict()

        for key, value in self.features.items():
            features[key] = value.to(device)

        return LatentDataFasterRcnn(images, self.original_image_sizes, features,
                                    self.extraction_layer)


class FasterRcnnExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for Faster R-CNN models.

    This class provides methods to construct a TorchLatentExtractor specifically
    configured for Faster R-CNN object detection models. It defines the forward pass
    split into backbone feature extraction (g) and FPN + RPN + RoI head processing (h).
    """

    @classmethod
    def build(
            cls,
            model,
            device: str = 'cuda',
            nb_classes: int = 91,
            extraction_layer: int = -1,
            extract_location: str = 'resnet',
            batch_size: int = 1) -> 'TorchLatentExtractor':
        """
        Build a LatentExtractor for a Faster R-CNN model.

        This method creates custom g and h functions that split the model's forward pass:
        g extracts ResNet backbone features (and optionally FPN features), and h processes
        them through the Feature Pyramid Network (FPN), Region Proposal Network (RPN), and RoI head.

        Parameters
        ----------
        model
            PyTorch Faster R-CNN model instance with backbone, rpn, roi_heads,
            transform attributes.
        device
            Device to run computations on ('cuda' or 'cpu'). Default is 'cuda'.
        nb_classes
            Number of object classes the model detects. Default is 91 (COCO).
        extraction_layer
            Index specifying which feature pyramid level to use as activations. Default is -1.
        extract_location
            Where to extract features from: 'resnet' (default) or 'fpn'.
        batch_size
            Batch size for processing. Default is 1.
        Returns
        -------
        latent_extractor
            Configured TorchLatentExtractor instance for the Faster R-CNN model.
        """

        def g(self, images):
            targets = None

            # get the original image sizes
            original_image_sizes: List[Tuple[int, int]] = []
            for img in images:
                val = img.shape[-2:]
                torch._assert(
                    len(val) == 2,
                    f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
                )
                original_image_sizes.append((val[0], val[1]))

            # transform the input
            images, targets = self.transform(images, targets)

            # get the features from the backbone
            # Split at backbone.body() to get ResNet features before FPN
            resnet_features = self.backbone.body(images.tensors)

            # Determine which features to pass based on extract_location
            if extract_location == 'fpn':
                # Compute and pass FPN features
                features = self.backbone.fpn(resnet_features)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([("0", features)])
            else:
                # Pass ResNet features
                features = resnet_features

            return LatentDataFasterRcnn(
                images,
                original_image_sizes,
                features,
                extraction_layer=extraction_layer)

        def h(self, latent_data: LatentDataFasterRcnn):
            images = latent_data.images
            original_image_sizes = latent_data.original_image_sizes
            features = latent_data.features

            # If extract_location is 'resnet', features are ResNet features and need FPN
            # If extract_location is 'fpn', features are already FPN features
            if extract_location == 'fpn':
                # Features are already FPN features from g()
                fpn_features = features
            else:
                # Features are ResNet features, need to compute FPN
                fpn_features = self.backbone.fpn(features)

            if isinstance(fpn_features, torch.Tensor):
                fpn_features = OrderedDict([("0", fpn_features)])

            # Get proposals from RPN
            proposals, proposal_losses = self.rpn(images, fpn_features, targets=None)

            # Get detections from RoI heads
            detections, detector_losses = self.roi_heads(fpn_features, proposals, images.image_sizes, targets=None)

            # Postprocess detections
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes)

            return detections

        model.g = types.MethodType(g, model)
        model.h = types.MethodType(h, model)

        # Informative logging about extraction configuration
        print(f"Building Faster R-CNN extractor: extracting from '{extract_location}' at layer index {extraction_layer}")

        processed_formatter = TorchvisionBoxFormatter(nb_classes=nb_classes)
        latent_extractor = TorchLatentExtractor(
            model,
            model.g,
            model.h,
            latent_data_class=LatentDataFasterRcnn,
            output_formatter=processed_formatter,
            batch_size=batch_size,
            device=device)
        return latent_extractor
