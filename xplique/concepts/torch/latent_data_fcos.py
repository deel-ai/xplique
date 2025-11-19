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


class LatentDataFcos(LatentData):
    """
    Stores latent representations (features and metadata) from FCOS's backbone.

    This class encapsulates the multi-scale feature maps from an FCOS model,
    along with image metadata needed for processing. Features are stored as an
    OrderedDict with keys representing different feature pyramid levels.

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
        Initialize FCOS latent data with images, sizes, and features.

        Parameters
        ----------
        images
            ImageList containing preprocessed image tensors and sizes.
        original_image_sizes
            List of original image dimensions.
        features
            OrderedDict of feature maps (either from ResNet backbone or FPN).
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
            New LatentDataFcos instance with data on the target device.
        """
        images = torchvision.models.detection.image_list.ImageList(
            self.images.tensors.to(device),
            self.images.image_sizes
        )
        features = OrderedDict()
        for key, value in self.features.items():
            features[key] = value.to(device)

        return LatentDataFcos(images, self.original_image_sizes, features,
                              self.extraction_layer)


class FcosExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for FCOS models.

    This class provides methods to construct a TorchLatentExtractor specifically
    configured for FCOS (Fully Convolutional One-Stage) object detection models.
    It defines the forward pass split into backbone feature extraction (g) and
    FPN + detection head processing (h).
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
        Build a LatentExtractor for an FCOS model.

        This method creates custom g and h functions that split the model's forward pass:
        g extracts ResNet backbone features (and optionally FPN features), and h processes
        them through the Feature Pyramid Network (FPN) and FCOS detection head.

        Parameters
        ----------
        model
            PyTorch FCOS model instance with backbone, head, anchor_generator,
            transform, and postprocess_detections attributes.
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
            Configured TorchLatentExtractor instance for the FCOS model.
        """

        def g(self, images):
            targets = None

            # get the original image sizes
            original_image_sizes: List[Tuple[int, int]] = []
            for img in images:
                val = img.shape[-2:]
                torch._assert(
                    len(val) == 2,
                    f"expecting the last two dimensions of the torch.Tensor to be H and W instead got {img.shape[-2:]}",
                )
                original_image_sizes.append((val[0], val[1]))

            # transform the input
            images, targets = self.transform(images, targets)

            # get the features from the backbone
            resnet_features = self.backbone.body(images.tensors)

            # Determine which features to use based on extract_location
            if extract_location == 'fpn':
                features = self.backbone.fpn(resnet_features)
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([("0", features)])
            else:
                features = resnet_features

            return LatentDataFcos(
                images,
                original_image_sizes,
                features,
                extraction_layer=extraction_layer)

        def h(self, latent_data: LatentDataFcos):
            images = latent_data.images
            original_image_sizes = latent_data.original_image_sizes

            # If extract_location='resnet', we need to apply FPN here
            # If extract_location='fpn', features are already FPN features
            if extract_location == 'fpn':
                features = latent_data.features
            else:
                # Apply FPN to ResNet features
                features = self.backbone.fpn(latent_data.features)

            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])

            # Convert features dict to list for FCOS head
            features_list = list(features.values())

            # compute the FCOS head outputs using the features
            head_outputs = self.head(features_list)

            # FCOS-specific postprocessing
            anchors = self.anchor_generator(images, features_list)

            # Recover level sizes (needed for splitting outputs)
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features_list]

            # Split outputs per level
            split_head_outputs: Dict[str, List[torch.Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            detections = self.postprocess_detections(
                split_head_outputs,
                split_anchors,
                images.image_sizes
            )

            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes)

            return detections

        model.g = types.MethodType(g, model)
        model.h = types.MethodType(h, model)

        # Informative logging about extraction configuration
        print(f"Building FCOS extractor: extracting from '{extract_location}' at layer index {extraction_layer}")

        processed_formatter = TorchvisionBoxFormatter(nb_classes=nb_classes)
        latent_extractor = TorchLatentExtractor(
            model,
            model.g,
            model.h,
            latent_data_class=LatentDataFcos,
            output_formatter=processed_formatter,
            batch_size=batch_size,
            device=device)
        return latent_extractor
