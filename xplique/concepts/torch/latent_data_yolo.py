import copy
import types
from typing import Union, List, Optional

import numpy as np
import torch
import ultralytics
from torch import Tensor

from xplique.concepts.torch.latent_extractor import TorchLatentExtractor
from xplique.utils_functions.object_detection.torch.box_formatter import (
    YoloRawBoxFormatter,
)

from ..latent_extractor import LatentData, LatentExtractorBuilder


class LatentDataYolo(LatentData):
    """
    Stores latent representations (activations and intermediate outputs) from YOLO models.

    This class encapsulates intermediate activations and layer outputs during YOLO's
    forward pass. The x attribute contains the main activation tensor at the split point,
    while y contains a list of outputs from earlier layers needed for skip connections.

    Attributes
    ----------
    x
        Main activation tensor at the model split point.
    y
        List of intermediate layer outputs, used for skip connections in later layers.
    """

    def __init__(self, x: torch.Tensor, y: List[Optional[torch.Tensor]]):
        """
        Initialize YOLO latent data with activation tensor and intermediate outputs.

        Parameters
        ----------
        x
            Main activation tensor at the split point.
        y
            List of intermediate layer outputs.
        """
        self.x = x
        self.y = y

    def get_activations(self, as_numpy: bool = True, keep_gradients: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract the main activation tensor.

        Parameters
        ----------
        as_numpy
            If True, convert tensors to numpy arrays. Default is True.
        keep_gradients
            If True, preserve gradient information. Default is False.

        Returns
        -------
        activations
            Activation tensor as numpy array or PyTorch tensor. If 4D (N, C, H, W),
            converted to (N, H, W, C) format for compatibility.
        """
        activations = self.x

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
        Update the main activation tensor with new values.

        Parameters
        ----------
        values
            New activation tensor values. Expected format is (N, H, W, C), which will
            be converted to PyTorch's (N, C, H, W) format.
        """
        is_4d = len(values.shape) == 4
        if is_4d:
            values = values.permute(0, 3, 1, 2)
        self.x = values

    def __str__(self) -> str:
        """
        Return string representation of the latent data.

        Returns
        -------
        representation
            String describing the shapes of x and length of y.
        """
        return f"LatentDataYolo(x shape: {self.x.shape}, y length: {len(self.y)})"


def make_head_differentiable(model: ultralytics.nn.tasks.DetectionModel) -> ultralytics.nn.tasks.DetectionModel:
    """
    Create a differentiable version of a YOLO model's detection head.

    The standard YOLO detection head uses operations that break gradient flow.
    This function creates a modified version with custom inference logic that
    preserves gradients during bounding box decoding.

    Parameters
    ----------
    model
        Original YOLO DetectionModel instance.

    Returns
    -------
    differentiable_model
        Deep copy of the model with modified detection head that preserves gradients.
    """

    def create_head_inference_hook():
        """
        Create custom inference function for YOLO detection head.

        This hook replaces the standard inference logic with a gradient-preserving
        version that uses clone() to maintain gradient flow during box decoding.

        Returns
        -------
        custom_inference
            Modified inference function with gradient preservation.
        """
        from ultralytics.utils.tal import make_anchors

        def custom_inference(self, x):  # same signature as original
            shape = x[0].shape  # BCHW
            x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

            if self.format != "imx" and (self.dynamic or self.shape != shape):
                self.anchors, self.strides = (x.transpose(0, 1)
                                              for x in make_anchors(x, self.stride, 0.5))
                self.shape = shape

            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
            dfl_results = self.dfl(box)

            # Utilisation de clone() pour préserver les gradients
            dbox = self.decode_bboxes(dfl_results, self.anchors.unsqueeze(
                0).clone()) * self.strides.clone()  # <-- ici

            return torch.cat((dbox, cls.sigmoid()), 1)

        return custom_inference

     # Create a copy of the model
    differentiable_model = copy.deepcopy(model)

    head_detect = list(differentiable_model.children())[0][-1]
    if head_detect.training:
        print("head_detect was in training mode, switching to eval mode")
        head_detect.eval()
        head_detect.training = False

    hook_func = create_head_inference_hook()
    # original_head_inference = head_detect._inference
    # print("original_head_inference replaced by custom hook")
    # replace _inference method of head_detect
    head_detect._inference = types.MethodType(hook_func, head_detect)
    head_detect.iscustom = True

    return differentiable_model


class YoloExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for YOLO models.

    This class provides methods to construct a TorchLatentExtractor specifically
    configured for YOLO (You Only Look Once) object detection models. It allows
    splitting the model at any layer to extract intermediate features.

    YOLO Architecture Overview
    ----------
    Backbone (sequential 0->9):
        Input -> Layer 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9

    Three Resolution Paths (with backbone layers 0-9), concatenation nodes are in parentheses:
        High-resolution path (P3 - small objects, 80x80):
        10 -> (11) -> 12 -> 13 -> (14) -> 15
        Mid-resolution path (P4 - medium objects, 40x40):
        10 -> (11) -> 12 -> 13 -> (14) -> 15 -> 16 -> (17) -> 18
        Low-resolution path (P5 - large objects, 20x20):
        10 -> (11) -> 12 -> 13 -> (14) -> 15 -> 16 -> (17) -> 18 -> 19 -> (20) -> 21
    """

    @classmethod
    def build(
            cls,
            model: ultralytics.nn.tasks.DetectionModel,
            extraction_layer: int,
            batch_size: int = 1) -> 'TorchLatentExtractor':
        """
        Build a LatentExtractor for a YOLO model.

        This method creates custom g and h functions that split the model's forward pass
        at a specified layer: g runs layers up to extraction_layer, and h runs remaining layers.
        The model is modified to preserve gradients through the detection head.

        Parameters
        ----------
        model
            Ultralytics YOLO DetectionModel instance.
        extraction_layer
            Index of the layer where the model should be split. Must not be a layer
            that only takes input from the previous layer (f != -1).
        batch_size
            Batch size for processing. Default is 1.

        Returns
        -------
        latent_extractor
            Configured TorchLatentExtractor instance for the YOLO model.

        Raises
        ------
        ValueError
            If extraction_layer is out of valid range or points to an invalid layer type.
        """

        def g(self, x) -> LatentDataYolo:
            y = []  # outputs
            for m in self.model[:extraction_layer]:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j]
                                                             for j in m.f]  # from earlier layers
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            return LatentDataYolo(x, y)

        def h(self, latent_data: LatentDataYolo) -> Tensor:

            x, y = latent_data.x, latent_data.y
            # to avoid in-place modifications
            # x = x.clone()
            y = [yi.clone() if yi is not None else None for yi in y]

            for m in self.model[extraction_layer:]:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j]
                                                             for j in m.f]  # from earlier layers
                x = m(x)  # run
                y.append(x if m.i in self.save else None)
            return x

        # check on extraction_layer value
        if extraction_layer < 0 or extraction_layer >= len(model.model):
            raise ValueError(f"extraction_layer must be between 0 and {len(model.model)-1}")
        if model.model[extraction_layer].f != -1:
            raise ValueError(
                f"extraction_layer must not be a layer that takes input only from the previous layer (but here f={model.model[extraction_layer].f})")

        if model.training:
            print("model was in training mode, switching to eval mode")
            model.eval()
            model.training = False

        differentiable_model = make_head_differentiable(model)

        differentiable_model.g = types.MethodType(g, differentiable_model)
        differentiable_model.h = types.MethodType(h, differentiable_model)

        processed_formatter = YoloRawBoxFormatter()
        latent_extractor = TorchLatentExtractor(
            differentiable_model,
            differentiable_model.g,
            differentiable_model.h,
            latent_data_class=LatentDataYolo,
            output_formatter=processed_formatter,
            batch_size=batch_size)
        return latent_extractor
