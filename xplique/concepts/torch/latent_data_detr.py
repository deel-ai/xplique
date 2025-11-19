import types
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torchvision
from torch import Tensor

from xplique.utils_functions.object_detection.torch.box_formatter import (
    DetrBoxFormatter,
)

from ..latent_extractor import LatentData, LatentExtractorBuilder
from .latent_extractor import TorchLatentExtractor


class NestedTensor(object):
    """
    Container for tensors with associated masks for handling variable-sized inputs.

    This class wraps tensors and their corresponding masks, enabling operations
    on batched inputs with different sizes (common in DETR models).

    Attributes
    ----------
    tensors
        The main tensor data.
    mask
        Optional boolean mask indicating valid regions (False) and padding (True).
    """

    def __init__(self, tensors: torch.Tensor, mask: Optional[Tensor]):
        """
        Initialize a NestedTensor with tensors and an optional mask.

        Parameters
        ----------
        tensors
            The main tensor data.
        mask
            Optional boolean mask for valid regions.
        """
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device) -> 'NestedTensor':
        """
        Move tensors and mask to the specified device.

        Parameters
        ----------
        device
            Target device (e.g., 'cuda', 'cpu').

        Returns
        -------
        nested_tensor
            New NestedTensor instance on the target device.
        """
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self) -> tuple:
        """
        Decompose the NestedTensor into separate tensors and mask.

        Returns
        -------
        tensors
            The main tensor data.
        mask
            The boolean mask.
        """
        return self.tensors, self.mask

    def __repr__(self) -> str:
        """
        Return string representation of the tensors.

        Returns
        -------
        representation
            String representation of the tensor data.
        """
        return str(self.tensors)


class LatentDataDetr(LatentData):
    """
    Stores latent representations (features and positional encodings) from DETR's backbone.

    This class encapsulates the multi-scale features and positional encodings extracted
    from the DETR (Detection Transformer) backbone network. Features are stored as
    NestedTensor objects to handle variable-sized inputs.

    Attributes
    ----------
    features
        List of NestedTensor objects containing backbone feature maps at different scales.
    pos
        List of positional encoding tensors corresponding to each feature scale.
    """

    def __init__(self, features: List, pos: List[torch.Tensor]):
        """
        Initialize DETR latent data with features and positional encodings.

        Parameters
        ----------
        features
            List of NestedTensor objects from the backbone.
        pos
            List of positional encoding tensors.
        """
        self.features = features
        self.pos = pos

    def pdim(self) -> None:
        """
        Print dimensional information about stored features and positional encodings.

        Displays the shapes of the first feature NestedTensor (both tensors and mask)
        and the first positional encoding tensor.
        """
        # Print the shapes of the tensors inside the NestedTensor and pos
        print(f"Features: list of NestedTensor of len:", len(self.features))
        print("\tFeatures[0].tensors shape:", self.features[0].tensors.shape)
        print("\tFeatures[0].mask shape:", self.features[0].mask.shape)
        print(f"Pos len:", len(self.features))
        print("\tPos[0] shape:", self.pos[0].shape)

    def __len__(self) -> int:
        """
        Return the batch size from the feature tensors.

        Returns
        -------
        batch_size
            Number of samples in the batch.
        """
        return len(self.features[0].tensors)

    def detach(self) -> None:
        """
        Detach all tensors from the computation graph.

        This method detaches features and positional encodings, preventing
        gradient computation through these tensors.
        """
        self.features[0].tensors = self.features[0].tensors.detach()
        self.features[0].mask = self.features[0].mask.detach()
        self.pos[0] = self.pos[0].detach()

    def to(self, device: torch.device) -> 'LatentDataDetr':
        """
        Move all data to the specified device.

        Parameters
        ----------
        device
            Target device (e.g., torch.device('cuda') or torch.device('cpu')).

        Returns
        -------
        latent_data
            New LatentDataDetr instance with data on the target device.
        """
        # Create a new instance with data moved to the specified device
        new_features = [
            NestedTensor(
                tensors=self.features[0].tensors.to(device),
                mask=self.features[0].mask.to(device)
            )
        ]
        new_pos = [self.pos[0].to(device)]
        return LatentDataDetr(new_features, new_pos)

    def get_activations(self, as_numpy: bool = True, keep_gradients: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract the feature tensors as activations.

        Parameters
        ----------
        as_numpy
            If True, convert tensors to numpy arrays. Default is True.
        keep_gradients
            If True, preserve gradient information. Default is False.

        Returns
        -------
        activations
            Feature tensors as numpy array or PyTorch tensor. If 4D (N, C, H, W),
            converted to (N, H, W, C) format for compatibility.
        """
        activations = self.features[0].tensors

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
        Update the feature tensors with new activation values.

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

        self.features[0].tensors = values


class DetrExtractorBuilder(LatentExtractorBuilder):
    """
    Builder for creating LatentExtractor instances for DETR models.

    This class provides methods to construct a TorchLatentExtractor specifically
    configured for DETR (Detection Transformer) object detection models. It defines
    the forward pass split into backbone feature extraction (g) and transformer-based
    prediction (h).
    """

    @classmethod
    def build(cls, model: Callable, device: str = 'cuda',
              batch_size: int = 1) -> 'TorchLatentExtractor':
        """
        Build a LatentExtractor for a DETR model.

        This method creates custom g and h functions that split the model's forward pass:
        g extracts backbone features and positional encodings, and h processes them through
        the transformer and prediction heads.

        Parameters
        ----------
        model
            PyTorch DETR model instance with backbone, transformer, input_proj,
            query_embed, class_embed, and bbox_embed attributes.
        device
            Device to run computations on ('cuda' or 'cpu'). Default is 'cuda'.
        batch_size
            Batch size for processing. Default is 1.

        Returns
        -------
        latent_extractor
            Configured TorchLatentExtractor instance for the DETR model.
        """

        def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
            # print('nested_tensor_from_tensor_list')
            # TODO make this more general
            if tensor_list[0].ndim == 3:
                if torchvision._is_tracing():
                    # nested_tensor_from_tensor_list() does not export well to ONNX
                    # call _onnx_nested_tensor_from_tensor_list() instead
                    return _onnx_nested_tensor_from_tensor_list(tensor_list)

                # TODO make it support different-sized images
                max_size = _max_by_axis([list(img.shape) for img in tensor_list])
                # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
                batch_shape = [len(tensor_list)] + max_size
                b, c, h, w = batch_shape
                dtype = tensor_list[0].dtype
                device_tensor = tensor_list[0].device
                tensor = torch.zeros(batch_shape, dtype=dtype, device=device_tensor)
                mask = torch.ones((b, h, w), dtype=torch.bool, device=device_tensor)

                # ori
                # for img, pad_img, m in zip(tensor_list, tensor, mask):
                #     pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                #     m[: img.shape[1], :img.shape[2]] = False

                # work around for
                # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                # m[: img.shape[1], :img.shape[2]] = False
                # which is not yet supported in onnx
                padded_imgs = []
                padded_masks = []
                for img in tensor_list:
                    padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
                    padded_img = torch.nn.functional.pad(
                        img, (0, padding[2], 0, padding[1], 0, padding[0]))
                    padded_imgs.append(padded_img)

                    m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
                    padded_mask = torch.nn.functional.pad(
                        m, (0, padding[2], 0, padding[1]), "constant", 1)
                    padded_masks.append(padded_mask.to(torch.bool))

                tensor = torch.stack(padded_imgs)
                mask = torch.stack(padded_masks)
            else:
                raise ValueError('not supported')
            return NestedTensor(tensor, mask)

        def _max_by_axis(the_list):
            # type: (List[List[int]]) -> List[int]
            maxes = the_list[0]
            for sublist in the_list[1:]:
                for index, item in enumerate(sublist):
                    maxes[index] = max(maxes[index], item)
            return maxes

        @torch.jit.unused
        def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
            print("_onnx_nested_tensor_from_tensor_list")
            max_size = []
            for i in range(tensor_list[0].dim()):
                max_size_i = torch.max(torch.stack(
                    [img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
                max_size.append(max_size_i)
            max_size = tuple(max_size)

            # work around for
            # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            # m[: img.shape[1], :img.shape[2]] = False
            # which is not yet supported in onnx
            padded_imgs = []
            padded_masks = []
            for img in tensor_list:
                padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
                padded_img = torch.nn.functional.pad(
                    img, (0, padding[2], 0, padding[1], 0, padding[0]))
                padded_imgs.append(padded_img)

                m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
                padded_mask = torch.nn.functional.pad(
                    m, (0, padding[2], 0, padding[1]), "constant", 1)
                padded_masks.append(padded_mask.to(torch.bool))

            tensor = torch.stack(padded_imgs)
            mask = torch.stack(padded_masks)

            return NestedTensor(tensor, mask=mask)

        def g(self, samples) -> LatentDataDetr:
            if isinstance(samples, (list, torch.Tensor)):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.backbone(samples)

            return LatentDataDetr(features, pos)

        def h(self, latent_data: LatentDataDetr):
            features, pos = latent_data.features, latent_data.pos

            src, mask = features[-1].decompose()
            assert mask is not None
            hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            return out

        model.g = types.MethodType(g, model)
        model.h = types.MethodType(h, model)

        processed_formatter = DetrBoxFormatter()
        latent_extractor = TorchLatentExtractor(
            model,
            model.g,
            model.h,
            latent_data_class=LatentDataDetr,
            output_formatter=processed_formatter,
            batch_size=batch_size,
            device=device)
        return latent_extractor
