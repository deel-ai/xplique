from typing import Callable, Optional, Tuple, List
import torch
from torch import Tensor
from .holistic_craft_object_detection import *

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
class LatentDataDetr(LatentData):

    def __init__(self, features: List, pos: List[torch.Tensor]):
        self.features = features
        self.pos = pos

    def pdim(self):
        # Print the shapes of the tensors inside the NestedTensor and pos
        print(f"Features: list of NestedTensor of len:", len(self.features))
        print("\tFeatures[0].tensors shape:", self.features[0].tensors.shape)
        print("\tFeatures[0].mask shape:", self.features[0].mask.shape)
        print(f"Pos len:", len(self.features))
        print("\tPos[0] shape:", self.pos[0].shape)

    def __len__(self) -> int:
        return len(self.features[0].tensors)
    
    def detach(self):
        self.features[0].tensors = self.features[0].tensors.detach()
        self.features[0].mask = self.features[0].mask.detach()
        self.pos[0] = self.pos[0].detach()
    
    def to(self, device: torch.device) -> 'LatentDataDetr':
        # Create a new instance with data moved to the specified device
        new_features = [
            NestedTensor(
                tensors=self.features[0].tensors.to(device),
                mask=self.features[0].mask.to(device)
            )
        ]
        new_pos = [self.pos[0].to(device)]
        return LatentDataDetr(new_features, new_pos)
    
    def get_activations(self):
        
        activations = self.features[0].tensors
        # ici
        is_4d = len(activations.shape) == 4
        if is_4d: 
            # torch -> tensorflow/numpy
            # activations: (N, C, H, W) -> (N, H, W, C)            
            activations = activations.permute(0, 2, 3, 1)
        return activations

    def set_activations(self, values):
        # tensorflow/numpy -> torch
        # activations: (N, H, W, C) -> (N, C, H, W)
        # ici
        values = values.permute(0, 3, 1, 2)
        
        self.features[0].tensors = values

    def __getitem__(self, index):
        if isinstance(index, int):
            index = slice(index, index + 1)

        # Slice the features and pos using the given index
        new_features = [
            NestedTensor(
                tensors=self.features[0].tensors[index],
                mask=self.features[0].mask[index]
            )
        ]
        new_pos = [self.pos[0][index]]

        return LatentDataDetr(new_features, new_pos)
    
    @classmethod
    def aggregate(self, *latent_data_list: 'LatentDataDetr') -> 'LatentDataDetr':
        # Agréger plusieurs LatentDataDetr instances
        combined_tensors = []
        combined_masks = []
        combined_pos = []

        for latent_data_detr in latent_data_list:
            for feature in latent_data_detr.features:
                combined_tensors.append(feature.tensors)
                if feature.mask is not None:
                    combined_masks.append(feature.mask)
            combined_pos.extend(latent_data_detr.pos)

        # Concatenation des tensors et masques
        agg_tensors = torch.cat(combined_tensors, dim=0)
        agg_masks = torch.cat(combined_masks, dim=0) if combined_masks else None
        agg_pos = torch.cat(combined_pos, dim=0)

        # Création du NestedTensor agrégé
        aggregated_features = [NestedTensor(agg_tensors, agg_masks)]
        return LatentDataDetr(aggregated_features, [agg_pos])

# class DetrWrapper(ObjectDetectionModelWrapper):
#     pass
    # def output_predict(self, latent_data: LatentDataDetr, resize = None, no_grad:bool = True) -> List[ObjectDetectionOutput]:
    #     output = self.batch_output_inference(self.latent_to_logit_model, latent_data,
    #                               self.batch_size, resize, device=self.device, no_grad=no_grad)

    #     # adaptation for Detr
    #     dict_list = [dict_to_cpu(boxes_scores_labels) for boxes_scores_labels in output]
    #     # list of dict -> 1 single dict
    #     merged_dict = merge_dicts(dict_list) # dict_keys(['pred_logits', 'pred_boxes'])
        
    #     output_detr = []
    #     for i in range(len(merged_dict['pred_logits'])):
    #         logits = merged_dict['pred_logits'][i]
    #         probas = logits.unsqueeze(0).softmax(-1)[0,:,:-1]
    #         scores, classes = probas.max(-1)
    #         raw_boxes = merged_dict['pred_boxes'][i]
    #         output_detr.append(ObjectDetectionOutput(probas, scores, classes, raw_boxes))
    #     return output_detr
    
    # def probas_predict(self, latent_data: LatentDataDetr, class_id:int, no_grad:bool = True):
    #     context = torch.no_grad() if no_grad else nullcontext()
    #     latent_data = latent_data.to(self.device)
    #     with context:
    #         boxes_scores_labels = self.latent_to_logit_model(latent_data)
        
    #     logits = boxes_scores_labels['pred_logits']
    #     probas = logits.softmax(-1)[:, :, :-1]
    #     probas_for_our_class = probas[:, :, class_id]
    #     return probas_for_our_class

        

import types

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple
from torch import Tensor
import torchvision
from xplique.concepts.latent_extractor import TorchLatentExtractor
from xplique.utils_functions.object_detection.torch.box_formatter import DetrBoxFormatter

def buildTorchDetrLatentExtractor(model: Callable) -> 'TorchLatentExtractor':
    
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
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            
            #ori
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
                padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
                padded_imgs.append(padded_img)

                m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
                padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
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
            max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
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
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)

            m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
            padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
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

    # processed_formatter = DetrBoxFormatter(nb_classes=nb_classes, input_image_size=(640, 462), output_image_size=(640, 462))
    processed_formatter = DetrBoxFormatter()
    latent_extractor = TorchLatentExtractor(model, model.g, model.h, latent_data_class=LatentDataDetr, output_formatter=processed_formatter, batch_size=1)
    return latent_extractor