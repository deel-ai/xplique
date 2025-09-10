from enum import Enum
import torch
import tensorflow as tf
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import logging

class BoxFormat(Enum):
    CXCYWH = "CXCYWH"
    XYWH = "XYWH"
    XYXY = "XYXY"


@dataclass
class BoxType:
    format: BoxFormat
    is_normalized: bool

class BoxManager(ABC):
    pass

class NumpyBoxManager(BoxManager):

    def __init__(self, format: BoxFormat, normalized: bool):
        self.format = format
        self.normalized = normalized

    @staticmethod
    def normalize_boxes(raw_boxes: np.ndarray, image_source_size):
        sx, sy = image_source_size
        normalized_boxes = raw_boxes.copy()
        normalized_boxes[:, [0, 2]] /= sx
        normalized_boxes[:, [1, 3]] /= sy
        return normalized_boxes

    @staticmethod
    def box_cxcywh_to_xyxy(normalized_boxes: np.ndarray):
        x_c = normalized_boxes[:, 0]
        y_c = normalized_boxes[:, 1]
        w   = normalized_boxes[:, 2]
        h   = normalized_boxes[:, 3]
        b = np.stack([
            x_c - 0.5 * w,
            y_c - 0.5 * h,
            x_c + 0.5 * w,
            y_c + 0.5 * h
        ], axis=1)
        return b

    @staticmethod
    def box_xywh_to_xyxy(normalized_boxes: np.ndarray):
        x = normalized_boxes[:, 0]
        y = normalized_boxes[:, 1]
        w = normalized_boxes[:, 2]
        h = normalized_boxes[:, 3]
        b = np.stack([
            x,
            y,
            x + w,
            y + h
        ], axis=1)
        return b

    @staticmethod
    def rescale_bboxes(boxes: np.ndarray, size):
        img_w, img_h = size
        rescaled_boxes = boxes * np.array([img_w, img_h, img_w, img_h])
        return rescaled_boxes

    def resize(
        self,
        raw_boxes: np.ndarray,
        image_source_size,
        image_target_size,
    ):
        # print(f"Resizing boxes from {image_source_size} to {image_target_size}")
        # print(f"Raw boxes: {raw_boxes[:5]}")
        normalized_boxes = raw_boxes

        # print("=== format conversion ===")
        if self.format == BoxFormat.CXCYWH:
            # print("Converting from CXCYWH to XYXY format")
            xyxy_boxes = NumpyBoxManager.box_cxcywh_to_xyxy(normalized_boxes)
        elif self.format == BoxFormat.XYWH:
            # print("Converting from XYWH to XYXY format")
            xyxy_boxes = NumpyBoxManager.box_xywh_to_xyxy(normalized_boxes)
        else:  # XYXY already
            # print("Boxes are already in XYXY format")
            xyxy_boxes = normalized_boxes
        # print(f"XYXY boxes: {xyxy_boxes[:5]}")

        scaled_boxes = xyxy_boxes
        # print(f"Scaled boxes: {scaled_boxes[:5]}")
        return scaled_boxes


# @dataclass
# class NbcDetection:
#     boxes: torch.Tensor
#     scores: torch.Tensor
#     logits: torch.Tensor

# class NbcDetectionFormater:
#     """
#     Format the input boxes and logits to the format (n, nb_boxes, 4+1+nb_classes)
#     """

#     def __init__(self, input_box_type: BoxType, nb_classes: int):
#         self.input_box_type = input_box_type
#         self.nb_classes = nb_classes
#         self.output_box_type = BoxType(format=BoxFormat.XYXY, is_normalized=False)
#         self.translator = TorchBoxCoordinatesTranslator(
#             input_box_type=self.input_box_type,
#             output_box_type=self.output_box_type,
#         )

#     def format_one_annotation(self, box, image_size):
#         xyxy_boxes = self.translator.translate(
#             box=torch.tensor(box).unsqueeze(0), image_size=image_size
#         )
#         return torch.cat(
#             [
#                 xyxy_boxes,  # boxes
#                 torch.tensor([1.0]).unsqueeze(0),  # score
#                 torch.tensor([1.0]).unsqueeze(0),  # logits 1st element, one hot element
#                 torch.zeros((1, self.nb_classes - 1)),
#             ],
#             dim=1,
#         )  # logits other elements one hot encoding (logits)

#     def format(
#         self,
#         boxes: torch.Tensor,
#         images_sizes: torch.Size,
#     ):

#         return torch.stack(
#             [
#                 self.format_one_annotation(ann, image_size)
#                 for (ann, image_size) in zip(boxes, images_sizes)
#             ],
#             dim=0,
#         )

# from matplotlib import pyplot as plt

# def display_image_with_nbc_boxes(image, nbc_annotations):
#     fig, ax = plt.subplots()
#     ax.imshow(image)
#     for ann in nbc_annotations:
#         xmin, ymin, xmax, ymax = ann[:4]
#         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                    fill=False, color='red', linewidth=3))
#         ax.text(xmin, ymin, f"{ann[4]:.2f}", color='red', fontsize=12)