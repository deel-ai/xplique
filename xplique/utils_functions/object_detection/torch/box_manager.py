
import torch
from xplique.utils_functions.object_detection.common.box_manager import BoxManager, BoxFormat, BoxType

class TorchBoxManager(BoxManager):

    def __init__(self, format: BoxFormat, normalized: bool):
        self.format = format
        self.normalized = normalized

    @staticmethod
    def normalize_boxes(raw_boxes: torch.Tensor, image_source_size: torch.Size):
        [sx, sy] = image_source_size
        normalized_boxes = raw_boxes.clone()
        normalized_boxes[:, [0, 2]] /= sx
        normalized_boxes[:, [1, 3]] /= sy
        return normalized_boxes

    @staticmethod
    def box_cxcywh_to_xyxy(normalized_boxes: torch.Tensor):
        x_c, y_c, w, h = normalized_boxes.unbind(1) # extract the columns
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @staticmethod
    def box_xywh_to_xyxy(normalized_boxes: torch.Tensor):
        x, y, w, h = normalized_boxes.unbind(1) # extract the columns
        b = [x, y, x + w, y + h]
        return torch.stack(b, dim=1)

    @staticmethod
    def rescale_bboxes(boxes: torch.Tensor, size: torch.Size):
        img_w, img_h = size
        rescaled_boxes = boxes * torch.tensor(
            [img_w, img_h, img_w, img_h], device=boxes.device
        )  # , dtype=torch.float32, device=boxes.device)
        return rescaled_boxes

    # Resize boxes in the input source size dimension, to output targe size dimension.
    # The input box format can be either CXCYWH or XYXY, the output format is XYXY (always), for Craft compatibility.

    def resize(
        self,
        raw_boxes: torch.Tensor,
        image_source_size: torch.Size,
        image_target_size: torch.Size,
    ):
        # print(f"Resizing boxes from {image_source_size} to {image_target_size}")
        # print(f"Raw boxes: {raw_boxes[:5]}")
        # print("=== normalization ===")
        # @TODO: normalization issue
        normalized_boxes = raw_boxes

        # print("=== format conversion ===")
        if self.format == BoxFormat.CXCYWH:
            # print("Converting from CXCYWH to XYXY format")
            xyxy_boxes = TorchBoxManager.box_cxcywh_to_xyxy(normalized_boxes)
        elif self.format == BoxFormat.XYWH:
            # print("Converting from XYWH to XYXY format")
            xyxy_boxes = TorchBoxManager.box_xywh_to_xyxy(normalized_boxes)
        else:  # XYXY already
            # print("Boxes are already in XYXY format")
            xyxy_boxes = normalized_boxes

        scaled_boxes = xyxy_boxes
        # @TODO: normalization issue
        return scaled_boxes


class TorchBoxCoordinatesTranslator:
    def __init__(self, input_box_type: BoxType, output_box_type: BoxType):
        self.input_box_type = input_box_type
        self.output_box_type = output_box_type

    def translate(self, box: torch.Tensor, image_size: torch.Size = None): # TODO: input/output image size
        if not isinstance(box, torch.Tensor):
            box = torch.tensor(box)

        # normalize the input box if needed
        if not self.input_box_type.is_normalized:
            if image_size is None:
                raise ValueError("Image size must be provided for non-normalized boxes.")
            box = TorchBoxManager.normalize_boxes(box, image_size)

        # convert the input box to XYXY format if needed
        if self.input_box_type.format.value == BoxFormat.CXCYWH.value:
            box = TorchBoxManager.box_cxcywh_to_xyxy(box)
        elif self.input_box_type.format.value == BoxFormat.XYWH.value:
            box = TorchBoxManager.box_xywh_to_xyxy(box)

        # now convert to the output format
        if self.output_box_type.format.value == BoxFormat.CXCYWH.value:
            box = TorchBoxManager.box_xyxy_to_cxcywh(box)
        elif self.output_box_type.format.value == BoxFormat.XYWH.value:
            box = TorchBoxManager.box_xyxy_to_xywh(box) # TODO: add this method

        # rescale the box to the output image size if needed
        if not self.output_box_type.is_normalized:
            box = TorchBoxManager.rescale_bboxes(box, image_size)

        return box


# Filter the boxes on their class id and accuracy, set the filterd boxes to zero to keep the same dimension.
# TODO: move to NBCTensor class
def filter_boxes_same_dim(model_output_nbc: list, class_id: int, accuracy: float=0.55) -> list:
    filtered = []
    for nbc_annotations in model_output_nbc:
        probas = nbc_annotations[:, 5:]
        class_ids = probas.argmax(axis=1)
        scores = nbc_annotations[:, 4]
        keep = (class_ids == class_id) & (scores > accuracy)
        # replace nbc_annotations where keep is False with zeros
        nbc_annotations[~keep, :] = 0
        filtered.append(nbc_annotations)
    return filtered

# TODO: move to NBCTensor class
def filter_boxes(model_output_nbc: list, class_id: int = None, accuracy: float = None) -> list:
    if class_id is None and accuracy is None:
        return model_output_nbc
    filtered = []
    for nbc_annotations in model_output_nbc:
        probas = nbc_annotations[:, 5:]
        class_ids = probas.argmax(dim=1)
        scores = nbc_annotations[:, 4]
        if class_id is None:
            keep = scores > accuracy
        elif accuracy is None:
            keep = class_ids == class_id
        else:
            keep = (class_ids == class_id) & (scores > accuracy)
        filtered.append(nbc_annotations[keep, :])
    return filtered
