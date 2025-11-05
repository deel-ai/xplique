"""
PyTorch box formatters for converting object detection model outputs to Xplique format.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import ultralytics
    from ultralytics.engine.results import Results as UltralyticsResults
except ImportError:
    ultralytics = None
    UltralyticsResults = None

from xplique.utils_functions.object_detection.base.box_formatter import (
    BaseBoxFormatter,
)
from xplique.utils_functions.object_detection.base.box_manager import (
    BoxFormat,
    BoxType,
)
from xplique.utils_functions.object_detection.torch.box_manager import (
    TorchBoxCoordinatesTranslator,
)
from xplique.utils_functions.object_detection.torch.multi_box_tensor import (
    MultiBoxTensor,
)


class TorchBaseBoxFormatter(BaseBoxFormatter, nn.Module, ABC):
    """
    Abstract base class for PyTorch-based box formatters.

    This class combines BaseBoxFormatter functionality with PyTorch nn.Module
    capabilities, enabling gradient computation through box formatting operations.
    """

    def __init__(self,
                 input_box_type: BoxType,
                 output_box_type: BoxType = BoxType(
                     BoxFormat.XYXY, is_normalized=True)) -> None:
        """
        Initialize the PyTorch box formatter.

        Parameters
        ----------
        input_box_type
            Format of input bounding boxes (coordinate system and normalization).
        output_box_type
            Desired output format for bounding boxes. Defaults to normalized XYXY.
        """
        super().__init__(input_box_type=input_box_type, output_box_type=output_box_type)
        self.box_translator = TorchBoxCoordinatesTranslator(
            self.input_box_type, self.output_box_type)

    @abstractmethod
    def forward(self, predictions: Any) -> List[MultiBoxTensor]:
        """
        Transform model predictions into Xplique MultiBoxTensor format.

        Parameters
        ----------
        predictions
            Raw predictions from the object detection model.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    # needs boxes, scores, probas
    def format_predictions(
            self, predictions: Dict[str, torch.Tensor]) -> MultiBoxTensor:
        """
        Convert prediction dictionary to MultiBoxTensor format.

        Parameters
        ----------
        predictions
            Dictionary containing 'boxes', 'scores', and 'probas' keys.

        Returns
        -------
        formatted_tensor
            MultiBoxTensor with concatenated boxes, scores, and class probabilities.
        """
        boxes = predictions['boxes']
        boxes = self.box_translator.translate(boxes)
        probas = predictions['probas']
        scores = predictions['scores']
        return MultiBoxTensor(torch.cat([boxes,  # boxes coordinates
                                         scores,  # detection probability
                                         probas],  # class logits predictions for the given box
                                        dim=1))


class DetrBoxFormatter(TorchBaseBoxFormatter):
    """
    Box formatter for DETR (DEtection TRansformer) object detection models.

    DETR models output predictions in CXCYWH (center-x, center-y, width, height)
    normalized format with separate logits and box coordinates.
    """

    def __init__(self) -> None:
        """
        Initialize DETR box formatter with CXCYWH normalized input format.
        """
        super().__init__(input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True))

    def forward(self, predictions) -> List[MultiBoxTensor]:
        """
        Format DETR model predictions.

        Parameters
        ----------
        predictions
            Dictionary with 'pred_logits' and 'pred_boxes' keys containing
            batched predictions from DETR model.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        results = []
        for logits, boxes in zip(predictions['pred_logits'], predictions['pred_boxes']):
            probas = logits.softmax(-1)[:, :-1]
            scores = probas.max(-1).values.unsqueeze(1)
            pred_dict = {
                'logits': logits,
                'boxes': boxes,
                'scores': scores,
                'probas': probas
            }
            formatted = self.format_predictions(pred_dict)
            results.append(formatted)
        return results


class TorchvisionBoxFormatter(TorchBaseBoxFormatter):
    """
    Box formatter for Torchvision object detection models (FCOS, RetinaNet, etc.).

    Torchvision models output predictions in XYXY normalized format with discrete
    class labels that need to be converted to one-hot encoded probabilities.
    """

    def __init__(self, nb_classes: int) -> None:
        """
        Initialize Torchvision box formatter.

        Parameters
        ----------
        nb_classes
            Total number of object classes in the detection model.
        """
        super().__init__(input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True))
        self.nb_classes = nb_classes

    def forward(self, predictions) -> List[MultiBoxTensor]:
        """
        Format Torchvision model predictions.

        Parameters
        ----------
        predictions
            List of dictionaries, one per image, each containing 'boxes',
            'scores', and 'labels' keys.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        results = []
        for prediction in predictions:
            prediction["scores"] = prediction["scores"].unsqueeze(dim=1)
            labels_one_hot = F.one_hot(
                prediction["labels"],
                num_classes=self.nb_classes).to(
                prediction["scores"].device)
            prediction["probas"] = labels_one_hot
            formatted = self.format_predictions(prediction)
            results.append(formatted)
        return results


class YoloResultBoxFormatter(TorchBaseBoxFormatter):
    """
    Box formatter for YOLO (Ultralytics) Results objects.

    Formats predictions from Ultralytics YOLO models that return Results objects
    containing boxes, confidence scores, and class predictions.
    """

    def __init__(self) -> None:
        """
        Initialize YOLO result formatter with XYXY normalized input format.
        """
        super().__init__(input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True))

    def forward(self, predictions: list) -> List[MultiBoxTensor]:
        """
        Format YOLO Results objects.

        Parameters
        ----------
        predictions
            List of ultralytics.engine.results.Results objects from YOLO model.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        assert (isinstance(predictions, list))
        assert (isinstance(predictions[0], ultralytics.engine.results.Results))

        formatted_preds = []
        for result in predictions:
            device = result.boxes.cls.device
            num_classes = len(result.names)
            classes_id = result.boxes.cls.long()
            labels_one_hot = F.one_hot(classes_id, num_classes).to(device)
            pred_dict = {
                'boxes': result.boxes.xyxy,
                'scores': result.boxes.conf.unsqueeze(dim=1),
                'probas': labels_one_hot
            }
            formatted = self.format_predictions(pred_dict)  # needs boxes, scores, probas
            formatted_preds.append(formatted)
        return formatted_preds


class YoloRawBoxFormatter(TorchBaseBoxFormatter):
    """
    Box formatter for raw YOLO model outputs (tensor format).

    Processes raw YOLO predictions in CXCYWH normalized format before they are
    converted to Results objects. Handles tuple output with detection tensors.
    """

    def __init__(self) -> None:
        """
        Initialize raw YOLO formatter with CXCYWH normalized input format.
        """
        super().__init__(input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True))

    def forward(self, predictions: torch.Tensor) -> List[MultiBoxTensor]:
        """
        Format raw YOLO tensor predictions.

        Parameters
        ----------
        predictions
            Tuple containing (detection_tensor, auxiliary_data) where
            detection_tensor has shape (batch, features, num_boxes) with features
            encoding boxes (4 values) and class probabilities.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        # check the raw structure of the YOLO predictions
        assert isinstance(
            predictions, tuple), f"predictions should be a tuple, got {type(predictions)}"
        assert len(predictions) == 2
        assert isinstance(
            predictions[0], torch.Tensor), f"predictions[0] should be a torch.Tensor, got {type(predictions[0])}"
        assert isinstance(
            predictions[1], list), f"predictions[1] should be a list, got {type(predictions[1])}"
        assert len(predictions[1]) == 3
        nb_preds = len(predictions[0])

        formatted_preds = []
        for i in range(nb_preds):
            pred = predictions[0][i]
            boxes = pred.squeeze().permute(1, 0)[:, :4]
            probas = pred.squeeze().permute(1, 0)[:, 4:]  # sigmoid result
            scores = probas.max(-1).values.unsqueeze(1)
            pred_dict = {
                'boxes': boxes,
                'scores': scores,
                'probas': probas
            }
            formatted = self.format_predictions(pred_dict)  # needs boxes, scores, probas
            formatted_preds.append(formatted)
        return formatted_preds
