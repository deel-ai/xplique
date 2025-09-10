from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from xplique.utils_functions.object_detection.common.box_manager import BoxFormat, BoxType
from xplique.utils_functions.object_detection.common.box_formatter import XpliqueBoxFormatter
from xplique.utils_functions.object_detection.torch.box_manager import TorchBoxCoordinatesTranslator

class NBCTensor(torch.Tensor):
    # Tensor with shape (N, B, C) where
    # (- N is the batch size)
    # - B is the number of boxes
    # - C is the encoding of a bounding box prediction, C = 4 + 1 + nb_classes
    # - - 4 coordinates
    # - - 1 objectness
    # - - nb_classes as soft class predictions or 1 hot encoded class predictions)

    def __format__(self, format_spec):
        if self.numel() == 1:
            scalar_value = self.item()
            return format(scalar_value, format_spec)
        return super().__format__(format_spec)

    def boxes(self, batch_index: int = 0) -> torch.Tensor:
        return self[batch_index, :, :4]

    def scores(self, batch_index: int = 0) -> torch.Tensor:
        return self[batch_index, :, 4]

    def probas(self, batch_index: int = 0) -> torch.Tensor:
        return self[batch_index, :, 5:]

    def filter(self, batch_index, class_id: int = None, accuracy: float = None) -> 'NBCTensor':
        if class_id is None and accuracy is None:
            return self
        probas = self.probas(batch_index=batch_index)
        class_ids = probas.argmax(dim=-1)
        scores = self.scores(batch_index=batch_index)
        if class_id is not None and accuracy is not None:
            keep = (class_ids == class_id) & (scores > accuracy)
        elif class_id is None:
            keep = scores > accuracy
        else:
            keep = class_ids == class_id
        return self[batch_index, keep].unsqueeze(0)
        
class TorchXpliqueBoxFormatter(XpliqueBoxFormatter, nn.Module, ABC):
    
    def __init__(self,
                 input_box_type: BoxType,
                 output_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True)):
        super().__init__(input_box_type=input_box_type, output_box_type=output_box_type)
        self.box_translator = TorchBoxCoordinatesTranslator(self.input_box_type, self.output_box_type)
                                                   
    @abstractmethod
    def forward(self, predictions):
        raise NotImplementedError("This method should be implemented in the subclass")

    # needs boxes, scores, probas
    def format_predictions(self, predictions) -> NBCTensor:
        boxes = predictions['boxes']
        boxes = self.box_translator.translate(boxes)
        probas = predictions['probas']
        scores = predictions['scores']
        return NBCTensor(torch.cat([boxes, # boxes coordinates
                          scores, # detection probability
                          probas], # class logits predictions for the given box
                         dim=1))

#########################################

class DetrBoxFormatter(TorchXpliqueBoxFormatter):

    def __init__(self):
        super().__init__(input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True))

    def forward(self, predictions):
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
        result = torch.stack(results, dim=0)
        return result
    

class TorchvisionBoxFormatter(TorchXpliqueBoxFormatter):

    def __init__(self, nb_classes: int):
        super().__init__(input_box_type = BoxType(BoxFormat.XYXY, is_normalized=True))
        self.nb_classes = nb_classes
        
    def forward(self, prediction):
        # results = []
        # for prediction in predictions:
        prediction["scores"] = prediction["scores"].unsqueeze(dim=1)
        labels_one_hot = F.one_hot(prediction["labels"], num_classes=self.nb_classes).to(prediction["scores"].device)
        prediction["probas"] = labels_one_hot
        formatted = self.format_predictions(prediction)
        # results.append(formatted)
        # result = torch.stack(results, dim=0)
        # return result
        return formatted

class YoloBoxFormatter(TorchXpliqueBoxFormatter):

    def __init__(self):
        super().__init__(input_box_type = BoxType(BoxFormat.XYXY, is_normalized=True))

    # def forward_perd_per_pred(self, predictions):
    #     device = predictions.boxes.cls.device
    #     num_classes=len(predictions.names)
    #     results = []
    #     for prediction in predictions:
    #         classes_id = prediction.boxes.cls.long()
    #         labels_one_hot = F.one_hot(classes_id, num_classes).to(device)
    #         pred_dict = {
    #             'boxes': prediction.boxes.xyxy,
    #             'scores': prediction.boxes.conf.unsqueeze(dim=1),
    #             'probas': labels_one_hot
    #         }
    #         formatted = self.format_predictions(pred_dict) # needs boxes, scores, probas
    #         results.append(formatted)
    #     result = torch.cat(results, dim=0)
    #     return result
    
    def forward(self, predictions):
        device = predictions.boxes.cls.device
        num_classes=len(predictions.names)
        classes_id = predictions.boxes.cls.long()
        labels_one_hot = F.one_hot(classes_id, num_classes).to(device)
        pred_dict = {
            'boxes': predictions.boxes.xyxy,
            'scores': predictions.boxes.conf.unsqueeze(dim=1),
            'probas': labels_one_hot
        }
        formatted = self.format_predictions(pred_dict) # needs boxes, scores, probas
        return formatted
