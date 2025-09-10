
from abc import ABC
import torch
from xplique.utils_functions.object_detection.torch.box_formatter import TorchXpliqueBoxFormatter, YoloBoxFormatter, TorchvisionBoxFormatter

class TorchBoxesModelWrapper(torch.nn.Module, ABC):
    def __init__(self, model, box_formatter: TorchXpliqueBoxFormatter):
        super().__init__()
        self.model = model.eval()
        self.box_formatter = box_formatter

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        predictions = self.model(x)
        return torch.stack([self.box_formatter(pred) for pred in predictions], dim=0)


class YoloBoxesModelWrapper(TorchBoxesModelWrapper):
    def __init__(self, model):
        box_formatter = YoloBoxFormatter()
        super().__init__(model, box_formatter=box_formatter)


class TorchvisionBoxesModelWrapper(TorchBoxesModelWrapper):
    def __init__(self, model, nb_classes: int):
        box_formatter = TorchvisionBoxFormatter(nb_classes=nb_classes)
        super().__init__(model, box_formatter=box_formatter)

# class TorchvisionBoxesModelWrapper(TorchBoxesModelWrapper):
#     def __init__(self, model, nb_classes: int):
#         super().__init__()
#         self.nb_classes = nb_classes

#     def format_predictions(self, predictions):
#         # format prediction for them to match Xplique object detection operator
#         # a single tensor of shape (nb_boxes, 4 + 1 + nb_classes)
#         # box coordinates defined by (x1, y1, x2, y2) respectively (left, bottom, right, top).

#         # translate the labels to one-hot encoded vectors
#         device = predictions["boxes"].device
#         labels_one_hot = F.one_hot(predictions["labels"], num_classes=self.nb_classes).to(device)
#         return torch.cat([predictions["boxes"],
#                           predictions["scores"].unsqueeze(dim=1),
#                           labels_one_hot],
#                          dim=1)
