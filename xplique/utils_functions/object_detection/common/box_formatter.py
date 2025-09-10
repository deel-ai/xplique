
from typing import Callable
from abc import ABC, abstractmethod
from .box_manager import BoxFormat, BoxType

class XpliqueBoxFormatter(Callable, ABC):
    
    def __init__(self, 
                 input_box_type: BoxType,
                 output_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True)):
        super().__init__()
        self.input_box_type = input_box_type
        self.output_box_type = output_box_type

    def __call__(self, predictions):
        return self.forward(predictions)

    @abstractmethod
    def forward(self, predictions):
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("This method should be implemented in the subclass")

    @abstractmethod
    def format_predictions(self, predictions):
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("This method should be implemented in the subclass")