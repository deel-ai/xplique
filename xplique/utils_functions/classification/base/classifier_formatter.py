from abc import ABC, abstractmethod


class BaseClassifierFormatter(ABC):

    def __call__(self, predictions):
        return self.forward(predictions)

    @abstractmethod
    def forward(self, predictions):
        raise NotImplementedError("This method should be implemented in the subclass")
