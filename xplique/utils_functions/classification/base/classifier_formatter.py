"""
Base class for formatting classification model predictions.
"""
from abc import ABC, abstractmethod


class BaseClassifierFormatter(ABC):
    """
    Abstract base class for formatting classification predictions.

    This class provides a common interface for converting model predictions into
    a standardized format for use with Xplique attribution methods.
    """

    def __call__(self, predictions):
        """
        Format predictions by calling the forward method.

        Parameters
        ----------
        predictions
            Raw model predictions to format.

        Returns
        -------
        formatted_predictions
            Formatted predictions in standardized format.
        """
        return self.forward(predictions)

    @abstractmethod
    def forward(self, predictions):
        """
        Abstract method to format model predictions.

        Subclasses must implement this method to provide framework-specific
        formatting logic.

        Parameters
        ----------
        predictions
            Raw model predictions to format.

        Returns
        -------
        formatted_predictions
            Formatted predictions in standardized format.
        """
        raise NotImplementedError("This method should be implemented in the subclass")
