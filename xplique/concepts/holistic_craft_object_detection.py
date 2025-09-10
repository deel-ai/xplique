from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np
from sklearn.exceptions import NotFittedError

from .latent_extractor import LatentData, LatentExtractor

def show_ax(img, ax, **kwargs):
    img = np.array(img)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    ax.imshow(img, **kwargs)
    ax.axis("off")

class HolisticCraftObjectDetection(ABC):
    
    def __init__(
        self,
        latent_extractor: LatentExtractor,
        number_of_concepts: int = 20
    ):
        self.latent_extractor = latent_extractor
        self.number_of_concepts = number_of_concepts
        self.batch_size = latent_extractor.batch_size
        self.factorization = None
        
    def fit(
        self, inputs, class_id: int = 0, max_iter: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def transform(
        self, inputs, resize=None
    ) -> Tuple[LatentData, np.ndarray]:
        raise NotImplementedError()

    def transform_latent(self, latent_data: LatentData):
        raise NotImplementedError()
    
    def check_if_fitted(self):
        """Checks if the factorization model has been fitted to input data.

        Raises
        ------
        NotFittedError
            If the factorization model has not been fitted to input data.
        """
        if self.factorization is None:
            raise NotFittedError(
                "The factorization model has not been fitted to input data yet."
            )

    @abstractmethod
    def transform(self, inputs, resize=None) -> np.ndarray:
        return NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def transform_latent(self, latent_data: LatentData) -> np.ndarray:
        return NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def encode(self, inputs, resize=None) -> np.ndarray:
        return NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def decode(self, latent_data: LatentData, coeffs_u: np.ndarray):
        return NotImplementedError("This method should be implemented in subclasses.")