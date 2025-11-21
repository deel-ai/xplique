"""
PyTorch-specific latent extractor implementations
"""

from .factorizer import TorchSklearnNMFFactorizer
from .holistic_craft import HolisticCraftTorch
from .latent_extractor import TorchLatentData, TorchLatentExtractor

__all__ = [
    'TorchLatentData',
    'TorchLatentExtractor',
    'HolisticCraftTorch',
    'TorchSklearnNMFFactorizer',
]
