"""
PyTorch-specific latent extractor implementations
"""

from .holistic_craft import HolisticCraftTorch
from .latent_extractor import TorchLatentData, TorchLatentExtractor

__all__ = [
    'TorchLatentData',
    'TorchLatentExtractor',
    'HolisticCraftTorch',
]
