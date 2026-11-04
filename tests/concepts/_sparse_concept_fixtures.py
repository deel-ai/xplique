"""Small analytical coefficient game shared by the HolisticCraft backend tests."""

import numpy as np

from xplique.attributions import Banzhaf, KernelBanzhaf, SparseHSIC, SparseSobol
from xplique.concepts.craft import Factorization

COEFFICIENTS = np.array(
    [
        [[[1.0, 2.0, 0.0]], [[3.0, 0.0, 0.0]]],
        [[[2.0, 0.0, 1.0]], [[0.0, 0.0, 2.0]]],
    ],
    dtype=np.float32,
)
WEIGHTS = np.array([2.0, -3.0, 4.0], dtype=np.float32)
EXPECTED_BANZHAF = np.array([[8.0, -6.0, 0.0], [8.0, 0.0, 24.0]], dtype=np.float32)
METHODS = [
    (Banzhaf, {"nb_samples": 4}, 4),
    (KernelBanzhaf, {"nb_samples": 4}, 4),
    (SparseSobol, {"nb_design": 4, "mask_distribution": "bernoulli"}, 16),
    (SparseHSIC, {"nb_samples": 16}, 16),
]


class IdentityFactorizer:
    """Encode the fixed coefficients without introducing fitting variability."""

    is_fitted = True

    def __init__(self):
        self.encode_calls = 0

    def encode(self, activations):
        self.encode_calls += 1
        return activations


def fitted_craft(craft_class, extractor, **kwargs):
    """Install a known concept bank while retaining real backend decoder machinery."""
    factorizer = IdentityFactorizer()
    craft = craft_class(extractor, number_of_concepts=3, factorizer=factorizer, **kwargs)
    craft.factorization = Factorization(
        None, 0, None, factorizer, None, np.eye(3, dtype=np.float32)
    )
    return craft
