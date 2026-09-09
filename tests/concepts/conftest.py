"""Shared fixtures for concept tests."""

import numpy as np
import pytest


@pytest.fixture
def identity_factorizer():
    """Return an identity factorizer class for deterministic framework tests."""

    class IdentityFactorizer:
        is_fitted = False
        requires_positive_activations = False

        def fit(self, activations):
            self.is_fitted = True
            return np.eye(2, dtype=np.float32), np.asarray(activations, dtype=np.float32)

        def encode(self, activations):
            return np.asarray(activations, dtype=np.float32)

        def encode_differentiable(self, activations):
            return activations

    return IdentityFactorizer
