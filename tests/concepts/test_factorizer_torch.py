"""Tests for PyTorch differentiable NMF factorizer encoding."""

import numpy as np
import pytest
import torch

from xplique.concepts.torch.factorizer import TorchSklearnNMFFactorizer


def _make_non_negative_data(seed, shape, minimum=0.05):
    """Generate deterministic non-negative test data."""
    rng = np.random.default_rng(seed)
    return rng.uniform(minimum, 1.0, size=shape).astype(np.float32)


def _nmf_encoding_objective(activations, coefficients, concept_bank, alpha_w, l1_ratio):
    """Compute the fixed-dictionary NMF encoding objective."""
    n_features = activations.shape[1]
    l1_reg = n_features * alpha_w * l1_ratio
    l2_reg = n_features * alpha_w * (1.0 - l1_ratio)

    reconstruction = coefficients @ concept_bank
    data_fit = 0.5 * np.square(activations - reconstruction).sum()
    l1_penalty = l1_reg * np.abs(coefficients).sum()
    l2_penalty = 0.5 * l2_reg * np.square(coefficients).sum()

    return data_fit + l1_penalty + l2_penalty


@pytest.mark.parametrize("alpha_w,l1_ratio", [(0.0, 0.0), (0.05, 0.3)])
@pytest.mark.parametrize("tol", [1e-6, 0.0])
def test_torch_encode_differentiable_matches_sklearn_transform(alpha_w, l1_ratio, tol):
    """The differentiable Torch solver should match sklearn's fixed-dictionary transform."""
    train = _make_non_negative_data(0, (80, 12))
    test = _make_non_negative_data(1, (20, 12))

    factorizer = TorchSklearnNMFFactorizer(
        n_components=5,
        alpha_W=alpha_w,
        l1_ratio=l1_ratio,
        max_iter=1000,
        tol=tol,
        random_state=0,
    )
    factorizer.fit(train)

    expected = factorizer.encode(test)
    actual = factorizer.encode_differentiable(torch.tensor(test))
    actual_np = actual.detach().cpu().numpy()
    concept_bank = factorizer.get_concept_bank()

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == expected.shape
    assert np.all(actual_np >= -1e-6)

    expected_objective = _nmf_encoding_objective(
        test, expected, concept_bank, alpha_w=alpha_w, l1_ratio=l1_ratio
    )
    actual_objective = _nmf_encoding_objective(
        test, actual_np, concept_bank, alpha_w=alpha_w, l1_ratio=l1_ratio
    )
    assert actual_objective <= expected_objective * 1.05 + 1e-5

    expected_reconstruction = expected @ concept_bank
    actual_reconstruction = actual_np @ concept_bank
    reconstruction_error = np.linalg.norm(actual_reconstruction - expected_reconstruction)
    reconstruction_norm = np.linalg.norm(expected_reconstruction)
    assert reconstruction_error / (reconstruction_norm + 1e-8) < 0.05


@pytest.mark.parametrize("tol", [1e-6, 0.0])
def test_torch_encode_differentiable_preserves_gradients(tol):
    """The differentiable Torch solver should backpropagate through activations."""
    train = _make_non_negative_data(2, (60, 10))
    test = torch.tensor(_make_non_negative_data(3, (16, 10)), requires_grad=True)

    factorizer = TorchSklearnNMFFactorizer(
        n_components=4,
        alpha_W=1e-2,
        max_iter=500,
        tol=tol,
        random_state=0,
    )
    factorizer.fit(train)

    coefficients = factorizer.encode_differentiable(test)
    loss = torch.sum(coefficients)
    loss.backward()

    gradients = test.grad

    assert gradients is not None
    assert torch.all(torch.isfinite(gradients)).item()
    assert torch.sum(torch.abs(gradients)).item() > 0


def test_torch_encode_differentiable_rejects_negative_activations():
    """NMF encoding should fail when differentiable inputs are negative."""
    train = _make_non_negative_data(6, (30, 6), minimum=0.1)
    test = torch.tensor(-_make_non_negative_data(7, (10, 6), minimum=0.1))

    factorizer = TorchSklearnNMFFactorizer(n_components=3, max_iter=200, random_state=0)
    factorizer.fit(train)

    with pytest.raises(ValueError, match="NMF requires non-negative activations"):
        factorizer.encode_differentiable(test)


def test_torch_encode_differentiable_rejects_unsupported_beta_loss():
    """Unsupported sklearn NMF configurations should fail explicitly."""
    train = _make_non_negative_data(4, (40, 8), minimum=0.1)
    test = torch.tensor(_make_non_negative_data(5, (12, 8), minimum=0.1))

    factorizer = TorchSklearnNMFFactorizer(
        n_components=3,
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=300,
        random_state=0,
    )
    factorizer.fit(train)

    with pytest.raises(NotImplementedError):
        factorizer.encode_differentiable(test)
