"""End-to-end concept-channel attributions through the PyTorch HolisticCraft decoder."""

from contextlib import contextmanager

import numpy as np
import pytest
import tensorflow as tf
import torch

from xplique import Tasks
from xplique.attributions import Banzhaf
from xplique.concepts import HolisticCraftTorch, PartialExplainer
from xplique.concepts.latent_extractor import LatentData
from xplique.utils_functions.classification.torch.classifier_tensor import TorchClassifierTensor

from ._sparse_concept_fixtures import (
    COEFFICIENTS,
    EXPECTED_BANZHAF,
    METHODS,
    WEIGHTS,
    fitted_craft,
)


class _LatentData(LatentData):
    def __init__(self, activations, gain):
        self.activations = torch.as_tensor(activations)
        self.gain = gain

    def get_activations(self, as_numpy=True, keep_gradients=False):
        if as_numpy:
            return self.activations.detach().cpu().numpy()
        return self.activations

    def set_activations(self, values):
        self.activations = values


class _Extractor:
    batch_size = 3
    device = "cpu"

    @contextmanager
    def temporary_force_batch_size(self, batch_size):
        previous = self.batch_size
        self.batch_size = batch_size
        try:
            yield
        finally:
            self.batch_size = previous

    def input_to_latent_generator(self, inputs, resize=None, keep_gradients=False):
        for index, coefficients in enumerate(inputs):
            yield _LatentData(coefficients[None], gain=index + 1)

    def latent_to_logit(self, latent_data):
        totals = latent_data.activations.sum(dim=(1, 2))
        score = latent_data.gain * (totals * torch.as_tensor(WEIGHTS)).sum(dim=-1)
        return TorchClassifierTensor.from_predictions(
            torch.stack([score, torch.ones_like(score) * 100.0], dim=-1)
        )


@pytest.mark.parametrize("explainer_class,kwargs,evaluations", METHODS)
def test_partial_explainer_with_wrapped_torch_decoder(explainer_class, kwargs, evaluations):
    """Wrapped decoder receives batched masks and fixed targets without re-encoding."""
    extractor = _Extractor()
    craft = fitted_craft(HolisticCraftTorch, extractor, device="cpu")
    batch_sizes = []

    def score(model, inputs, targets):
        batch_sizes.append(int(inputs.shape[0]))
        np.testing.assert_array_equal(targets, np.tile([[1.0, 0.0]], (len(inputs), 1)))
        return tf.reduce_sum(model(inputs) * targets, axis=-1)

    previous_eager = tf.config.functions_run_eagerly()
    try:
        explanation = craft.compute_explanation_per_concept(
            COEFFICIENTS,
            PartialExplainer(explainer_class, operator=score, seed=9, **kwargs),
            class_id=0,
        )
        assert batch_sizes == ([3] * (evaluations // 3) + [1]) * 2
        assert craft.factorizer.encode_calls == 2
        assert explanation.shape == COEFFICIENTS.shape
        assert extractor.batch_size == 3
        for index, encoded in enumerate(craft.encode(COEFFICIENTS)):
            direct = explainer_class(
                craft.make_concept_decoder(encoded.latent_data),
                batch_size=3,
                operator=score,
                seed=9,
                **kwargs,
            ).explain(encoded.coeffs_u, np.array([[1.0, 0.0]], dtype=np.float32))
            np.testing.assert_allclose(explanation[index], direct[0], rtol=1e-5, atol=1e-5)
    finally:
        tf.config.run_functions_eagerly(previous_eager)
    assert tf.config.functions_run_eagerly() == previous_eager

    np.testing.assert_array_equal(explanation[0, :, :, 2], 0)
    np.testing.assert_array_equal(explanation[1, :, :, 1], 0)
    np.testing.assert_allclose(explanation[:, 0], explanation[:, 1], rtol=0, atol=0)
    if explainer_class.__name__ in ("Banzhaf", "KernelBanzhaf"):
        np.testing.assert_allclose(explanation[:, 0, 0], EXPECTED_BANZHAF, atol=1e-5)
        per_image = craft.reduce_to_importance(
            explanation,
            spatial_reducer="mean",
            abs_before_reduce=False,
            aggregation_reducer=None,
        )
        np.testing.assert_allclose(per_image, EXPECTED_BANZHAF, atol=1e-5)


def test_classification_operator_uses_the_selected_class():
    """The documented classification operator also works through TorchWrapper."""
    craft = fitted_craft(HolisticCraftTorch, _Extractor(), device="cpu")
    previous_eager = tf.config.functions_run_eagerly()
    try:
        explanation = craft.compute_explanation_per_concept(
            COEFFICIENTS,
            PartialExplainer(Banzhaf, operator=Tasks.CLASSIFICATION, nb_samples=4),
            class_id=0,
        )
    finally:
        tf.config.run_functions_eagerly(previous_eager)
    np.testing.assert_allclose(explanation[:, 0, 0], EXPECTED_BANZHAF, atol=1e-5)
