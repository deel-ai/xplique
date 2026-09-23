"""Cross-framework parity tests for HolisticCraft concept localization."""

import numpy as np
import pytest
import tensorflow as tf
import torch

from xplique.concepts.tf.holistic_craft import HolisticCraftTf
from xplique.concepts.tf.latent_extractor import TfLatentExtractor
from xplique.concepts.tf.layered_model_latent_extractor import (
    LayeredLatentData as TfLatentData,
)
from xplique.concepts.torch.holistic_craft import HolisticCraftTorch
from xplique.concepts.torch.latent_extractor import TorchLatentExtractor
from xplique.concepts.torch.layered_model_latent_extractor import (
    LayeredLatentData as TorchLatentData,
)


@pytest.mark.parametrize("reducer", ["mean", "sum", "max"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tf_torch_concept_localizer_score_parity(reducer, device, identity_factorizer):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    values = np.arange(2 * 4 * 4 * 2, dtype=np.float32).reshape(2, 4, 4, 2)
    tf_images = tf.constant(values)
    tf_extractor = TfLatentExtractor(
        model=lambda inputs: inputs,
        input_to_latent_model=lambda inputs: TfLatentData(inputs),
        latent_to_logit_model=lambda latent_data: latent_data.activations,
        batch_size=2,
    )
    tf_craft = HolisticCraftTf(
        tf_extractor,
        number_of_concepts=2,
        factorizer=identity_factorizer(),
    )
    tf_craft.fit(tf_images)

    torch_images = torch.from_numpy(values.transpose(0, 3, 1, 2)).to(device)
    torch_extractor = TorchLatentExtractor(
        model=torch.nn.Identity().eval(),
        input_to_latent_model=lambda inputs: TorchLatentData(inputs),
        latent_to_logit_model=lambda latent_data: latent_data.activations,
        device=device,
        batch_size=2,
    )
    torch_craft = HolisticCraftTorch(
        torch_extractor,
        number_of_concepts=2,
        device=device,
        factorizer=identity_factorizer(),
    )
    torch_craft.fit(torch_images)

    tf_scores = tf_craft.make_concept_localizer(reducer)(tf_images)
    torch_scores = torch_craft.make_concept_localizer(reducer)(values)

    for scores in (tf_scores, torch_scores):
        assert isinstance(scores, tf.Tensor)
        assert scores.shape == (2, 2)
        assert scores.dtype == tf.float32
        assert np.all(np.isfinite(scores.numpy()))

    expected_scores = getattr(np, reducer)(values, axis=(1, 2))
    np.testing.assert_allclose(tf_scores.numpy(), expected_scores, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(torch_scores.numpy(), expected_scores, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tf_scores.numpy(), torch_scores.numpy(), rtol=1e-6, atol=1e-6)


def test_torch_localizer_does_not_change_tf_execution_policy(identity_factorizer):
    values = np.ones((1, 4, 4, 2), dtype=np.float32)
    extractor = TorchLatentExtractor(
        model=torch.nn.Identity().eval(),
        input_to_latent_model=lambda inputs: TorchLatentData(inputs),
        latent_to_logit_model=lambda latent_data: latent_data.activations,
        device="cpu",
        batch_size=1,
    )
    craft = HolisticCraftTorch(extractor, number_of_concepts=2, factorizer=identity_factorizer())
    craft.fit(torch.from_numpy(values.transpose(0, 3, 1, 2)))

    original_policy = tf.config.functions_run_eagerly()
    try:
        tf.config.run_functions_eagerly(False)
        localizer = craft.make_concept_localizer()
        assert not tf.config.functions_run_eagerly()
        scores = localizer(values)
        assert scores.shape == (1, 2)
        assert not tf.config.functions_run_eagerly()
    finally:
        tf.config.run_functions_eagerly(original_policy)
