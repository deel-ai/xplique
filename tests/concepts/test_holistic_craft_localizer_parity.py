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
def test_tf_torch_concept_localizer_score_parity(reducer, identity_factorizer):
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

    torch_images = torch.from_numpy(values.transpose(0, 3, 1, 2))
    torch_extractor = TorchLatentExtractor(
        model=torch.nn.Identity(),
        input_to_latent_model=lambda inputs: TorchLatentData(inputs),
        latent_to_logit_model=lambda latent_data: latent_data.activations,
        device="cpu",
        batch_size=2,
    )
    torch_craft = HolisticCraftTorch(
        torch_extractor,
        number_of_concepts=2,
        device="cpu",
        factorizer=identity_factorizer(),
    )
    torch_craft.fit(torch_images)

    tf_scores = tf_craft.make_concept_localizer(reducer)(tf_images).numpy()
    torch_scores = torch_craft.make_concept_localizer(reducer)(values).numpy()

    np.testing.assert_allclose(tf_scores, torch_scores, rtol=1e-6, atol=1e-6)
    expected_torch_scores = getattr(np, reducer)(
        torch_craft.transform(torch_images),
        axis=(1, 2),
    )
    np.testing.assert_allclose(
        torch_scores,
        expected_torch_scores,
        rtol=1e-6,
        atol=1e-6,
    )
