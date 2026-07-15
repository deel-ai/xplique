"""Regression tests for PyTorch latent extractors."""
# ruff: noqa: E402

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xplique.concepts.torch.holistic_craft import HolisticCraftTorch
from xplique.concepts.torch.latent_extractor import TorchLatentExtractor
from xplique.concepts.torch.layered_model_latent_extractor import LayeredModelExtractorBuilder


def _identity_extractor(device=None, batch_size=2):
    return TorchLatentExtractor(
        model=torch.nn.Identity(),
        input_to_latent_model=lambda inputs: inputs * 2,
        latent_to_logit_model=lambda latent_data: latent_data,
        device=device,
        batch_size=batch_size,
    )


def test_torch_builder_includes_the_requested_layer_for_positive_and_negative_indices():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )
    samples = torch.randn(2, 1, 2, 2)

    for split_layer in (1, -2):
        extractor = LayeredModelExtractorBuilder.build(model, split_layer, device="cpu")
        latent_data = extractor.input_to_latent(samples)
        predictions = extractor.latent_to_logit(latent_data)
        split_index = split_layer % len(model)

        assert torch.allclose(latent_data.activations, model[: split_index + 1](samples))
        assert torch.allclose(predictions, model(samples))


@pytest.mark.parametrize("split_layer", ["layer", True, -3, 2])
def test_torch_builder_validates_split_layer(split_layer):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU())

    with pytest.raises(ValueError, match="split_layer"):
        LayeredModelExtractorBuilder.build(model, split_layer, device="cpu")


def test_generator_exits_no_grad_before_yielding():
    extractor = _identity_extractor(device="cpu")
    generator = extractor.input_to_latent_generator(torch.ones((2, 3, 2, 2), requires_grad=True))

    latent_data = next(generator)
    value_created_by_the_consumer = torch.ones((), requires_grad=True) * 2

    assert not latent_data.requires_grad
    assert value_created_by_the_consumer.requires_grad
    generator.close()


def test_device_selection_transfer_and_input_validation():
    import tensorflow as tf

    extractor = _identity_extractor()

    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert extractor.device.type == expected_device
    craft = HolisticCraftTorch(extractor)
    assert craft.device == extractor.device
    np.testing.assert_array_equal(craft._to_numpy(tf.constant([1.0])), [1.0])
    assert extractor.to("cpu").device.type == "cpu"
    assert next(extractor.input_to_latent_generator(torch.ones((1, 3, 2, 2)))).device.type == "cpu"

    with pytest.raises(ValueError, match="Invalid PyTorch device"):
        _identity_extractor(device="not-a-device")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA was requested"):
            _identity_extractor(device="cuda")
    with pytest.raises(ValueError, match="rank 3"):
        extractor.input_to_latent(torch.ones((2, 2)))
    with pytest.raises(ValueError, match="at least one"):
        list(extractor.input_to_latent_generator(torch.empty((0, 3, 2, 2))))
    with pytest.raises(ValueError, match="batch_size"):
        _identity_extractor(batch_size=0)


def test_classifier_tensor_mode_keeps_existing_batch_dimension():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )
    extractor = LayeredModelExtractorBuilder.build(model, split_layer=1, device="cpu")
    extractor.output_as_tensor = True

    predictions = extractor(torch.ones((2, 1, 2, 2)))

    assert predictions.shape == (2, 2)
