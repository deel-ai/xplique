"""Regression tests for padded object-detection tensor outputs."""

import pytest
import tensorflow as tf

from xplique.concepts.tf.holistic_craft import ConceptDecoderTf
from xplique.concepts.tf.latent_extractor import TfLatentExtractor
from xplique.utils_functions.object_detection.tf.box_model_wrapper import TfBoxesModelWrapper
from xplique.utils_functions.object_detection.tf.multi_box_tensor import TfMultiBoxTensor

try:
    import torch

    from xplique.concepts.torch.holistic_craft import ConceptDecoderTorch
    from xplique.concepts.torch.latent_extractor import TorchLatentExtractor
    from xplique.utils_functions.object_detection.torch.box_model_wrapper import (
        TorchBoxesModelWrapper,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _TfVariableDetectionFormatter:
    def __call__(self, predictions):
        return [TfMultiBoxTensor(predictions[0, :0]), TfMultiBoxTensor(predictions[1, :3])]


class _TfDecoderParent:
    @staticmethod
    def _decode_coefficients(_latent_data, coefficients):
        return [
            TfMultiBoxTensor(coefficients[0, :1]),
            TfMultiBoxTensor(coefficients[1, :3]),
        ]


def test_tf_box_wrapper_pads_variable_detection_counts():
    wrapper = TfBoxesModelWrapper(
        tf.keras.layers.Lambda(lambda value: value), _TfVariableDetectionFormatter()
    )
    wrapper.output_as_tensor = True

    @tf.function
    def run(inputs):
        return wrapper(inputs)

    predictions = run(tf.ones((2, 3, 7)))

    assert predictions.shape == (2, 3, 7)
    tf.debugging.assert_equal(predictions[0], tf.zeros((3, 7)))


def test_tf_latent_extractor_pads_variable_detection_counts():
    extractor = TfLatentExtractor(
        model=tf.keras.layers.Lambda(lambda value: value),
        input_to_latent_model=lambda samples: samples[:, 0],
        latent_to_logit_model=lambda latent_data: latent_data,
        output_formatter=_TfVariableDetectionFormatter(),
    )
    extractor.output_as_tensor = True

    predictions = extractor(tf.ones((2, 1, 3, 7)))

    assert predictions.shape == (2, 3, 7)
    tf.debugging.assert_equal(predictions[0], tf.zeros((3, 7)))


def test_tf_concept_decoder_pads_batched_variable_detection_counts():
    decoder = ConceptDecoderTf(_TfDecoderParent(), latent_data=None)
    coefficients = tf.Variable(tf.ones((2, 3, 7)))

    with tf.GradientTape() as tape:
        predictions = decoder(coefficients)
        loss = tf.reduce_sum(predictions)
    gradients = tape.gradient(loss, coefficients)

    assert predictions.shape == (2, 3, 7)
    tf.debugging.assert_equal(predictions[0, 1:], tf.zeros((2, 7)))
    assert gradients is not None
    tf.debugging.assert_positive(tf.reduce_sum(tf.abs(gradients)))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_torch_box_wrapper_pads_variable_detection_counts():
    def formatter(predictions):
        return [predictions[0, :0], predictions[1, :3]]

    wrapper = TorchBoxesModelWrapper(torch.nn.Identity(), formatter)
    wrapper.output_as_tensor = True

    predictions = wrapper(torch.ones((2, 3, 7)))

    assert predictions.shape == (2, 3, 7)
    assert torch.equal(predictions[0], torch.zeros((3, 7)))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_torch_latent_extractor_pads_variable_detection_counts():
    def formatter(predictions):
        return [predictions[0, :0], predictions[1, :3]]

    extractor = TorchLatentExtractor(
        model=torch.nn.Identity(),
        input_to_latent_model=lambda samples: samples[:, 0],
        latent_to_logit_model=lambda latent_data: latent_data,
        output_formatter=formatter,
        device="cpu",
    )
    extractor.output_as_tensor = True

    predictions = extractor(torch.ones((2, 1, 3, 7)))

    assert predictions.shape == (2, 3, 7)
    assert torch.equal(predictions[0], torch.zeros((3, 7)))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_torch_concept_decoder_pads_batched_variable_detection_counts():
    class DecoderParent:
        @staticmethod
        def _decode_coefficients(_latent_data, coefficients):
            return [coefficients[0, :1], coefficients[1, :3]]

    decoder = ConceptDecoderTorch(DecoderParent(), latent_data=None)
    coefficients = torch.ones((2, 3, 7), requires_grad=True)

    predictions = decoder(coefficients)
    predictions.sum().backward()

    assert predictions.shape == (2, 3, 7)
    assert torch.equal(predictions[0, 1:], torch.zeros((2, 7)))
    assert coefficients.grad is not None
    assert torch.sum(torch.abs(coefficients.grad)) > 0
