"""Regression tests for TensorFlow latent extractors."""

import numpy as np
import pytest
import tensorflow as tf

from xplique.concepts.tf.latent_extractor import TfLatentExtractor
from xplique.concepts.tf.layered_model_latent_extractor import LayeredModelExtractorBuilder


def _identity_extractor(batch_size=2):
    return TfLatentExtractor(
        model=lambda inputs: inputs,
        input_to_latent_model=lambda inputs: inputs,
        latent_to_logit_model=lambda latent_data: latent_data,
        batch_size=batch_size,
    )


def _functional_classifier():
    inputs = tf.keras.Input(shape=(2, 2, 1))
    hidden = tf.keras.layers.Conv2D(4, 1, activation="relu", name="hidden")(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="pool")(hidden)
    outputs = tf.keras.layers.Dense(2, name="classifier")(pooled)
    return tf.keras.Model(inputs, outputs)


def test_functional_builder_preserves_residual_graph():
    inputs = tf.keras.Input(shape=(2, 2, 1))
    split = tf.keras.layers.Conv2D(4, 1, activation="relu", name="split")(inputs)
    left = tf.keras.layers.Conv2D(4, 1, name="left")(split)
    right = tf.keras.layers.Conv2D(4, 1, name="right")(split)
    merged = tf.keras.layers.Add(name="merge")([left, right])
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="pool")(merged)
    outputs = tf.keras.layers.Dense(2, name="classifier")(pooled)
    model = tf.keras.Model(inputs, outputs)

    split_index = model.layers.index(model.get_layer("split"))
    extractor = LayeredModelExtractorBuilder.build(model, split_layer=split_index)
    samples = tf.constant(np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1))

    latent_data = extractor.input_to_latent(samples)
    predictions = extractor.latent_to_logit(latent_data).tensor

    tf.debugging.assert_near(predictions, model(samples))
    tf.debugging.assert_near(latent_data.activations, model.get_layer("split")(samples))


@pytest.mark.parametrize("split_layer", ["hidden", True, -5, 4])
def test_functional_builder_validates_split_layer(split_layer):
    with pytest.raises(ValueError, match="split_layer"):
        LayeredModelExtractorBuilder.build(_functional_classifier(), split_layer=split_layer)


def test_functional_builder_requires_split_to_be_a_graph_cut():
    inputs = tf.keras.Input(shape=(2, 2, 1))
    split = tf.keras.layers.Conv2D(2, 1, name="split")(inputs)
    bypass = tf.keras.layers.Conv2D(2, 1, name="bypass")(inputs)
    outputs = tf.keras.layers.Add()([split, bypass])
    model = tf.keras.Model(inputs, outputs)

    with pytest.raises(ValueError, match="graph cut"):
        LayeredModelExtractorBuilder.build(
            model, split_layer=model.layers.index(model.get_layer("split"))
        )


def test_classifier_tensor_mode_keeps_existing_batch_dimension():
    extractor = LayeredModelExtractorBuilder.build(_functional_classifier(), split_layer=1)
    extractor.output_as_tensor = True

    predictions = extractor(tf.ones((2, 2, 2, 1)))

    assert isinstance(predictions, tf.Tensor)
    assert predictions.shape == (2, 2)


def test_input_validation_and_single_image_batching():
    extractor = _identity_extractor()

    latent_data = extractor.input_to_latent(tf.ones((2, 2, 1)))

    assert latent_data.shape == (1, 2, 2, 1)
    with pytest.raises(ValueError, match="rank 3"):
        extractor.input_to_latent(tf.ones((2, 2)))
    with pytest.raises(ValueError, match="at least one"):
        list(extractor.input_to_latent_generator(tf.zeros((0, 2, 2, 1))))
    with pytest.raises(ValueError, match="batch_size"):
        _identity_extractor(batch_size=0)
