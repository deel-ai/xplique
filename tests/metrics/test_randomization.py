import numpy as np
import pytest
import tensorflow as tf

from xplique.metrics.randomization import (
    ssim,
    batched_spearman,
    ProgressiveLayerRandomization,
    RandomLogitMetric,
    ModelRandomizationMetric,
    ModelRandomizationStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_cnn(num_classes: int = 4,
                     input_shape=(16, 16, 3)) -> tf.keras.Model:
    """Small conv net for tests; only needs to support forward passes."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(8, 3, activation="relu", name="conv1"),
            tf.keras.layers.Conv2D(8, 3, activation="relu", name="conv2"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="dense"),
        ]
    )
    return model


def _make_dummy_data(num_samples: int = 8,
                     num_classes: int = 4,
                     input_shape=(16, 16, 3),
                     seed: int = 42):
    """Random images + one-hot labels."""
    tf.random.set_seed(seed)
    x = tf.random.normal((num_samples,) + tuple(input_shape), dtype=tf.float32)
    y_indices = tf.random.uniform(
        (num_samples,), minval=0, maxval=num_classes, dtype=tf.int32
    )
    y = tf.one_hot(y_indices, depth=num_classes, dtype=tf.float32)
    return x, y


class InputOnlyExplainer:
    """Explainer that ignores targets and always returns the (normalized) input."""

    def __init__(self, model=None):
        self.model = model

    def explain(self, inputs, targets):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # Just normalize to [0, 1] for stability, but keep same shape
        x_min = tf.reduce_min(inputs, axis=list(range(1, inputs.shape.rank)), keepdims=True)
        x_max = tf.reduce_max(inputs, axis=list(range(1, inputs.shape.rank)), keepdims=True)
        return (inputs - x_min) / (x_max - x_min + 1e-8)


class TargetScaledExplainer:
    """
    Simple explainer that multiplies the input by a scalar depending on the target class.
    This makes explanations *change* when the class changes in a predictable way.
    """

    def __init__(self, num_classes: int, model=None):
        self.num_classes = num_classes
        self.model = model

    def explain(self, inputs, targets):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)
        class_idx = tf.argmax(targets, axis=-1, output_type=tf.int32)  # (B,)
        scale = (tf.cast(class_idx, tf.float32) + 1.0) / float(self.num_classes)
        scale = tf.reshape(scale, (-1, 1, 1, 1))
        return inputs * scale


class ModelBasedExplainer:
    """
    Explainer that returns the model logits as a flat attribution vector.
    Used to verify that ModelRandomizationMetric reacts to parameter changes.
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model

    def explain(self, inputs, targets):
        logits = self.model(tf.convert_to_tensor(inputs, dtype=tf.float32))
        return logits


class IdentityRandomization(ModelRandomizationStrategy):
    """Randomization strategy that does nothing (for sanity checks)."""

    def randomize(self, model: tf.keras.Model) -> tf.keras.Model:
        return model


# ---------------------------------------------------------------------------
# SSIM tests
# ---------------------------------------------------------------------------

def test_ssim_identical_images_is_one():
    tf.random.set_seed(42)
    img = tf.random.uniform((32, 32, 3), dtype=tf.float32)
    score = ssim(img, img)
    score_val = float(score.numpy())
    assert score_val == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_ssim_zero_dynamic_range_returns_one():
    img1 = tf.zeros((16, 16, 1), dtype=tf.float32)
    img2 = tf.zeros_like(img1)
    score = ssim(img1, img2)
    score_val = float(score.numpy())
    assert score_val == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_ssim_batched_matches_unbatched():
    tf.random.set_seed(42)
    imgs1 = tf.random.uniform((4, 8, 8, 3), dtype=tf.float32)
    imgs2 = tf.random.uniform((4, 8, 8, 3), dtype=tf.float32)

    batched_scores = ssim(imgs1, imgs2, batched=True).numpy()
    unbatched_scores = np.array(
        [float(ssim(imgs1[i], imgs2[i]).numpy()) for i in range(4)]
    )

    assert batched_scores.shape == (4,)
    assert batched_scores == pytest.approx(unbatched_scores, rel=1e-5, abs=1e-5)


def test_ssim_different_images_lower_than_one():
    img1 = tf.random.uniform((32, 32, 3), dtype=tf.float32, seed=1)
    img2 = tf.random.uniform((32, 32, 3), dtype=tf.float32, seed=2)
    score = float(ssim(img1, img2).numpy())
    assert score < 1.0


def test_ssim_small_image_adapts_filter_size():
    """Test SSIM works with images smaller than default filter size (11)."""
    tf.random.set_seed(42)
    img = tf.random.uniform((5, 5, 1), dtype=tf.float32)
    score = ssim(img, img)
    assert float(score.numpy()) == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# batched_spearman tests
# ---------------------------------------------------------------------------

def test_batched_spearman_perfect_positive_correlation():
    tf.random.set_seed(42)
    a = tf.random.normal((5, 10), dtype=tf.float32)
    b = tf.identity(a)
    corr = batched_spearman(a, b).numpy()
    assert corr.shape == (5,)
    # Should be close to 1 for each row
    assert corr == pytest.approx(np.ones_like(corr), rel=1e-4, abs=1e-4)


def test_batched_spearman_perfect_negative_correlation():
    tf.random.set_seed(42)
    a = tf.random.normal((5, 10), dtype=tf.float32)
    b = -a
    corr = batched_spearman(a, b).numpy()
    # Should be close to -1 for each row
    assert corr == pytest.approx(-np.ones_like(corr), rel=1e-4, abs=1e-4)


def test_batched_spearman_handles_constant_vectors():
    # constant vector -> zero variance -> correlation should be finite (near 0)
    tf.random.set_seed(42)
    a = tf.zeros((3, 10), dtype=tf.float32)
    b = tf.random.normal((3, 10), dtype=tf.float32)
    corr = batched_spearman(a, b).numpy()
    assert np.all(np.isfinite(corr))
    # With zero variance in ranks for a, expect something ~0
    assert np.all(np.abs(corr) < 1e-3)


def test_batched_spearman_random_vectors_bounded():
    """Test that Spearman correlation of two random vectors is in [-1, 1]."""
    tf.random.set_seed(42)
    a = tf.random.normal((10, 50), dtype=tf.float32)
    b = tf.random.normal((10, 50), dtype=tf.float32)
    corr = batched_spearman(a, b).numpy()
    assert corr.shape == (10,)
    assert np.all(np.isfinite(corr))
    assert np.all(corr >= -1.0)
    assert np.all(corr <= 1.0)


def test_rankdata_average_ties_simple_ties():
    """Test that tied values receive average ranks."""
    from xplique.metrics.randomization import _rankdata_average_ties

    x = tf.constant([[30.0, 10.0, 20.0, 20.0, 10.0]], dtype=tf.float32)
    ranks = _rankdata_average_ties(x).numpy()

    # 10s at sorted positions 0,1 -> avg rank 0.5
    # 20s at sorted positions 2,3 -> avg rank 2.5
    # 30 at sorted position 4 -> rank 4.0
    expected = np.array([[4.0, 0.5, 2.5, 2.5, 0.5]])
    assert ranks == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# ProgressiveLayerRandomization tests
# ---------------------------------------------------------------------------

def test_progressive_layer_randomization_by_name_and_reverse():
    tf.random.set_seed(42)
    model = _make_simple_cnn(num_classes=4)
    conv1 = model.get_layer("conv1")
    conv2 = model.get_layer("conv2")
    dense = model.get_layer("dense")

    original_conv1_w = [w.copy() for w in conv1.get_weights()]
    original_conv2_w = [w.copy() for w in conv2.get_weights()]
    original_dense_w = [w.copy() for w in dense.get_weights()]

    # Randomize up to (but excluding) "conv1" in reverse order.
    strategy = ProgressiveLayerRandomization(stop_layer="conv1", reverse=True)
    strategy.randomize(model)

    new_conv1_w = [w for w in conv1.get_weights()]
    new_conv2_w = [w for w in conv2.get_weights()]
    new_dense_w = [w for w in dense.get_weights()]

    # conv1 should have been randomized, conv2 untouched
    assert any(
        np.allclose(o, n) for o, n in zip(original_conv1_w, new_conv1_w)
    ), "conv1 weights should remain unchanged"

    assert all(
        not np.allclose(o, n) for o, n in zip(original_conv2_w, new_conv2_w)
    ), "conv2 weights should change"

    assert all(
        not np.allclose(o, n) for o, n in zip(original_dense_w, new_dense_w)
    ), "dense weights should change"


def test_progressive_layer_randomization_invalid_stop_layer_type():
    with pytest.raises(TypeError):
        ProgressiveLayerRandomization(stop_layer={"not": "allowed"})


def test_progressive_layer_randomization_fractional_stop_layer_bounds():
    with pytest.raises(ValueError):
        ProgressiveLayerRandomization(stop_layer=1.5)

    with pytest.raises(ValueError):
        ProgressiveLayerRandomization(stop_layer=-0.1)


def test_progressive_layer_randomization_layer_not_found():
    model = _make_simple_cnn(num_classes=4)
    strategy = ProgressiveLayerRandomization(stop_layer="nonexistent_layer")

    with pytest.raises(ValueError, match="not found"):
        strategy.randomize(model)


def test_progressive_layer_randomization_with_int_stop_layer():
    tf.random.set_seed(42)
    model = _make_simple_cnn(num_classes=4)
    # Store original weights for layers with trainable weights
    original_weights = {l.name: [w.copy() for w in l.get_weights()]
                        for l in model.layers if l.get_weights()}

    # reverse=False means we go from input to output, randomizing first 2 layers
    strategy = ProgressiveLayerRandomization(stop_layer=2, reverse=False)
    strategy.randomize(model)

    # Get layers with weights in forward order
    layers_with_weights = [l for l in model.layers if l.get_weights()]

    # First 2 layers with weights should be randomized
    randomized_count = 0
    for i, layer in enumerate(layers_with_weights):
        new_weights = layer.get_weights()
        orig_weights = original_weights[layer.name]
        weights_changed = any(
            not np.allclose(o, n) for o, n in zip(orig_weights, new_weights)
        )
        if i < 2:
            # These should be randomized
            if weights_changed:
                randomized_count += 1
        else:
            # These should remain unchanged
            assert not weights_changed, f"Layer {layer.name} should not be randomized"

    assert randomized_count == min(2, len(layers_with_weights)), \
        "Expected exactly 2 layers to be randomized"


# ---------------------------------------------------------------------------
# RandomLogitMetric tests
# ---------------------------------------------------------------------------

def test_random_logit_metric_requires_targets():
    tf.random.set_seed(42)
    model = _make_simple_cnn(num_classes=4)
    inputs, _ = _make_dummy_data(num_samples=4, num_classes=4)

    # Passing targets=None should trigger the ValueError in __init__
    with pytest.raises(ValueError):
        _ = RandomLogitMetric(
            model=model,
            inputs=inputs,
            targets=None,
            batch_size=2,
        )


def test_random_logit_metric_runs_and_outputs_in_0_1():
    tf.random.set_seed(42)
    num_classes = 5
    model = _make_simple_cnn(num_classes=num_classes)
    inputs, targets = _make_dummy_data(num_samples=6, num_classes=num_classes)

    explainer = TargetScaledExplainer(num_classes=num_classes, model=model)
    metric = RandomLogitMetric(
        model=model,
        inputs=inputs,
        targets=targets,
        batch_size=3,
        seed=123,
    )

    score = metric.evaluate(explainer)
    assert isinstance(score, float)
    # SSIM should be between 0 and 1
    assert 0.0 <= score <= 1.0


def test_random_logit_metric_invariant_explainer_gives_high_ssim():
    """
    If the explainer ignores targets and always returns the same attribution,
    then explanations for true and random classes coincide and SSIM ≈ 1.
    """
    tf.random.set_seed(42)
    num_classes = 4
    model = _make_simple_cnn(num_classes=num_classes)
    inputs, targets = _make_dummy_data(num_samples=5, num_classes=num_classes)

    explainer = InputOnlyExplainer(model)
    metric = RandomLogitMetric(
        model=model,
        inputs=inputs,
        targets=targets,
        batch_size=2,
        seed=42,
    )

    score = metric.evaluate(explainer)
    assert score == pytest.approx(1.0, rel=1e-3, abs=1e-3)


# ---------------------------------------------------------------------------
# ModelRandomizationMetric tests
# ---------------------------------------------------------------------------

def test_model_randomization_metric_requires_targets():
    tf.random.set_seed(42)
    model = _make_simple_cnn(num_classes=4)
    inputs, _ = _make_dummy_data(num_samples=4, num_classes=4)

    with pytest.raises(ValueError):
        _ = ModelRandomizationMetric(
            model=model,
            inputs=inputs,
            targets=None,
        )


def test_model_randomization_metric_with_invariant_explainer_corr_one():
    """
    If the explainer ignores the model parameters, randomization should not
    change explanations and Spearman ≈ 1.
    """
    tf.random.set_seed(42)
    num_classes = 4
    model = _make_simple_cnn(num_classes=num_classes)
    inputs, targets = _make_dummy_data(num_samples=6, num_classes=num_classes)

    explainer = InputOnlyExplainer(model)

    metric = ModelRandomizationMetric(
        model=model,
        inputs=inputs,
        targets=targets,
        randomization_strategy=ProgressiveLayerRandomization(stop_layer=0.5),
        batch_size=3,
        seed=123,
    )

    score = metric.evaluate(explainer)
    assert isinstance(score, float)
    # Spearman correlation of identical vectors is 1
    assert score == pytest.approx(1.0, rel=1e-4, abs=1e-4)


def test_model_randomization_metric_with_identity_strategy_behaves_like_no_randomization():
    """
    Using an IdentityRandomization strategy is equivalent to comparing explanations from the
    same model twice; correlation should be ≈ 1 even if explainer depends on the model.
    """
    tf.random.set_seed(42)
    num_classes = 3
    model = _make_simple_cnn(num_classes=num_classes)
    inputs, targets = _make_dummy_data(num_samples=5, num_classes=num_classes)

    explainer = ModelBasedExplainer(model)

    metric = ModelRandomizationMetric(
        model=model,
        inputs=inputs,
        targets=targets,
        randomization_strategy=IdentityRandomization(),
        batch_size=5,
        seed=7,
    )

    score = metric.evaluate(explainer)
    assert score == pytest.approx(1.0, rel=1e-4, abs=1e-4)


def test_model_randomization_metric_scores_are_finite_and_bounded():
    """
    With a model-dependent explainer and an actual randomization strategy,
    Spearman correlations should remain finite and in [-1, 1].
    """
    tf.random.set_seed(42)
    num_classes = 4
    model = _make_simple_cnn(num_classes=num_classes)
    inputs, targets = _make_dummy_data(num_samples=8, num_classes=num_classes)

    explainer = ModelBasedExplainer(model)

    metric = ModelRandomizationMetric(
        model=model,
        inputs=inputs,
        targets=targets,
        randomization_strategy=ProgressiveLayerRandomization(stop_layer=0.5),
        batch_size=4,
        seed=99,
    )

    score = metric.evaluate(explainer)
    assert np.isfinite(score)
    assert -1.0 <= score <= 1.0


def test_model_randomization_metric_with_integer_targets():
    """Test that integer labels are correctly converted to one-hot."""
    tf.random.set_seed(42)
    num_classes = 4
    model = _make_simple_cnn(num_classes=num_classes)
    inputs, _ = _make_dummy_data(num_samples=6, num_classes=num_classes)
    targets = tf.random.uniform((6,), minval=0, maxval=num_classes, dtype=tf.int32)

    explainer = InputOnlyExplainer(model)
    metric = ModelRandomizationMetric(
        model=model,
        inputs=inputs,
        targets=targets,
        batch_size=3,
    )

    score = metric.evaluate(explainer)
    assert np.isfinite(score)


def test_model_randomization_restores_original_model():
    """Verify explainer.model is restored after evaluation."""
    model = _make_simple_cnn(num_classes=4)
    inputs, targets = _make_dummy_data(num_samples=4, num_classes=4)
    explainer = InputOnlyExplainer(model)

    metric = ModelRandomizationMetric(model=model, inputs=inputs, targets=targets)
    metric.evaluate(explainer)

    assert explainer.model is model
