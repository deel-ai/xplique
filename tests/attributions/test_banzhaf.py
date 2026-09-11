"""Focused contracts for signed, support-restricted Banzhaf attribution."""

import numpy as np
import pytest
import tensorflow as tf

from xplique import attributions
from xplique.attributions import Banzhaf, concept_attributions
from xplique.attributions.base import BlackBoxExplainer
from xplique.attributions.concept_attributions.base import _ConceptChannelExplainer


def _sum(inputs):
    return tf.reduce_sum(inputs, axis=tf.range(1, tf.rank(inputs)))


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


def test_exports_and_defaults():
    """Both public imports expose the same black-box concept explainer."""
    assert Banzhaf is concept_attributions.Banzhaf
    assert issubclass(Banzhaf, _ConceptChannelExplainer)
    assert issubclass(_ConceptChannelExplainer, BlackBoxExplainer)
    assert "_ConceptChannelExplainer" not in getattr(attributions, "__all__", [])
    assert "_ConceptChannelExplainer" not in getattr(concept_attributions, "__all__", [])
    explainer = Banzhaf(_sum, operator=_operator)
    assert explainer.batch_size == 32
    assert explainer.nb_samples == 1024
    assert explainer.seed == 0


@pytest.mark.parametrize("interaction", [0.0, 3.0, -4.0])
@pytest.mark.parametrize("column", [False, True])
def test_exact_additive_and_pair_effects(interaction, column):
    """Conditional contrasts retain signs and split a pair interaction equally."""

    def model(inputs):
        scores = 7.0 + 2.0 * inputs[:, 0] - inputs[:, 1] + interaction * inputs[:, 0] * inputs[:, 1]
        return scores[:, None] if column else scores

    inputs = np.array([[2.0, 3.0, 0.0], [-1.0, 2.0, 5.0]], np.float32)
    expected = np.column_stack((2 * inputs[:, 0], -inputs[:, 1], np.zeros(2)))
    expected[:, :2] += (interaction * inputs[:, 0] * inputs[:, 1] / 2)[:, None]
    result = Banzhaf(model, operator=_operator, nb_samples=8)(inputs, np.ones((2, 1)))
    assert isinstance(result, tf.Tensor)
    assert result.dtype == tf.float32
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("game", ["constant", "xor"])
def test_exact_zero_effects(game):
    """Constants and a balanced two-player XOR have exactly zero effects."""

    def model(inputs):
        if game == "constant":
            return tf.fill([tf.shape(inputs)[0]], 13.0)
        return tf.cast(tf.not_equal(inputs[:, 0], inputs[:, 1]), tf.float32)

    result = Banzhaf(model, operator=_operator, nb_samples=4)([[1.0, 1.0, 0.0]], [[1.0]])
    np.testing.assert_array_equal(result, np.zeros((1, 3), np.float32))


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4), (2, 2, 3, 4), (2, 2, 2, 3, 4)])
def test_support_broadcast_and_whole_channel_masking(shape):
    """Support is per input, includes canceling channels, and masks whole channels."""
    inputs = np.zeros(shape, np.float32)
    inputs[0, ..., 0] = 2.0
    inputs[0, ..., 2] = -3.0
    inputs[1, ..., 1] = 4.0
    if len(shape) > 2:
        inputs[0, 0, ..., 0] = -2.0
        inputs[1, 0, ..., 1] = 0.0
    calls = []

    def operator(model, perturbed, targets):
        rows = perturbed.numpy()
        for row, target in zip(rows, targets.numpy()):
            original = inputs[int(target[0])]
            for channel in range(shape[-1]):
                assert np.array_equal(row[..., channel], original[..., channel]) or np.all(
                    row[..., channel] == 0
                )
        calls.append(len(rows))
        return model(perturbed)

    result = Banzhaf(_sum, operator=operator, nb_samples=8, batch_size=3)(inputs, [[0.0], [1.0]])
    effects = inputs.sum(axis=tuple(range(1, len(shape) - 1)), keepdims=True)
    np.testing.assert_allclose(result, np.broadcast_to(effects, shape), atol=1e-6)
    assert sum(calls) == 6
    assert max(calls) <= 3
    np.testing.assert_array_equal(result.numpy()[..., 3], 0)


@pytest.mark.parametrize("value", [1e-12, -1e-12, 2.5, -7.0])
@pytest.mark.parametrize("channels", [1, 3])
def test_no_support_threshold_or_magnitude_normalization(value, channels):
    """Every exactly nonzero coefficient participates, regardless of magnitude."""
    inputs = np.zeros((1, channels), np.float32)
    inputs[0, channels // 2] = value
    result = Banzhaf(_sum, operator=_operator, nb_samples=2)(inputs, [[1.0]])
    np.testing.assert_array_equal(result, inputs)


@pytest.mark.parametrize("kind", ["numpy", "tensor", "dataset", "batched_dataset"])
def test_input_containers(kind):
    """Sanitization handles arrays, tensors, and datasets with explicit None targets."""
    inputs = np.array([[1.0, -2.0, 0.0], [0.0, 3.0, 4.0]], np.float32)
    targets = np.ones((2, 1), np.float32)
    source = inputs
    if kind == "tensor":
        source, targets = tf.constant(inputs), tf.constant(targets)
    elif "dataset" in kind:
        source = tf.data.Dataset.from_tensor_slices((inputs, targets))
        if kind == "batched_dataset":
            source = source.batch(1)
        targets = None
    result = Banzhaf(_sum, operator=_operator, nb_samples=4).explain(source, targets)
    np.testing.assert_array_equal(result, inputs)


@pytest.mark.parametrize("batch_size", [1, 3, None])
def test_fixed_structured_targets_and_forward_counts(batch_size):
    """Each coalition keeps the complete target belonging to its original input."""
    inputs = np.array([[1.0, 2.0, 0.0], [0.0, -3.0, 0.0]], np.float32)
    targets = np.array([[[2.0, 3.0], [4.0, 5.0]], [[-1.0, 6.0], [7.0, 8.0]]], np.float32)
    seen = []

    def operator(model, perturbed, repeated):
        repeated = repeated.numpy()
        index = 0 if repeated[0, 0, 0] == 2 else 1
        np.testing.assert_array_equal(
            repeated, np.repeat(targets[index : index + 1], len(repeated), axis=0)
        )
        seen.append((index, len(repeated)))
        return model(perturbed) * repeated[:, 0, 0]

    result = Banzhaf(_sum, operator=operator, nb_samples=16, batch_size=batch_size)(inputs, targets)
    np.testing.assert_array_equal(result, inputs * targets[:, 0, 0, None])
    assert [sum(size for index, size in seen if index == i) for i in range(2)] == [4, 2]
    if batch_size is None:
        assert seen == [(0, 4), (1, 2)]
    else:
        assert all(size <= batch_size for _, size in seen)


@pytest.mark.parametrize("shape", [(0, 3), (0, 2, 3), (2, 3)])
def test_empty_batch_and_inactive_inputs_skip_inference(shape):
    """Empty batches and entirely inactive inputs return float32 zeros without calls."""

    def operator(model, inputs, targets):
        pytest.fail("Zero-support inputs must not invoke the operator")

    result = Banzhaf(_sum, operator=operator)(np.zeros(shape), np.ones((shape[0], 1)))
    assert result.dtype == tf.float32
    np.testing.assert_array_equal(result, np.zeros(shape))


@pytest.mark.parametrize(
    "name, value",
    [("nb_samples", value) for value in [0, -2, 1, 3, 2.0, True, np.bool_(False), None]]
    + [("batch_size", value) for value in [0, -1, 1.5, True, np.bool_(True)]]
    + [("seed", value) for value in [True, np.bool_(False), 1.0, None, -(2**63) - 1, 2**63]],
)
def test_invalid_parameters(name, value):
    """Integer parameters reject booleans, invalid ranges, and fractional values."""
    with pytest.raises(ValueError):
        Banzhaf(_sum, operator=_operator, **{name: value})


@pytest.mark.parametrize("seed", [-(2**63), -1, 2**63 - 1, np.int64(7)])
def test_signed_integer_seeds(seed):
    """The entire signed64 seed range and NumPy integer parameters are accepted."""
    explainer = Banzhaf(
        _sum, operator=_operator, seed=seed, nb_samples=np.int64(6), batch_size=np.int64(2)
    )
    masks = explainer._sample_masks(5, 0)
    assert masks.shape == (6, 5)


@pytest.mark.parametrize(
    "inputs, targets",
    [
        (1.0, [[1.0]]),
        ([1.0, 2.0], [[1.0]]),
        (np.zeros((1, 0)), [[1.0]]),
        (np.zeros((1, 2, 0)), [[1.0]]),
        ([[np.nan]], [[1.0]]),
        ([[np.inf]], [[1.0]]),
        ([[-np.inf]], [[1.0]]),
        ([[1.0]], 1.0),
        ([[1.0], [2.0]], [[1.0]]),
        ([[1.0]], np.zeros((0, 1))),
        (np.zeros((0, 3)), [[1.0]]),
    ],
)
def test_invalid_inputs_and_targets(inputs, targets):
    """Input finiteness, ranks, channel count, and target alignment are validated."""
    with pytest.raises(ValueError):
        Banzhaf(_sum, operator=_operator)(inputs, targets)


@pytest.mark.parametrize(
    "output",
    [
        "scalar",
        "wide",
        "rank3",
        "short",
        "long",
        "nan",
        "inf",
        "-inf",
    ],
)
def test_invalid_operator_outputs(output):
    """Only finite (B,) and (B, 1) operator scores are accepted."""

    def operator(model, inputs, targets):
        size = tf.shape(inputs)[0]
        if output == "scalar":
            return tf.constant(1.0)
        if output == "wide":
            return tf.ones((size, 2))
        if output == "rank3":
            return tf.ones((size, 1, 1))
        if output == "short":
            return tf.ones((size - 1,))
        if output == "long":
            return tf.ones((size + 1,))
        return tf.fill((size,), float(output))

    with pytest.raises(ValueError):
        Banzhaf(_sum, operator=operator, nb_samples=4)([[1.0, 2.0]], [[1.0]])


@pytest.mark.parametrize("nb_active", [1, 2, 3])
def test_exhaustive_mask_order(nb_active):
    """Exact designs enumerate integers with channel zero as the least significant bit."""
    explainer = Banzhaf(_sum, operator=_operator, nb_samples=8)
    masks = explainer._sample_masks(nb_active, 17)
    expected = (np.arange(2**nb_active)[:, None] >> np.arange(nb_active)) & 1
    assert masks.dtype == tf.float32
    np.testing.assert_array_equal(masks, expected)


def test_antithetic_masks_are_stateless_balanced_and_indexed():
    """Monte Carlo designs append complements and depend on seed and input index."""
    explainer = Banzhaf(_sum, operator=_operator, nb_samples=30, seed=91)
    masks = explainer._sample_masks(12, 0).numpy()
    assert masks.dtype == np.float32
    assert np.all((masks == 0) | (masks == 1))
    np.testing.assert_array_equal(masks[15:], 1 - masks[:15])
    np.testing.assert_array_equal(masks.sum(axis=0), np.full(12, 15))
    tf.random.uniform((100,))
    np.testing.assert_array_equal(masks, explainer._sample_masks(12, 0))
    other = Banzhaf(_sum, operator=_operator, nb_samples=30, seed=91)
    np.testing.assert_array_equal(masks, other._sample_masks(12, 0))
    assert not np.array_equal(masks, explainer._sample_masks(12, 1))
    other = Banzhaf(_sum, operator=_operator, nb_samples=30, seed=92)
    assert not np.array_equal(masks, other._sample_masks(12, 0))


def test_sampled_effects_counts_and_batch_invariance():
    """Sampled estimates match conditional means and use exactly the requested rows."""
    inputs = np.arange(1, 25, dtype=np.float32).reshape(2, 12)
    results = []
    for batch_size in [1, 7, 32, None]:
        calls = []

        def operator(model, perturbed, targets):
            calls.append(int(perturbed.shape[0]))
            return model(perturbed)

        explainer = Banzhaf(_sum, operator=operator, nb_samples=30, batch_size=batch_size, seed=91)
        result = explainer(inputs, np.ones(2)).numpy()
        expected = []
        for index, row in enumerate(inputs):
            masks = explainer._sample_masks(12, index).numpy()
            scores = (masks * row).sum(axis=1)
            expected.append(
                [
                    scores[masks[:, j] == 1].mean() - scores[masks[:, j] == 0].mean()
                    for j in range(12)
                ]
            )
        np.testing.assert_allclose(result, expected, atol=2e-5)
        assert sum(calls) == 60
        assert max(calls) <= (30 if batch_size is None else batch_size)
        results.append(result)
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], atol=2e-5)


def test_eight_active_channels_in_384_use_exact_design():
    """Enumeration cost depends on active support, not the ambient channel count."""
    inputs = np.zeros((1, 384), np.float32)
    active = np.array([0, 3, 17, 80, 160, 255, 300, 383])
    inputs[0, active] = np.arange(1, 9) * np.array([1, -1] * 4)
    calls = []

    def operator(model, perturbed, targets):
        assert perturbed.shape[1:] == (384,)
        calls.append(int(perturbed.shape[0]))
        return model(perturbed)

    result = Banzhaf(_sum, operator=operator, nb_samples=256, batch_size=19)(inputs, [[1.0]])
    np.testing.assert_array_equal(result, inputs)
    assert sum(calls) == 256
    assert max(calls) <= 19


def test_inactive_inputs_preserve_original_sampling_index():
    """Skipping an inactive row does not renumber subsequent Monte Carlo designs."""
    inputs = np.zeros((3, 12), np.float32)
    inputs[1] = np.arange(1, 13, dtype=np.float32)
    seen = []

    def operator(model, perturbed, targets):
        np.testing.assert_array_equal(targets, np.ones(len(perturbed)))
        seen.append(len(perturbed))
        return model(perturbed)

    explainer = Banzhaf(_sum, operator=operator, nb_samples=30, batch_size=7, seed=91)
    masks = explainer._sample_masks(12, 1).numpy()
    scores = (masks * inputs[1]).sum(axis=1)
    expected = np.zeros_like(inputs)
    expected[1] = [
        scores[masks[:, j] == 1].mean() - scores[masks[:, j] == 0].mean() for j in range(12)
    ]
    result = explainer(inputs, np.arange(3, dtype=np.float32))
    np.testing.assert_allclose(result, expected, atol=2e-5)
    np.testing.assert_array_equal(result.numpy()[[0, 2]], 0)
    assert sum(seen) == 30
    assert max(seen) <= 7


@pytest.mark.parametrize("kind", ["keras", "numpy"])
def test_default_operator_smoke(kind):
    """Default target selection works with Functional Keras and NumPy callables."""
    weights = np.array([[2.0, -1.0], [-3.0, 4.0]], np.float32)
    if kind == "keras":
        model_inputs = tf.keras.Input(shape=(2,))
        outputs = tf.keras.layers.Dense(
            2, use_bias=False, kernel_initializer=tf.keras.initializers.Constant(weights)
        )(model_inputs)
        model = tf.keras.Model(model_inputs, outputs)
    else:

        def model(inputs):
            assert isinstance(inputs, np.ndarray)
            return inputs @ weights

    inputs = np.array([[1.0, 2.0], [-2.0, 3.0]], np.float32)
    result = Banzhaf(model, nb_samples=4)(inputs, np.eye(2, dtype=np.float32))
    np.testing.assert_allclose(result, inputs * weights.T, atol=1e-6)


def test_float64_scores_preserve_small_effects_with_large_offsets():
    """Do not round valid operator scores to the attribution dtype before centering."""

    def model(inputs):
        return tf.cast(inputs[:, 0], tf.float64) + tf.constant(1e8, tf.float64)

    result = Banzhaf(model, operator=_operator, nb_samples=2)([[1.0]], [[1.0]])
    assert result.dtype == tf.float32
    np.testing.assert_array_equal(result, [[1.0]])


def test_inference_only_torch_wrapper():
    """Channel-last concept decoders work without PyTorch gradients."""
    torch = pytest.importorskip("torch")
    from xplique.wrappers import TorchWrapper

    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -3.0, 4.0]]))
    model.eval()
    eager = tf.config.functions_run_eagerly()
    try:
        wrapper = TorchWrapper(model, "cpu", is_channel_first=False, requires_grad=False)
        inputs = np.array([[1.0, 2.0, 0.0], [-2.0, 0.0, 3.0]], np.float32)
        result = Banzhaf(wrapper, nb_samples=4)(inputs, np.ones((2, 1)))
        np.testing.assert_array_equal(result, inputs * [2.0, -3.0, 4.0])
    finally:
        tf.config.run_functions_eagerly(eager)
