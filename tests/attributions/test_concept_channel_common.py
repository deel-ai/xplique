"""Shared execution contracts for every concept-channel explainer.

Method-specific estimators and designs are tested in each method's own file.
"""

from typing import Callable, NamedTuple

import numpy as np
import pytest
import tensorflow as tf

from xplique import attributions
from xplique.attributions import (
    Banzhaf,
    KernelBanzhaf,
    SparseHSIC,
    SparseSobol,
    concept_attributions,
)
from xplique.attributions.base import BlackBoxExplainer
from xplique.attributions.concept_attributions.base import _ConceptChannelExplainer


class _Method(NamedTuple):
    """A small explainer configuration and its known per-input evaluation budget."""

    cls: type
    kwargs: dict
    evaluations: Callable[[int], int]  # model evaluations for d active channels
    defaults: dict
    sampled_kwargs: dict  # Monte Carlo configuration used by sampling tests
    sampled_dim: int


METHODS = [
    pytest.param(
        _Method(
            Banzhaf,
            {"nb_samples": 8},
            lambda d: 2**d if d < 4 else 8,
            {"nb_samples": 1024},
            {"nb_samples": 30},
            12,
        ),
        id="Banzhaf",
    ),
    pytest.param(
        _Method(
            KernelBanzhaf,
            {"nb_samples": 8},
            lambda d: 2**d if d < 4 else 8,
            {"nb_samples": 1024},
            {"nb_samples": 64},
            8,
        ),
        id="KernelBanzhaf",
    ),
    pytest.param(
        _Method(
            SparseSobol,
            {"nb_design": 4},
            lambda d: 4 * (d + 2),
            {"nb_design": 32, "mask_distribution": "uniform"},
            {"nb_design": 11},
            6,
        ),
        id="SparseSobol",
    ),
    pytest.param(
        _Method(
            SparseHSIC,
            {"nb_samples": 8},
            lambda d: 8,
            {"nb_samples": 1024},
            {"nb_samples": 17},
            8,
        ),
        id="SparseHSIC",
    ),
]


@pytest.fixture(params=METHODS)
def method(request):
    return request.param


def _sum(inputs):
    return tf.reduce_sum(inputs, axis=tf.range(1, tf.rank(inputs)))


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


def _make(method, model=_sum, **overrides):
    return method.cls(model, **{"operator": _operator, **method.kwargs, **overrides})


def _forbidden(model, inputs, targets):
    pytest.fail("This path must not sample masks or invoke the operator")


def test_exports_and_defaults(method):
    """Both public namespaces expose the method; the shared base class stays private."""
    name = method.cls.__name__
    assert getattr(concept_attributions, name) is method.cls
    assert name in attributions.__all__
    assert name in concept_attributions.__all__
    assert issubclass(method.cls, _ConceptChannelExplainer)
    assert issubclass(_ConceptChannelExplainer, BlackBoxExplainer)
    assert "_ConceptChannelExplainer" not in getattr(attributions, "__all__", [])
    assert "_ConceptChannelExplainer" not in getattr(concept_attributions, "__all__", [])
    explainer = method.cls(_sum, operator=_operator)
    assert explainer.batch_size == 32
    assert explainer.seed == 0
    for key, value in method.defaults.items():
        assert getattr(explainer, key) == value


@pytest.mark.parametrize(
    "name, value",
    [("batch_size", value) for value in [0, -1, 1.5, True, np.bool_(True)]]
    + [("seed", value) for value in [True, np.bool_(False), 1.0, None, -(2**63) - 1, 2**63]],
)
def test_invalid_shared_parameters(method, name, value):
    """Integer parameters reject booleans, invalid ranges, and fractional values."""
    with pytest.raises(ValueError):
        _make(method, **{name: value})


@pytest.mark.parametrize("seed", [-(2**63), -1, 2**63 - 1, np.int64(7)])
def test_signed_integer_seeds(method, seed):
    """The entire signed64 seed range and NumPy integer parameters are accepted."""
    explainer = _make(method, seed=seed, batch_size=np.int64(2))
    assert explainer.seed == int(seed)
    assert explainer.batch_size == 2
    masks = explainer._sample_masks(2, 0)
    assert masks.dtype == tf.float32
    assert masks.shape == (method.evaluations(2), 2)


def test_sampling_is_stateless_indexed_and_uses_every_seed_bit(method):
    """Designs depend on the seed and original input index, not the global RNG."""

    def sample(seed, input_index):
        explainer = method.cls(_sum, operator=_operator, seed=seed, **method.sampled_kwargs)
        return explainer._sample_masks(method.sampled_dim, input_index).numpy()

    masks = sample(23, 0)
    tf.random.uniform((100,))
    np.testing.assert_array_equal(masks, sample(23, 0))
    assert not np.array_equal(masks, sample(23, 1))
    assert not np.array_equal(masks, sample(24, 0))
    assert not np.array_equal(masks, sample(23 + 2**32, 0))


@pytest.mark.parametrize("kind", ["tensor", "dataset", "batched_dataset"])
def test_input_containers(method, kind):
    """Sanitization handles tensors and datasets with explicit None targets."""
    inputs = np.array([[1.0, -2.0, 0.0], [0.0, 3.0, 4.0]], np.float32)
    targets = np.ones((2, 1), np.float32)
    expected = _make(method)(inputs, targets)
    source = inputs
    if kind == "tensor":
        source, targets = tf.constant(inputs), tf.constant(targets)
    else:
        source = tf.data.Dataset.from_tensor_slices((inputs, targets))
        if kind == "batched_dataset":
            source = source.batch(1)
        targets = None
    result = _make(method).explain(source, targets)
    assert result.dtype == tf.float32
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(result.numpy()[inputs == 0], 0)


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
def test_invalid_inputs_and_targets(method, inputs, targets):
    """Input finiteness, ranks, channel count, and target alignment are validated."""
    with pytest.raises(ValueError):
        _make(method)(inputs, targets)


@pytest.mark.parametrize(
    "output", ["scalar", "wide", "rank3", "short", "long", "nan", "inf", "-inf"]
)
def test_invalid_operator_outputs(method, output):
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
        _make(method, operator=operator)([[1.0, 2.0]], [[1.0]])


@pytest.mark.parametrize("shape", [(0, 3), (0, 2, 3), (2, 3), (2, 2, 3)])
def test_empty_batch_and_inactive_inputs_skip_sampling_and_inference(method, shape, monkeypatch):
    """Empty batches and entirely inactive inputs return float32 zeros without calls."""
    monkeypatch.setattr(method.cls, "_sample_masks", _forbidden)
    result = _make(method, operator=_forbidden)(np.zeros(shape), np.ones((shape[0], 1)))
    assert result.dtype == tf.float32
    np.testing.assert_array_equal(result, np.zeros(shape))


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4), (2, 2, 3, 4), (2, 2, 2, 3, 4)])
def test_support_broadcast_and_whole_channel_masking(method, shape):
    """Support is per input, includes canceling channels, and scales whole channels."""
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
                # One mask value in [0, 1] multiplies every position of the channel.
                support = original[..., channel] != 0
                np.testing.assert_array_equal(row[..., channel][~support], 0)
                if support.any():
                    ratios = row[..., channel][support] / original[..., channel][support]
                    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)
                    assert 0 <= ratios[0] <= 1
        calls.append(len(rows))
        return model(perturbed)

    result = _make(method, operator=operator, batch_size=3)(inputs, [[0.0], [1.0]]).numpy()
    assert result.shape == shape
    flat = result.reshape(shape[0], -1, shape[-1])
    np.testing.assert_array_equal(flat, np.broadcast_to(flat[:, :1], flat.shape))
    active = np.any(inputs.reshape(flat.shape) != 0, axis=1)
    np.testing.assert_array_equal(flat[:, 0][~active], 0)
    assert sum(calls) == method.evaluations(2) + method.evaluations(1)
    assert max(calls) <= 3


@pytest.mark.parametrize("batch_size", [1, 3, None])
def test_batching_and_skipped_rows_change_only_call_sizes(method, batch_size, monkeypatch):
    """Batching and skipped rows alter neither the designs nor the explanations."""
    inputs = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 0.0, -2.0]], np.float32)
    targets = np.arange(3, dtype=np.float32)[:, None]
    reference = _make(method, batch_size=None, seed=41)(inputs, targets)
    sampled = []
    calls = []
    original = method.cls._sample_masks

    def recording_sample(self, nb_active, input_index):
        sampled.append((nb_active, input_index))
        return original(self, nb_active, input_index)

    def operator(model, perturbed, repeated_targets):
        index = int(repeated_targets[0, 0])
        assert np.all(repeated_targets.numpy() == index)
        calls.append((index, len(perturbed)))
        return model(perturbed)

    monkeypatch.setattr(method.cls, "_sample_masks", recording_sample)
    result = _make(method, operator=operator, batch_size=batch_size, seed=41)(inputs, targets)
    np.testing.assert_allclose(result, reference, rtol=0, atol=1e-5)
    # Skipping an inactive row does not renumber subsequent designs.
    assert sampled == [(3, 0), (2, 2)]
    totals = [sum(size for index, size in calls if index == row) for row in range(3)]
    assert totals == [method.evaluations(3), 0, method.evaluations(2)]
    if batch_size is None:
        assert calls == [(0, method.evaluations(3)), (2, method.evaluations(2))]
    else:
        assert max(size for _, size in calls) <= batch_size


def test_fixed_structured_targets(method):
    """Each perturbation keeps the complete target belonging to its original input."""
    inputs = np.array([[1.0, 2.0, 0.0], [0.0, -3.0, 0.0]], np.float32)
    targets = np.array([[[2.0, 3.0], [4.0, 5.0]], [[-1.0, 6.0], [7.0, 8.0]]], np.float32)
    seen = []

    def operator(model, perturbed, repeated):
        repeated = repeated.numpy()
        index = 0 if repeated[0, 0, 0] == 2 else 1
        np.testing.assert_array_equal(
            repeated, np.repeat(targets[index : index + 1], len(repeated), axis=0)
        )
        seen.append(index)
        return model(perturbed) * repeated[:, 0, 0]

    _make(method, operator=operator, batch_size=3)(inputs, targets)
    assert sorted(set(seen)) == [0, 1]


@pytest.mark.parametrize("kind", ["keras", "numpy"])
def test_default_operator_matches_explicit_target_selection(method, kind):
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

    def select_target(model, perturbed, repeated_targets):
        return tf.reduce_sum(tf.matmul(perturbed, weights) * repeated_targets, axis=1)

    inputs = np.array([[1.0, 2.0], [-2.0, 3.0]], np.float32)
    targets = np.eye(2, dtype=np.float32)
    actual = method.cls(model, seed=5, **method.kwargs)(inputs, targets)
    expected = method.cls(model, operator=select_target, seed=5, **method.kwargs)(inputs, targets)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_inference_only_torch_wrapper(method):
    """Channel-last concept decoders work without PyTorch gradients."""
    torch = pytest.importorskip("torch")
    from xplique.wrappers import TorchWrapper

    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -3.0, 4.0]]))
    model.eval()
    inputs = np.array([[1.0, 2.0, 0.0], [-2.0, 0.0, 3.0]], np.float32)
    targets = np.ones((2, 1), np.float32)
    weights = tf.constant([[2.0], [-3.0], [4.0]])
    expected = method.cls(lambda values: values @ weights, **method.kwargs)(inputs, targets)
    eager = tf.config.functions_run_eagerly()
    try:
        wrapper = TorchWrapper(model, "cpu", is_channel_first=False, requires_grad=False)
        result = method.cls(wrapper, **method.kwargs)(inputs, targets)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-5)
    finally:
        tf.config.run_functions_eagerly(eager)
