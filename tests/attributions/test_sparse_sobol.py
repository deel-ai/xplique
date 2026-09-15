"""Focused contracts for support-restricted sparse Sobol attribution."""

from numbers import Integral

import numpy as np
import pytest
import tensorflow as tf

from xplique import attributions
from xplique.attributions import SparseSobol, concept_attributions
from xplique.attributions.concept_attributions.base import _ConceptChannelExplainer
from xplique.attributions.global_sensitivity_analysis import JansenEstimator
from xplique.attributions.global_sensitivity_analysis.replicated_designs import ReplicatedSampler


def _sum(inputs):
    return tf.reduce_sum(inputs, axis=tf.range(1, tf.rank(inputs)))


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


def _reference_ab(seed, input_index, nb_design, dimension):
    folded = tf.random.experimental.stateless_fold_in(
        tf.constant([seed, 0], tf.int64), tf.cast(input_index, tf.int64)
    )
    seed_a = tf.random.experimental.stateless_fold_in(folded, tf.constant(0, tf.int64))
    seed_b = tf.random.experimental.stateless_fold_in(folded, tf.constant(1, tf.int64))
    shape = [nb_design, dimension]
    return (
        tf.random.stateless_uniform(shape, seed_a, dtype=tf.float32),
        tf.random.stateless_uniform(shape, seed_b, dtype=tf.float32),
    )


def test_exports_inheritance_and_defaults():
    """Public imports expose the concept explainer with documented defaults."""
    assert SparseSobol is concept_attributions.SparseSobol
    assert issubclass(SparseSobol, _ConceptChannelExplainer)
    assert "SparseSobol" in attributions.__all__
    assert "SparseSobol" in concept_attributions.__all__
    explainer = SparseSobol(_sum, operator=_operator)
    assert explainer.batch_size == 32
    assert explainer.nb_design == 32
    assert explainer.mask_distribution == "uniform"
    assert explainer.seed == 0


@pytest.mark.parametrize("nb_design", [2, 3, 7, np.int64(9)])
@pytest.mark.parametrize("distribution", ["uniform", "bernoulli"])
def test_valid_parameters_include_non_power_of_two(nb_design, distribution):
    """Any integral design size of at least two is accepted."""
    explainer = SparseSobol(
        _sum, operator=_operator, nb_design=nb_design, mask_distribution=distribution
    )
    assert explainer.nb_design == int(nb_design)
    assert isinstance(explainer.nb_design, Integral)


@pytest.mark.parametrize(
    "name,value",
    [("nb_design", value) for value in [0, 1, -2, 2.0, True, np.bool_(False), None]]
    + [("mask_distribution", value) for value in ["Uniform", "binary", "", None, 1]],
)
def test_invalid_sparse_sobol_parameters(name, value):
    """Design size and distribution use strict public validation."""
    with pytest.raises(ValueError):
        SparseSobol(_sum, operator=_operator, **{name: value})


@pytest.mark.parametrize("distribution", ["uniform", "bernoulli"])
def test_exact_stateless_ab_c_design(distribution):
    """Masks are ordered A, B, then dimension-major A-with-one-B-column blocks."""
    nb_design, dimension = 5, 3
    explainer = SparseSobol(
        _sum, operator=_operator, nb_design=nb_design, mask_distribution=distribution, seed=-19
    )
    actual = explainer._sample_masks(dimension, 4)
    expected_a, expected_b = _reference_ab(-19, 4, nb_design, dimension)
    if distribution == "bernoulli":
        expected_a = tf.cast(expected_a < 0.5, tf.float32)
        expected_b = tf.cast(expected_b < 0.5, tf.float32)
    expected_c = ReplicatedSampler.build_replicated_design(expected_a, expected_b)
    expected = tf.concat([expected_a, expected_b, expected_c], axis=0)

    assert actual.dtype == tf.float32
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:nb_design], expected_a)
    np.testing.assert_array_equal(actual[nb_design : 2 * nb_design], expected_b)
    for column in range(dimension):
        block = actual[(2 + column) * nb_design : (3 + column) * nb_design].numpy()
        expected_block = expected_a.numpy().copy()
        expected_block[:, column] = expected_b.numpy()[:, column]
        np.testing.assert_array_equal(block, expected_block)
    if distribution == "uniform":
        assert bool(tf.reduce_all((actual >= 0) & (actual < 1)))
    else:
        assert set(np.unique(actual.numpy())) <= {0.0, 1.0}


def test_sampling_is_reproducible_indexed_and_global_rng_independent():
    """Sampling depends only on all seed bits and the original input index."""
    explainer = SparseSobol(_sum, operator=_operator, nb_design=11, seed=23)
    masks = explainer._sample_masks(6, 0)
    tf.random.uniform((100,))
    np.testing.assert_array_equal(masks, explainer._sample_masks(6, 0))
    np.testing.assert_array_equal(
        masks, SparseSobol(_sum, operator=_operator, nb_design=11, seed=23)._sample_masks(6, 0)
    )
    assert not np.array_equal(masks, explainer._sample_masks(6, 1))
    assert not np.array_equal(
        masks, SparseSobol(_sum, operator=_operator, nb_design=11, seed=24)._sample_masks(6, 0)
    )
    assert not np.array_equal(
        masks,
        SparseSobol(_sum, operator=_operator, nb_design=11, seed=23 + 2**32)._sample_masks(6, 0),
    )


@pytest.mark.parametrize("seed", [-(2**63), -1, 2**63 - 1, np.int64(7)])
def test_signed64_seed_endpoints(seed):
    """The complete signed 64-bit seed domain follows the same reference folding."""
    explainer = SparseSobol(_sum, operator=_operator, nb_design=3, seed=seed)
    actual = explainer._sample_masks(2, 9)
    sampling_a, sampling_b = _reference_ab(int(seed), 9, 3, 2)
    expected_c = ReplicatedSampler.build_replicated_design(sampling_a, sampling_b)
    np.testing.assert_array_equal(actual, tf.concat([sampling_a, sampling_b, expected_c], 0))


def test_estimate_is_direct_jansen_evaluation():
    """SparseSobol delegates its sampled outputs unchanged to Jansen total indices."""
    explainer = SparseSobol(_sum, operator=_operator, nb_design=7, seed=31)
    masks = explainer._sample_masks(4, 2)
    outputs = tf.cast(tf.range(masks.shape[0]), tf.float64) ** 2 + 0.25
    expected = JansenEstimator()(masks, outputs, explainer.nb_design)
    np.testing.assert_array_equal(explainer._estimate(masks, outputs), expected)


def test_constant_scores_are_exactly_zero():
    """Jansen's protected zero-variance case returns exact zero indices."""

    def constant(inputs):
        return tf.fill([tf.shape(inputs)[0]], 13.0)

    result = SparseSobol(constant, operator=_operator, nb_design=7)([[1.0, -2.0, 0.0]], [[1.0]])
    np.testing.assert_array_equal(result, np.zeros((1, 3), np.float32))


def test_bernoulli_xor_has_positive_total_indices():
    """Unlike signed main effects, total indices detect a pure XOR interaction."""

    def xor(inputs):
        return tf.cast(tf.not_equal(inputs[:, 0], inputs[:, 1]), tf.float32)

    result = SparseSobol(
        xor, operator=_operator, nb_design=256, mask_distribution="bernoulli", seed=17
    )([[1.0, 1.0, 0.0]], [[1.0]])
    assert np.all(result.numpy()[0, :2] > 0)
    assert result.numpy()[0, 2] == 0


def test_support_broadcast_and_inactive_channels():
    """Whole-channel indices broadcast over positions and absent channels stay zero."""
    inputs = np.array([[[2.0, 1.0, 0.0], [-2.0, 3.0, 0.0]]], np.float32)
    calls = []

    def operator(model, perturbed, targets):
        calls.append(len(perturbed))
        return model(perturbed)

    result = SparseSobol(_sum, operator=operator, nb_design=9, batch_size=13)(inputs, [[1.0]])
    np.testing.assert_array_equal(result.numpy()[..., 2], 0)
    np.testing.assert_array_equal(result.numpy()[:, 0, :2], result.numpy()[:, 1, :2])
    assert sum(calls) == 9 * (2 + 2)
    assert max(calls) <= 13


@pytest.mark.parametrize("batch_size", [1, 11, None])
def test_exact_evaluation_budget_call_counts_and_batch_invariance(batch_size):
    """Each active row evaluates N(d+2) masks with batching affecting only call sizes."""
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, -2.0]], np.float32)
    calls = []

    def operator(model, perturbed, targets):
        calls.append(len(perturbed))
        return model(perturbed)

    explainer = SparseSobol(_sum, operator=operator, nb_design=7, batch_size=batch_size, seed=41)
    result = explainer(inputs, np.ones((2, 1), np.float32))
    reference = SparseSobol(_sum, operator=_operator, nb_design=7, batch_size=None, seed=41)(
        inputs, np.ones((2, 1), np.float32)
    )
    np.testing.assert_allclose(result, reference, rtol=0, atol=1e-6)
    assert sum(calls) == 7 * (3 + 2) + 7 * (2 + 2)
    if batch_size is None:
        assert calls == [35, 28]
    else:
        assert max(calls) <= batch_size


def test_inactive_rows_preserve_original_sampling_index(monkeypatch):
    """Skipping zero support does not renumber the next row's stateless design."""
    inputs = np.array([[0.0, 0.0], [2.0, -3.0], [0.0, 0.0]], np.float32)
    sampled = []
    original = SparseSobol._sample_masks

    def recording_sample(self, nb_active, input_index):
        sampled.append((nb_active, input_index))
        return original(self, nb_active, input_index)

    monkeypatch.setattr(SparseSobol, "_sample_masks", recording_sample)
    result = SparseSobol(_sum, operator=_operator, nb_design=5)(inputs, np.ones((3, 1)))
    assert sampled == [(2, 1)]
    np.testing.assert_array_equal(result.numpy()[[0, 2]], 0)


@pytest.mark.parametrize("kind", ["numpy", "tensor", "dataset", "batched_dataset"])
def test_numpy_tensor_and_dataset_inputs(kind):
    """Inherited sanitization accepts dense inputs and paired datasets."""
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
    result = SparseSobol(_sum, operator=_operator, nb_design=7).explain(source, targets)
    assert result.shape == inputs.shape
    assert result.dtype == tf.float32
    assert bool(tf.reduce_all(tf.math.is_finite(result)))
    np.testing.assert_array_equal(result.numpy()[inputs == 0], 0)


def test_default_keras_operator_matches_explicit_target_selection():
    """The default operator selects fixed Keras outputs for every perturbation."""
    weights = np.array([[2.0, -1.0], [-3.0, 4.0]], np.float32)
    model_inputs = tf.keras.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(
        2, use_bias=False, kernel_initializer=tf.keras.initializers.Constant(weights)
    )(model_inputs)
    model = tf.keras.Model(model_inputs, outputs)
    inputs = np.array([[1.0, 2.0], [-2.0, 3.0]], np.float32)
    targets = np.eye(2, dtype=np.float32)

    def select_target(model, perturbed, repeated_targets):
        return tf.reduce_sum(model(perturbed) * repeated_targets, axis=1)

    actual = SparseSobol(model, nb_design=11, seed=5)(inputs, targets)
    expected = SparseSobol(model, operator=select_target, nb_design=11, seed=5)(inputs, targets)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("output", ["scalar", "wide", "short", "nan"])
def test_inherited_malformed_operator_scores(output):
    """Only finite scalar-per-perturbation operator scores reach Jansen."""

    def malformed(model, inputs, targets):
        del model, targets
        size = tf.shape(inputs)[0]
        if output == "scalar":
            return tf.constant(1.0)
        if output == "wide":
            return tf.ones((size, 2))
        if output == "short":
            return tf.ones((size - 1,))
        return tf.fill((size,), np.nan)

    with pytest.raises(ValueError):
        SparseSobol(_sum, operator=malformed, nb_design=3, batch_size=None)([[1.0, 2.0]], [[1.0]])


def test_inference_only_torch_wrapper():
    """Optional Torch models work through the inherited inference-only wrapper path."""
    torch = pytest.importorskip("torch")
    from xplique.wrappers import TorchWrapper

    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -3.0]]))
    model.eval()
    eager = tf.config.functions_run_eagerly()
    try:
        wrapper = TorchWrapper(model, "cpu", is_channel_first=False, requires_grad=False)
        result = SparseSobol(wrapper, nb_design=7)([[1.0, 2.0]], [[1.0]])
        assert result.shape == (1, 2)
        assert bool(tf.reduce_all(tf.math.is_finite(result)))
    finally:
        tf.config.run_functions_eagerly(eager)
