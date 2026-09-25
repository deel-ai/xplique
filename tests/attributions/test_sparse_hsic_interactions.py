"""Mathematical and execution contracts for indexed SparseHSIC interactions."""

from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest
import tensorflow as tf

from xplique import attributions
from xplique.attributions import ConceptInteractionResult, SparseHSIC, concept_attributions


def _sum(inputs):
    return tf.reduce_sum(inputs, axis=tf.range(1, tf.rank(inputs)))


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


def _grams(outputs):
    outputs = np.asarray(outputs, np.float64)
    n = len(outputs)
    distances = np.abs(outputs[:, None] - outputs[None, :])
    positive = distances[distances > 0]
    bandwidth = np.median(positive) if positive.size else 1.0
    gram = np.exp(-0.5 * np.square(distances / bandwidth))
    centering = np.eye(n) - np.ones((n, n)) / n
    return gram, centering


def _reference_pair(masks, outputs, left, right):
    gram, centering = _grams(outputs)
    signs = 2 * np.asarray(masks, np.float64) - 1
    kernel_left = 0.5 * np.outer(signs[:, left], signs[:, left])
    kernel_right = 0.5 * np.outer(signs[:, right], signs[:, right])
    interaction_kernel = kernel_left * kernel_right
    n = len(signs)
    direct = np.trace(centering @ interaction_kernel @ centering @ gram) / n**2
    joint = (1 + kernel_left) * (1 + kernel_right)
    joint_minus_singletons = (
        np.trace(centering @ joint @ centering @ gram)
        - np.trace(centering @ kernel_left @ centering @ gram)
        - np.trace(centering @ kernel_right @ centering @ gram)
    ) / n**2
    np.testing.assert_allclose(direct, joint_minus_singletons, rtol=0, atol=1e-16)
    return direct


def _reference_singletons(masks, outputs):
    gram, centering = _grams(outputs)
    n = len(outputs)
    return [
        np.trace(centering @ (column[:, None] == column[None, :]) @ centering @ gram) / n**2
        for column in np.asarray(masks).T
    ]


def _fixed_explainer(masks, outputs):
    """Evaluate an exhaustive mask design with a known scalar score per row."""
    masks = np.asarray(masks, np.float32)
    values = {tuple(row): float(score) for row, score in zip(masks, outputs)}

    def model(inputs):
        return tf.constant([values[tuple(row)] for row in inputs.numpy()], tf.float64)

    explainer = SparseHSIC(model, operator=_operator, nb_samples=len(masks), batch_size=2)
    explainer._sample_masks = lambda nb_active, input_index: tf.constant(masks)
    return explainer


def test_pair_matches_centered_gram_and_joint_minus_singletons_on_unbalanced_design():
    masks = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    outputs = np.array([0.0, 1.0, 1.0, 3.0, 5.0])
    explainer = SparseHSIC(_sum, operator=_operator, nb_samples=len(masks))
    gram = explainer._output_gram(tf.constant(outputs))
    signed = tf.cast(2 * masks - 1, tf.float64)
    pairs = tf.constant([[0, 1], [1, 2], [0, 2], [1, 0]], tf.int64)
    actual = explainer._estimate_pair_chunk(signed, pairs, gram)
    expected = [_reference_pair(masks, outputs, *pair) for pair in pairs.numpy()]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-8)
    np.testing.assert_allclose(
        explainer._estimate(masks, outputs), _reference_singletons(masks, outputs), atol=2e-8
    )
    assert actual[0] == actual[-1]
    # Centering the individual signs before multiplying is incorrect here.
    centered_signs = signed.numpy() - signed.numpy().mean(axis=0)
    wrong = centered_signs[:, 0] * centered_signs[:, 1]
    assert not np.allclose(
        wrong - wrong.mean(),
        (signed.numpy()[:, 0] * signed.numpy()[:, 1])
        - (signed.numpy()[:, 0] * signed.numpy()[:, 1]).mean(),
    )


@pytest.mark.parametrize("n_bits", [2, 3])
def test_exhaustive_parity_is_purely_pairwise_only_for_two_bits(n_bits):
    masks = np.array(list(product((0, 1), repeat=n_bits)), np.float32)
    outputs = (masks.sum(axis=1) % 2).astype(np.float64)
    result = _fixed_explainer(masks, outputs).explain_interactions(
        np.ones((1, n_bits), np.float32), [[1.0]]
    )[0]
    np.testing.assert_array_equal(result.main_effects, np.zeros(n_bits, np.float32))
    if n_bits == 2:
        expected = (1 - np.exp(-0.5)) / 8
        np.testing.assert_allclose(result.interaction_scores, [expected], atol=1e-8)
    else:
        np.testing.assert_array_equal(result.interaction_scores, np.zeros(3, np.float32))


def test_exhaustive_single_variable_output_has_no_irrelevant_pair():
    masks = np.array(list(product((0, 1), repeat=2)), np.float32)
    result = _fixed_explainer(masks, masks[:, 0]).explain_interactions([[1.0, 1.0]], [[1.0]])[0]
    assert result.main_effects[0] > 0
    assert result.main_effects[1] == 0
    assert result.interaction_scores[0] == 0


def test_result_exports_dtypes_frozen_fields_and_singleton_compatibility():
    assert ConceptInteractionResult is concept_attributions.ConceptInteractionResult
    assert "ConceptInteractionResult" in attributions.__all__
    explainer = SparseHSIC(_sum, operator=_operator, nb_samples=17, seed=11)
    inputs = np.array([[2.0, 0.0, 3.0, 4.0]], np.float32)
    result = explainer.explain_interactions(inputs, [[1.0]])[0]
    np.testing.assert_array_equal(result.active_ids, [0, 2, 3])
    np.testing.assert_array_equal(result.pair_indices, [[0, 2], [0, 3], [2, 3]])
    np.testing.assert_array_equal(
        result.main_effects, explainer.explain(inputs, [[1.0]]).numpy()[0, [0, 2, 3]]
    )
    assert result.n_concepts == 4
    assert result.active_ids.dtype == result.pair_indices.dtype == tf.int64
    assert result.main_effects.dtype == result.interaction_scores.dtype == tf.float32
    with pytest.raises(FrozenInstanceError):
        result.n_concepts = 7


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [[0, 1, 2]],
        [[0, 0]],
        [[1, 0]],
        [[0, 3]],
        [[-1, 1]],
        [[0, 1], [0, 1]],
        [[False, True]],
        [[0.0, 1.0]],
        np.array([[0, 1]], dtype=np.uint64) * np.uint64(2**63),
    ],
)
def test_bad_pairs_fail_before_any_inference_even_on_empty_batch(pairs):
    def forbidden(model, inputs, targets):
        pytest.fail("Invalid pair selection must not invoke the operator")

    explainer = SparseHSIC(_sum, operator=forbidden, nb_samples=5)
    with pytest.raises(ValueError):
        explainer.explain_interactions(np.zeros((0, 3)), np.zeros((0, 1)), pairs=pairs)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.bool_(False), None])
def test_invalid_pair_batch_size(value):
    with pytest.raises(ValueError):
        SparseHSIC(_sum, operator=_operator).explain_interactions(
            np.zeros((0, 2)), np.zeros((0, 1)), pair_batch_size=value
        )


def test_empty_inputs_support_and_explicit_empty_pairs():
    def forbidden(model, inputs, targets):
        pytest.fail("No model calls")

    explainer = SparseHSIC(_sum, operator=forbidden)
    assert explainer.explain_interactions(np.zeros((0, 3)), np.zeros((0, 1))) == []
    requested = np.array([[0, 2], [0, 1]], np.int64)
    result = explainer.explain_interactions(np.zeros((1, 3)), [[1.0]], pairs=requested)[0]
    assert result.active_ids.shape == result.main_effects.shape == (0,)
    np.testing.assert_array_equal(result.pair_indices, requested)
    np.testing.assert_array_equal(result.interaction_scores, [0.0, 0.0])
    result = explainer.explain_interactions(np.zeros((1, 3)), [[1.0]])[0]
    assert result.pair_indices.shape == (0, 2)
    assert result.interaction_scores.shape == (0,)


@pytest.mark.parametrize("shape", [(1, 3), (1, 2, 3), (1, 2, 2, 3)])
def test_spatial_shapes_inactive_pairs_and_explicit_row_order(shape):
    inputs = np.ones(shape, np.float32)
    inputs[..., 1] = 0
    requested = tf.constant([[1, 2], [0, 2], [0, 1]], tf.int32)
    result = SparseHSIC(_sum, operator=_operator, nb_samples=9).explain_interactions(
        inputs, [[1.0]], pairs=requested, pair_batch_size=1
    )[0]
    np.testing.assert_array_equal(result.active_ids, [0, 2])
    np.testing.assert_array_equal(result.pair_indices, requested)
    assert result.main_effects.shape == (2,)
    assert result.interaction_scores.shape == (3,)
    np.testing.assert_array_equal(result.interaction_scores.numpy()[[0, 2]], [0.0, 0.0])


def test_one_channel_and_empty_explicit_pairs_still_evaluate_singletons():
    calls = []

    def operator(model, inputs, targets):
        calls.append(len(inputs))
        return model(inputs)

    explainer = SparseHSIC(_sum, operator=operator, nb_samples=7, batch_size=3)
    result = explainer.explain_interactions([[0.0, 2.0]], [[1.0]])[0]
    assert result.main_effects.shape == (1,)
    assert result.pair_indices.shape == (0, 2)
    result = explainer.explain_interactions(
        [[1.0, 2.0]], [[1.0]], pairs=np.empty((0, 2), np.int64)
    )[0]
    assert result.main_effects.shape == (2,)
    assert result.interaction_scores.shape == (0,)
    assert calls == [3, 3, 1, 3, 3, 1]


def test_support_is_determined_after_float32_sanitization():
    result = SparseHSIC(_sum, operator=_operator, nb_samples=7).explain_interactions(
        np.array([[1.0, 1e-50, 2.0]], np.float64), [[1.0]]
    )[0]
    np.testing.assert_array_equal(result.active_ids, [0, 2])
    np.testing.assert_array_equal(result.pair_indices, [[0, 2]])


@pytest.mark.parametrize("output", ["wide", "nan"])
def test_interactions_preserve_operator_score_validation(output):
    def invalid(model, inputs, targets):
        del model, targets
        if output == "wide":
            return tf.ones((len(inputs), 2))
        return tf.fill([len(inputs)], np.nan)

    with pytest.raises(ValueError, match="operator must return"):
        SparseHSIC(_sum, operator=invalid, nb_samples=3).explain_interactions([[1.0, 2.0]], [[1.0]])


def test_extreme_output_bandwidth_remains_finite_for_pairs():
    masks = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], tf.float32)
    outputs = tf.constant([-1e308, -1e308, 1e308, 1e308], tf.float64)
    explainer = SparseHSIC(_sum, operator=_operator, nb_samples=4)
    gram = explainer._output_gram(outputs)
    scores = explainer._estimate_pair_chunk(
        2.0 * tf.cast(masks, tf.float64) - 1.0, tf.constant([[0, 1]]), gram
    )
    assert np.isfinite(scores.numpy()).all()


def test_torch_wrapper_interactions_require_no_gradients():
    torch = pytest.importorskip("torch")
    from xplique.wrappers import TorchWrapper

    model = torch.nn.Linear(2, 1, bias=False)
    model.eval()
    eager = tf.config.functions_run_eagerly()
    try:
        wrapper = TorchWrapper(model, "cpu", is_channel_first=False, requires_grad=False)
        result = SparseHSIC(wrapper, nb_samples=7).explain_interactions([[1.0, 2.0]], [[1.0]])[0]
        assert result.pair_indices.shape == (1, 2)
        assert np.isfinite(result.interaction_scores.numpy()).all()
    finally:
        tf.config.run_functions_eagerly(eager)


def test_chunk_bounds_and_inference_budget_independent_of_pair_count(monkeypatch):
    calls = []
    chunks = []
    original = SparseHSIC._estimate_pair_chunk

    def recording(signed_masks, local_pairs, gram):
        chunks.append(len(local_pairs))
        return original(signed_masks, local_pairs, gram)

    monkeypatch.setattr(SparseHSIC, "_estimate_pair_chunk", staticmethod(recording))

    def operator(model, inputs, targets):
        calls.append(len(inputs))
        return model(inputs)

    inputs = np.ones((1, 6), np.float32)
    for batch_size in (2, None):
        explainer = SparseHSIC(_sum, operator=operator, nb_samples=7, batch_size=batch_size, seed=3)
        automatic = explainer.explain_interactions(inputs, [[1.0]], pair_batch_size=4)[0]
        assert chunks == [4, 4, 4, 3]
        assert calls == ([2, 2, 2, 1] if batch_size == 2 else [7])
        chunks.clear()
        calls.clear()
        requested = explainer.explain_interactions(
            inputs, [[1.0]], pairs=automatic.pair_indices, pair_batch_size=1
        )[0]
        np.testing.assert_array_equal(requested.interaction_scores, automatic.interaction_scores)
        assert max(chunks) == 1
        assert calls == ([2, 2, 2, 1] if batch_size == 2 else [7])
        chunks.clear()
        calls.clear()


def test_constant_scores_dataset_and_split_call_seed_semantics():
    def constant(inputs):
        return tf.fill([len(inputs)], tf.constant(1e100, tf.float64))

    zeros = SparseHSIC(constant, operator=_operator, nb_samples=9).explain_interactions(
        [[1.0, 2.0]], [[1.0]]
    )[0]
    np.testing.assert_array_equal(zeros.main_effects, [0.0, 0.0])
    np.testing.assert_array_equal(zeros.interaction_scores, [0.0])

    inputs = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], np.float32)
    targets = np.ones((2, 1), np.float32)
    explainer = SparseHSIC(_sum, operator=_operator, nb_samples=19, seed=17)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(1)
    grouped = explainer.explain_interactions(dataset)
    direct = explainer.explain_interactions(inputs, targets)
    for left, right in zip(grouped, direct):
        np.testing.assert_array_equal(left.main_effects, right.main_effects)
        np.testing.assert_array_equal(left.interaction_scores, right.interaction_scores)
    split = explainer.explain_interactions(inputs[1:], targets[1:])[0]
    np.testing.assert_array_equal(split.main_effects, grouped[0].main_effects)
    assert not np.array_equal(grouped[1].main_effects, split.main_effects)
