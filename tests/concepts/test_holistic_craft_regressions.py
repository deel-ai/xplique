"""Regression tests for framework-agnostic HolisticCraft behavior."""

from contextlib import contextmanager

import matplotlib
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from xplique.attributions.gradient_input import GradientInput
from xplique.concepts.craft import Factorization
from xplique.concepts.holistic_craft import ConceptLocalizer, HolisticCraft, PartialExplainer
from xplique.concepts.latent_extractor import LatentData


class _LatentData(LatentData):
    def __init__(self, activations):
        self.activations = activations

    def get_activations(self, as_numpy=True, keep_gradients=False):
        return self.activations

    def set_activations(self, values):
        self.activations = values


class _Prediction:
    def filter(self, class_id=None, confidence=None):
        return self

    def to_attribution_target(self, class_id=None):
        return self

    def to_batched_tensor(self):
        return np.ones((1, 1), dtype=np.float32)

    def __len__(self):
        return 1


class _Extractor:
    def __init__(self, latent_data, batch_size=1):
        self.latent_data = latent_data
        self.batch_size = batch_size
        self.forced_batch_sizes = []

    @contextmanager
    def temporary_force_batch_size(self, batch_size):
        old_batch_size = self.batch_size
        self.forced_batch_sizes.append(batch_size)
        self.batch_size = batch_size
        try:
            yield
        finally:
            self.batch_size = old_batch_size

    def input_to_latent_generator(self, inputs, resize=None, keep_gradients=False):
        yield from self.latent_data

    def latent_to_logit(self, latent_data):
        return _Prediction()


class _SemanticExtractor:
    batch_size = 2

    def input_to_latent_generator(self, inputs, resize=None, keep_gradients=False):
        del resize, keep_gradients
        inputs = np.asarray(inputs)
        concepts = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], 2))
        concepts[:, :2, :2, 0] = inputs[:, :2, :2, 0]
        concepts[:, 2:, 2:, 1] = inputs[:, 2:, 2:, 1]
        yield _LatentData(concepts.astype(np.float32))


class _Factorizer:
    is_fitted = True
    requires_positive_activations = False

    def encode(self, activations):
        return activations


class _UnfittedFactorizer(_Factorizer):
    is_fitted = False


class _NonInductiveFactorizer(_Factorizer):
    def encode(self, activations):
        raise NotImplementedError("out-of-sample encoding is not supported")


class _ArrayLike:
    def __init__(self, values):
        self.values = values


class _Explainer:
    def __init__(self, model, batch_size):
        pass

    def explain(self, coeffs_u, targets):
        return _ArrayLike(np.ones_like(coeffs_u))


class _BadShapeExplainer(_Explainer):
    def explain(self, coeffs_u, targets):
        return _ArrayLike(np.ones(coeffs_u.shape[:-1] + (1,)))


class _Framework:
    float32 = np.float32


class _NumpyConceptLocalizer(ConceptLocalizer):
    def __call__(self, inputs):
        return self._compute_scores(inputs)


class _Craft(HolisticCraft):
    def __init__(
        self,
        latent_data,
        batch_size=1,
        number_of_concepts=2,
        factorizer=None,
        extractor=None,
    ):
        factorizer = factorizer or _Factorizer()
        super().__init__(
            extractor or _Extractor(latent_data, batch_size),
            number_of_concepts=number_of_concepts,
            factorizer=factorizer,
        )
        concept_bank = np.eye(number_of_concepts, dtype=np.float32)
        self.factorization = Factorization(None, 0, None, factorizer, None, concept_bank)
        self.framework = "tf"
        self._framework_module = _Framework
        self.localizer_build_count = 0

    def latent_to_concept_differentiable(self, latent_data):
        return latent_data.activations

    def _to_numpy(self, tensor):
        return tensor if isinstance(tensor, np.ndarray) else tensor.values

    def _to_tensor(self, array, dtype=None):
        return np.asarray(array, dtype=dtype)

    def make_concept_decoder(self, latent_data):
        return object()

    def make_concept_localizer(self, concept_reducer="mean"):
        self.localizer_build_count += 1
        return _NumpyConceptLocalizer(self, concept_reducer)


class _DummyModel:
    def __call__(self, _):
        return np.ones((1, 1), dtype=np.float32)


class _MinimalStructuredPrediction:
    def __init__(self, empty):
        self.is_empty = empty

    def filter(self, class_id=None, confidence=None):
        return self

    def to_attribution_target(self, class_id=None):
        return self

    def to_batched_tensor(self):
        return np.ones((1, 1), dtype=np.float32)


def _make_spatial_craft(number_of_concepts=3, factorizer=None):
    activations = np.arange(2 * 2 * 2 * number_of_concepts, dtype=np.float32).reshape(
        2, 2, 2, number_of_concepts
    )
    return _Craft(
        [_LatentData(activations)],
        batch_size=2,
        number_of_concepts=number_of_concepts,
        factorizer=factorizer,
    )


def test_factorization_preserves_its_positional_field_order():
    factorization = Factorization("inputs", 3, "crops", "reducer", "crops_u", "concept_bank")

    assert factorization.inputs == "inputs"
    assert factorization.class_id == 3
    assert factorization.crops == "crops"
    assert factorization.reducer == "reducer"
    assert factorization.crops_u == "crops_u"
    assert factorization.concept_bank_w == "concept_bank"
    assert factorization.coeffs_u is None


def test_token_explanations_use_token_axes_and_framework_conversion():
    craft = _Craft([_LatentData(np.ones((1, 2, 2), dtype=np.float32))])
    explanation = craft.compute_explanation_per_concept(
        np.ones((1, 2, 2, 1)), PartialExplainer(_Explainer)
    )

    assert explanation.shape == (1, 2, 2)
    token_explanations = np.array([[[1.0, 0.0], [3.0, 0.0]], [[0.0, 2.0], [0.0, 4.0]]])
    np.testing.assert_allclose(
        craft.reduce_to_importance(token_explanations, spatial_reducer="mean"), [1.0, 1.5]
    )
    np.testing.assert_allclose(craft.reduce_to_prevalence(token_explanations), [0.5, 0.5])
    np.testing.assert_allclose(
        craft.reduce_to_reliability(token_explanations, np.array([1.0, 0.5])), [1.0, 0.5]
    )


def test_explanations_use_configured_batch_size_for_perturbations():
    explainer_batch_sizes = []

    class RecordingExplainer(_Explainer):
        def __init__(self, model, batch_size):
            super().__init__(model, batch_size)
            explainer_batch_sizes.append(batch_size)

    craft = _Craft([_LatentData(np.ones((1, 2, 2), dtype=np.float32))], batch_size=4)
    explanation = craft.compute_explanation_per_concept(
        np.ones((1, 2, 2, 1)), PartialExplainer(RecordingExplainer)
    )

    assert explanation.shape == (1, 2, 2)
    assert explainer_batch_sizes == [4]
    assert craft.latent_extractor.forced_batch_sizes == [1]
    assert craft.latent_extractor.batch_size == 4


def test_invalid_explanation_shape_and_empty_extractions_raise_value_errors():
    craft = _Craft([_LatentData(np.ones((1, 2, 2), dtype=np.float32))])

    with pytest.raises(ValueError, match="Explanation shape"):
        craft.compute_explanation_per_concept(
            np.ones((1, 2, 2, 1)), PartialExplainer(_BadShapeExplainer)
        )

    empty_craft = _Craft([])
    with pytest.raises(ValueError, match="No activations"):
        empty_craft.fit(np.ones((1, 2, 2, 1)))
    with pytest.raises(ValueError, match="No activations"):
        empty_craft.transform(np.ones((1, 2, 2, 1)))
    with pytest.raises(ValueError, match="No latent data"):
        empty_craft.compute_explanation_per_concept(
            np.ones((1, 2, 2, 1)), PartialExplainer(_Explainer)
        )


def test_check_if_fitted_requires_factorization_and_fitted_factorizer():
    craft = _Craft([])
    craft.factorizer = _UnfittedFactorizer()
    craft.factorization = Factorization(None, 0, None, craft.factorizer, None, np.eye(2))

    with pytest.raises(NotFittedError):
        craft.check_if_fitted()

    craft.factorizer = _Factorizer()
    craft.factorization = None
    with pytest.raises(NotFittedError):
        craft.check_if_fitted()


def test_estimate_importance_accepts_operator_in_method_kwargs():
    craft = _Craft([_LatentData(np.ones((1, 2, 2), dtype=np.float32))])
    captured = {}

    def _capture_partial_explainer(
        _, partial_explainer, class_id=None, confidence=None, verbose=False
    ):
        captured.update(partial_explainer.kwargs)
        return np.ones((1, 2), dtype=np.float32)

    craft.compute_explanation_per_concept = _capture_partial_explainer

    craft.estimate_importance(
        images=np.ones((1, 2, 2, 1), dtype=np.float32),
        operator=_DummyModel(),
        class_id=0,
        method="gradient_input",
        reducer="sum",
        spatial_reducer="mean",
        aggregation_reducer="mean",
    )
    assert isinstance(captured["operator"], _DummyModel)
    assert captured["reducer"] == "sum"

    captured.clear()
    craft.estimate_importance(
        images=np.ones((1, 2, 2, 1), dtype=np.float32),
        operator=_DummyModel(),
        class_id=0,
        method="sobol",
        nb_channels=7,
        spatial_reducer="mean",
        aggregation_reducer="mean",
    )
    assert isinstance(captured["operator"], _DummyModel)
    assert captured["nb_channels"] == 7


def test_compute_explanation_accepts_structured_predictions_without_len():
    craft = _Craft([_LatentData(np.ones((1, 2, 2), dtype=np.float32))])

    craft.decode = lambda latent_data, coeffs_u: _MinimalStructuredPrediction(empty=True)
    explanation = craft.compute_explanation_per_concept(
        np.ones((1, 2, 2, 1), dtype=np.float32),
        PartialExplainer(_Explainer),
    )
    assert explanation.shape == (1, 2, 2)

    craft.decode = lambda latent_data, coeffs_u: _MinimalStructuredPrediction(empty=False)
    explanation = craft.compute_explanation_per_concept(
        np.ones((1, 2, 2, 1), dtype=np.float32),
        PartialExplainer(_Explainer),
    )
    assert explanation.shape == (1, 2, 2)


def test_display_validates_concept_order_and_handles_a_single_column():
    craft = _Craft([])
    images = np.ones((2, 4, 4, 3), dtype=np.float32)
    coeffs_u = np.ones((2, 2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="between 0"):
        craft.display_images_per_concept(images, coeffs_u, order=[2])
    with pytest.raises(ValueError, match="more IDs"):
        craft.display_images_per_concept(images, coeffs_u, order=[0, 1, 0])

    figure = craft.display_images_per_concept(images, coeffs_u, order=[0])
    assert len(figure.axes) == 2
    plt.close(figure)

    figure = craft.display_top_images_per_concept(images, topk=2, coeffs_u=coeffs_u, order=[0])
    assert len(figure.axes) == 2
    plt.close(figure)


def test_concept_localizer_reducers_handle_spatial_and_global_coefficients():
    craft = _make_spatial_craft(number_of_concepts=3)
    localizer = craft.make_concept_localizer("mean")
    inputs = np.ones((2, 4, 4, 3), dtype=np.float32)

    scores_mean = localizer(inputs)
    expected_mean = np.mean(craft.transform(inputs), axis=(1, 2))
    np.testing.assert_allclose(scores_mean, expected_mean)
    assert scores_mean.dtype == np.float32

    scores_sum = craft.make_concept_localizer("sum")(inputs)
    expected_sum = np.sum(craft.transform(inputs), axis=(1, 2))
    np.testing.assert_allclose(scores_sum, expected_sum)

    scores_max = craft.make_concept_localizer("max")(inputs)
    expected_max = np.max(craft.transform(inputs), axis=(1, 2))
    np.testing.assert_allclose(scores_max, expected_max)

    tokens = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    token_scores = localizer._reduce_coefficients(tokens)
    np.testing.assert_allclose(token_scores, np.mean(tokens, axis=1))

    globals_only = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    global_scores = localizer._reduce_coefficients(globals_only)
    np.testing.assert_allclose(global_scores, globals_only)


def test_concept_localizer_reducer_validation_and_shape_errors():
    craft = _make_spatial_craft(number_of_concepts=3)

    with pytest.raises(ValueError, match="concept_reducer"):
        craft.make_concept_localizer("median")

    with pytest.raises(ValueError, match="concept_reducer"):
        craft.make_concept_localizer(42)

    callable_localizer = craft.make_concept_localizer(
        lambda coeffs: np.mean(np.abs(coeffs), axis=(1, 2), dtype=np.float64)
    )
    callable_inputs = np.ones((2, 4, 4, 3), dtype=np.float32)
    callable_scores = callable_localizer(callable_inputs)
    assert callable_scores.dtype == np.float32
    np.testing.assert_allclose(
        callable_scores,
        np.mean(np.abs(craft.transform(callable_inputs)), axis=(1, 2)),
    )

    with pytest.raises(ValueError, match="must have shape"):
        craft.make_concept_localizer(lambda coeffs: np.mean(coeffs, axis=(0, 1, 2)))(
            np.ones((2, 4, 4, 3), dtype=np.float32)
        )

    with pytest.raises(ValueError, match="empty dimensions"):
        craft.make_concept_localizer("mean")._reduce_coefficients(
            np.empty((0, 2, 2, 3), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="empty dimensions"):
        craft.make_concept_localizer("max")._reduce_coefficients(
            np.empty((2, 0, 2, 3), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="finite"):
        craft.make_concept_localizer(lambda coeffs: np.full((2, 3), np.nan))(
            np.ones((2, 4, 4, 3), dtype=np.float32)
        )

    localizer = craft.make_concept_localizer("mean")
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        localizer._reduce_coefficients(np.array([1.0, 2.0], dtype=np.float32))


def test_compute_concept_attributions_orchestration_and_targets():
    craft = _make_spatial_craft(number_of_concepts=3)
    images = np.ones((2, 4, 4, 3), dtype=np.float32)

    records = {"models": [], "batch_sizes": [], "inputs": [], "targets": []}

    class RecordingExplainer:
        def __init__(self, model, batch_size):
            self.model = model
            records["models"].append(model)
            records["batch_sizes"].append(batch_size)

        def explain(self, inputs, targets):
            concept_id = int(np.argmax(targets[0]))
            records["inputs"].append(inputs.copy())
            records["targets"].append(targets.copy())
            scores = self.model(inputs)
            assert scores.shape == (inputs.shape[0], craft.number_of_concepts)
            return np.full(inputs.shape[:3], fill_value=float(concept_id), dtype=np.float32)

    maps = craft.compute_concept_attributions(
        images,
        partial_explainer=PartialExplainer(RecordingExplainer),
        concept_ids=[2, 0],
    )

    assert craft.localizer_build_count == 1
    assert len(records["models"]) == 1
    assert records["batch_sizes"] == [craft.batch_size]
    assert len(records["inputs"]) == 2
    np.testing.assert_array_equal(records["inputs"][0], images)
    np.testing.assert_array_equal(records["inputs"][1], images)
    assert len(records["targets"]) == 2
    np.testing.assert_array_equal(records["targets"][0], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(records["targets"][1], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    assert maps.shape == (2, 4, 4, 3)
    assert np.all(np.isnan(maps[..., 1]))
    assert np.all(np.isfinite(maps[..., 0]))
    assert np.all(np.isfinite(maps[..., 2]))
    np.testing.assert_allclose(maps[..., 2], 2.0)
    np.testing.assert_allclose(maps[..., 0], 0.0)


def test_compute_concept_attributions_validates_inputs_and_explainer_type():
    craft = _make_spatial_craft(number_of_concepts=3)
    images = np.ones((2, 4, 4, 3), dtype=np.float32)

    class ShapeExplainer:
        def __init__(self, model, batch_size):
            del model, batch_size

        def explain(self, inputs, targets):
            del targets
            return np.ones((inputs.shape[0], inputs.shape[1], inputs.shape[2], 2), dtype=np.float32)

    with pytest.raises(TypeError, match="PartialExplainer"):
        craft.compute_concept_attributions(images, partial_explainer=ShapeExplainer)

    with pytest.raises(ValueError, match="between 0"):
        craft.compute_concept_attributions(images, PartialExplainer(_Explainer), concept_ids=[3])
    assert craft.localizer_build_count == 0
    with pytest.raises(ValueError, match="duplicate"):
        craft.compute_concept_attributions(images, PartialExplainer(_Explainer), concept_ids=[1, 1])
    assert craft.localizer_build_count == 0
    with pytest.raises(ValueError, match="at least one"):
        craft.compute_concept_attributions(images, PartialExplainer(_Explainer), concept_ids=[])
    assert craft.localizer_build_count == 0
    with pytest.raises(ValueError, match="between 0"):
        craft.compute_concept_attributions(images, PartialExplainer(_Explainer), concept_ids=[True])
    assert craft.localizer_build_count == 0
    with pytest.raises(ValueError, match="one channel"):
        craft.compute_concept_attributions(images, PartialExplainer(ShapeExplainer))

    class NonFiniteExplainer(ShapeExplainer):
        def explain(self, inputs, targets):
            del targets
            return np.full(inputs.shape[:3], np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        craft.compute_concept_attributions(images, PartialExplainer(NonFiniteExplainer))


@pytest.mark.parametrize(
    "images",
    [
        np.empty((0, 4, 4, 3), dtype=np.float32),
        np.empty((2, 4), dtype=np.float32),
        np.empty((2, 0, 4, 3), dtype=np.float32),
        [],
    ],
)
def test_compute_concept_attributions_rejects_invalid_image_batches(images):
    craft = _make_spatial_craft(number_of_concepts=3)

    with pytest.raises(ValueError, match="images"):
        craft.compute_concept_attributions(images, PartialExplainer(_Explainer), concept_ids=[0])
    assert craft.localizer_build_count == 0


def test_compute_concept_attributions_rejects_explicit_operators_before_setup():
    craft = _make_spatial_craft(number_of_concepts=3)
    images = np.ones((2, 4, 4, 3), dtype=np.float32)
    partial_explainer = PartialExplainer(
        _Explainer,
        operator=lambda model, inputs, targets: model(inputs),
    )

    with pytest.raises(ValueError, match="does not accept a custom operator"):
        craft.compute_concept_attributions(images, partial_explainer, concept_ids=[0])
    assert craft.localizer_build_count == 0


def test_compute_concept_attributions_all_concepts_are_finite_by_default():
    craft = _make_spatial_craft(number_of_concepts=3)
    images = np.ones((2, 4, 4, 3), dtype=np.float32)

    class ConstantExplainer:
        def __init__(self, model, batch_size):
            del model, batch_size

        def explain(self, inputs, targets):
            concept_id = int(np.argmax(targets[0]))
            return np.full(inputs.shape[:3], concept_id + 1.0, dtype=np.float32)

    maps = craft.compute_concept_attributions(images, PartialExplainer(ConstantExplainer))
    assert maps.shape == (2, 4, 4, 3)
    assert np.all(np.isfinite(maps))
    np.testing.assert_allclose(maps[..., 0], 1.0)
    np.testing.assert_allclose(maps[..., 1], 2.0)
    np.testing.assert_allclose(maps[..., 2], 3.0)


def test_compute_concept_attributions_rejects_whitebox_explainers_and_non_inductive_factorizer():
    craft = _make_spatial_craft(number_of_concepts=3)
    images = np.ones((2, 4, 4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="black-box attribution"):
        craft.compute_concept_attributions(images, PartialExplainer(GradientInput))

    craft_non_inductive = _make_spatial_craft(
        number_of_concepts=3,
        factorizer=_NonInductiveFactorizer(),
    )

    class PassThroughExplainer:
        def __init__(self, model, batch_size):
            self.model = model

        def explain(self, inputs, targets):
            del targets
            self.model(inputs)
            return np.ones(inputs.shape[:3], dtype=np.float32)

    with pytest.raises(RuntimeError, match="cannot encode unseen activations"):
        craft_non_inductive.compute_concept_attributions(
            images,
            PartialExplainer(PassThroughExplainer),
            concept_ids=[0],
        )

    class ExplainerNotImplemented:
        def __init__(self, model, batch_size):
            del model, batch_size

        def explain(self, inputs, targets):
            del inputs, targets
            raise NotImplementedError("explainer operation is unavailable")

    with pytest.raises(NotImplementedError, match="explainer operation"):
        craft.compute_concept_attributions(
            images,
            PartialExplainer(ExplainerNotImplemented),
            concept_ids=[0],
        )


def test_display_accepts_concept_maps_and_preserves_ranking_behavior():
    craft = _make_spatial_craft(number_of_concepts=3)
    images = np.ones((2, 4, 4, 3), dtype=np.float32)
    coeffs_u = np.zeros((2, 2, 2, 3), dtype=np.float32)
    coeffs_u[0, :, :, 0] = 10.0
    coeffs_u[1, :, :, 0] = 1.0

    concept_maps = np.full((2, 4, 4, 3), np.nan, dtype=np.float32)
    concept_maps[..., 0] = 0.0
    concept_maps[..., 2] = 2.0

    displayed_maps = []

    def fake_display_concept_heatmap(image, concept_heatmap, concept_idx, ax, **kwargs):
        del ax, kwargs
        displayed_maps.append((image.copy(), concept_idx, np.array(concept_heatmap)))

    craft.display_concept_heatmap = fake_display_concept_heatmap

    legacy_figure = craft.display_images_per_concept(images, coeffs_u=coeffs_u, order=[0])
    assert [entry[1] for entry in displayed_maps] == [0, 0]
    np.testing.assert_allclose([entry[2].max() for entry in displayed_maps], [10.0, 1.0])
    plt.close(legacy_figure)
    displayed_maps.clear()

    def fail_if_transform_called(_):
        raise AssertionError("map-only display must not recompute concept coefficients")

    craft.transform = fail_if_transform_called
    figure = craft.display_images_per_concept(images, coeffs_u=coeffs_u, concept_maps=concept_maps)
    assert len(figure.axes) == 4
    assert [entry[1] for entry in displayed_maps] == [0, 0, 2, 2]
    np.testing.assert_allclose([entry[2].max() for entry in displayed_maps], [0.0, 0.0, 2.0, 2.0])
    plt.close(figure)

    with pytest.raises(ValueError, match="not available"):
        craft.display_images_per_concept(
            images,
            coeffs_u=coeffs_u,
            concept_maps=concept_maps,
            order=[1],
        )

    partially_invalid_maps = concept_maps.copy()
    partially_invalid_maps[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="entirely finite or entirely NaN"):
        craft.display_images_per_concept(
            images,
            concept_maps=partially_invalid_maps,
            order=[0],
        )

    rank_calls = []

    def recording_topk(captured_coeffs, topk):
        rank_calls.append(np.array(captured_coeffs))
        return np.array([[0], [0], [0]], dtype=int)

    craft.get_topk_images_per_concept = recording_topk
    figure = craft.display_top_images_per_concept(
        images,
        topk=1,
        coeffs_u=coeffs_u,
        concept_maps=concept_maps,
        order=[0],
    )
    assert len(rank_calls) == 1
    np.testing.assert_array_equal(rank_calls[0], coeffs_u)
    assert displayed_maps[-1][1] == 0
    np.testing.assert_allclose(displayed_maps[-1][2], 0.0)
    plt.close(figure)


def test_display_concept_heatmap_resizing_and_finite_validation(monkeypatch):
    craft = _make_spatial_craft(number_of_concepts=3)
    image = np.ones((4, 4, 3), dtype=np.float32)
    figure, ax = plt.subplots(1, 1)

    resize_calls = {"count": 0}

    def recording_resize(*args, **kwargs):
        resize_calls["count"] += 1
        return np.ones((4, 4, 1), dtype=np.float32)

    monkeypatch.setattr("xplique.concepts.holistic_craft.cv2.resize", recording_resize)

    craft.display_concept_heatmap(image, np.ones((4, 4), dtype=np.float32), concept_idx=0, ax=ax)
    assert resize_calls["count"] == 0

    craft.display_concept_heatmap(image, np.ones((2, 2), dtype=np.float32), concept_idx=0, ax=ax)
    assert resize_calls["count"] == 1

    craft.display_concept_heatmap(
        image,
        np.ones((4, 4, 1), dtype=np.float32),
        concept_idx=0,
        ax=ax,
    )

    with pytest.raises(ValueError, match="only finite"):
        craft.display_concept_heatmap(
            image,
            np.where(np.indices((4, 4))[0] == 0, np.nan, 1.0).astype(np.float32),
            concept_idx=0,
            ax=ax,
        )
    plt.close(figure)


def test_semantic_localization_follows_spatial_concept_dependencies():
    craft = _Craft(
        latent_data=None,
        number_of_concepts=2,
        extractor=_SemanticExtractor(),
    )
    images = np.ones((1, 4, 4, 2), dtype=np.float32)

    class PixelOcclusionExplainer:
        def __init__(self, model, batch_size):
            del batch_size
            self.model = model

        def explain(self, inputs, targets):
            concept_id = int(np.argmax(targets[0]))
            base_scores = self.model(inputs)
            maps = np.zeros(inputs.shape[:3], dtype=np.float32)
            for row in range(inputs.shape[1]):
                for column in range(inputs.shape[2]):
                    perturbed = inputs.copy()
                    perturbed[:, row, column, :] = 0.0
                    perturbed_scores = self.model(perturbed)
                    maps[:, row, column] = (
                        base_scores[:, concept_id] - perturbed_scores[:, concept_id]
                    )
            return maps

    maps = craft.compute_concept_attributions(
        images,
        partial_explainer=PartialExplainer(PixelOcclusionExplainer),
    )

    assert maps.shape == (1, 4, 4, 2)
    assert np.all(np.isfinite(maps))

    concept_0_inside = maps[0, :2, :2, 0].sum()
    concept_0_outside = maps[0, :, :, 0].sum() - concept_0_inside
    concept_1_inside = maps[0, 2:, 2:, 1].sum()
    concept_1_outside = maps[0, :, :, 1].sum() - concept_1_inside
    assert concept_0_inside > concept_0_outside
    assert concept_1_inside > concept_1_outside
