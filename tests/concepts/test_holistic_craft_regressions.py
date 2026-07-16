"""Regression tests for framework-agnostic HolisticCraft behavior."""

from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from xplique.concepts.craft import Factorization
from xplique.concepts.holistic_craft import HolisticCraft, PartialExplainer
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
    batch_size = 1

    def __init__(self, latent_data):
        self.latent_data = latent_data

    @contextmanager
    def temporary_force_batch_size(self, batch_size):
        yield

    def input_to_latent_generator(self, inputs, resize=None, keep_gradients=False):
        yield from self.latent_data

    def latent_to_logit(self, latent_data):
        return _Prediction()


class _Factorizer:
    is_fitted = True
    requires_positive_activations = False

    def encode(self, activations):
        return activations


class _UnfittedFactorizer(_Factorizer):
    is_fitted = False


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


class _Craft(HolisticCraft):
    def __init__(self, latent_data):
        factorizer = _Factorizer()
        super().__init__(_Extractor(latent_data), number_of_concepts=2, factorizer=factorizer)
        self.factorization = Factorization(None, 0, None, factorizer, None, np.eye(2))
        self.framework = "tf"
        self._framework_module = _Framework

    def latent_to_concept_differentiable(self, latent_data):
        return latent_data.activations

    def _to_numpy(self, tensor):
        return tensor if isinstance(tensor, np.ndarray) else tensor.values

    def _to_tensor(self, array, dtype=None):
        return np.asarray(array, dtype=dtype)

    def make_concept_decoder(self, latent_data):
        return object()


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
