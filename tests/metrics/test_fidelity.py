import tensorflow as tf
import numpy as np

from tests.utils import (generate_model, generate_regression_model, generate_timeseries_model, 
                     generate_data, almost_equal)
from xplique.attributions import Saliency, GradCAM, Occlusion
from xplique.metrics import (Insertion, Deletion, MuFidelity,
                             AverageDropMetric, AverageIncreaseMetric, AverageGainMetric)
from xplique.metrics.base import ExplanationMetric
from xplique.metrics.fidelity import BaseAverageXMetric


def test_mu_fidelity():
    # ensure we can compute the metric with consistent arguments
    input_shape, nb_labels, nb_samples = ((32, 32, 3), 10, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_model(input_shape, nb_labels)
    explanations = np.random.uniform(0, 1, x.shape[:-1] + (1,)).astype(np.float32)

    nb_estimation = 10 # number of samples to test correlation for each samples

    for grid_size in [None, 5]:
        for subset_percent in [0.1, 0.9]:
            for baseline_mode in [0.0, lambda x : x-0.5]:
                score = MuFidelity(model, x, y, grid_size=grid_size,
                                   subset_percent=subset_percent,
                                   baseline_mode=baseline_mode,
                                   nb_samples=nb_estimation)(explanations)
                assert -1.0 < score < 1.0


def test_causal_metrics():
    # ensure we can compute insertion/deletion metric with consistent arguments
    input_shape, nb_labels, nb_samples = ((10, 10, 3), 10, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_model(input_shape, nb_labels)
    explanations = np.random.uniform(0, 1, x.shape[:-1] + (1,)).astype(np.float32)

    for step in [-1, 5, 10]:
        for baseline_mode in [0.0, lambda x: x-0.5]:
            score_insertion = Insertion(model, x, y,
                                        baseline_mode=baseline_mode,
                                        steps=step)(explanations)
            score_deletion = Deletion(model, x, y,
                                      baseline_mode=baseline_mode,
                                      steps=step)(explanations)

            for score in [score_insertion, score_deletion]:
                assert 0.0 <= score <= 1.0


def test_perfect_correlation():
    """Ensure we get perfect score if the correlation is perfect"""
    # we ensure perfect correlation if the model return the sum of the input,
    # and the input is the explanations: corr( sum(phi), sum(x) - sum(x-phi) )
    # to do so we define f(x) -> sum(x) and phi = x
    nb_classes = 2

    input_shape, nb_labels, nb_samples = ((32, 32, 1), nb_classes, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = lambda x: tf.repeat(tf.reduce_sum(x, axis=(1, 2, 3))[:, None], nb_classes, -1)
    explanations = x
    
    perfect_score = MuFidelity(model, x, y, grid_size=None,
                               subset_percent=0.1,
                               baseline_mode=0.0,
                               nb_samples=200)(explanations)
    assert almost_equal(perfect_score, 1.0)


def test_worst_correlation():
    """Ensure we get worst score if the correlation is inversed"""
    # we ensure worst correlation if the model return the -sum of the input,
    # and the input is the explanations: corr( sum(phi), sum(x) - sum(x-phi) )
    # to do so we define f(x) -> sum(x) and phi = x
    nb_classes = 2

    input_shape, nb_labels, nb_samples = ((32, 32, 1), nb_classes, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = lambda x: tf.repeat(tf.reduce_sum(x, axis=(1, 2, 3))[:, None], nb_classes, -1)
    explanations = -x

    perfect_score = MuFidelity(model, x, y, grid_size=None,
                               subset_percent=0.1,
                               baseline_mode=0.0,
                               nb_samples=200)(explanations)
    assert almost_equal(perfect_score, -1.0)


def test_perfect_deletion():
    """Ensure we get perfect score if the model is sensible to deletion"""
    # we ensure perfect deletion if the model return 0.0 as soon as there is
    # one element set to baseline
    dim = 16
    steps = dim**2

    input_shape, nb_labels, nb_samples = ((dim, dim, 1), 2, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)

    model = lambda x: 1.0 - tf.reduce_max(tf.cast(x == 0.0, tf.float32), (1, 2))
    explanations = x

    perfect_score = Deletion(model, x, y, steps=steps)(explanations)
    assert almost_equal(perfect_score, 0.0, 1e-2)


def test_perfect_insertion():
    """Ensure we get perfect score if the model is sensible to insertion"""
    # we ensure perfect deletion if the model return 1.0 as soon as there is
    # one element to non-baseline
    dim = 16
    steps = dim**2

    input_shape, nb_labels, nb_samples = ((dim, dim, 1), 2, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)

    model = lambda x: tf.reduce_max(tf.cast(x != 0.0, tf.float32), (1, 2))
    explanations = x

    perfect_score = Insertion(model, x, y, steps=steps)(explanations)
    assert almost_equal(perfect_score, 1.0, 1e-2)


def test_MuFidelity_batch_size():
    """Ensure we get perfect score if the correlation is perfect"""
    # we ensure perfect correlation if the model return the sum of the input,
    # and the input is the explanations: corr( sum(phi), sum(x) - sum(x-phi) )
    # to do so we define f(x) -> sum(x) and phi = x
    for batch_size in [1, 4, 9, None]:
        input_shape, nb_labels, nb_samples = ((5, 5, 1), 2, 3)
        x = tf.reshape(tf.range(nb_samples * np.prod(input_shape), dtype=tf.float32),
                       (nb_samples, *input_shape))
        y = tf.concat([tf.zeros((nb_samples, nb_labels - 1)), tf.ones((nb_samples, 1))], axis=1)
        model = lambda x: tf.repeat(tf.reduce_sum(x, axis=(1, 2, 3))[:, None], nb_labels, -1)
        explanations = x

        perfect_score = MuFidelity(model, x, y, grid_size=None,
                                subset_percent=0.2,
                                baseline_mode=0.0,
                                batch_size=batch_size,
                                nb_samples=7)(explanations)
        assert almost_equal(perfect_score, 1.0)


def test_average_metric_perturb_with_mask_invariant_to_sign():
    """Mask should be built from |explanations| and min-max normalized per sample."""
    batch, h, w, c = 2, 2, 2, 3
    # inputs = ones: perturbed == mask
    inputs = tf.ones((batch, h, w, c), dtype=tf.float32)

    # some heterogeneous explanations
    exps = tf.constant(
        [
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            ],
            [
                [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
                [[7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]],
            ],
        ],
        dtype=tf.float32,
    )

    pert = BaseAverageXMetric._perturb_with_mask(inputs, exps)
    pert_neg = BaseAverageXMetric._perturb_with_mask(inputs, -exps)

    # sign invariance
    np.testing.assert_allclose(pert.numpy(), pert_neg.numpy(), rtol=1e-6, atol=1e-6)

    # since inputs are ones, perturbed == mask (after broadcasting)
    mask = pert.numpy()

    # mask must be in [0, 1] (up to numerical epsilon)
    assert np.all(mask >= -1e-7)
    assert np.all(mask <= 1.0 + 1e-7)

    # per-sample min ~= 0, max ~= 1
    for b in range(batch):
        mb = mask[b]
        m_min = mb.min()
        m_max = mb.max()
        assert np.isclose(m_min, 0.0, atol=1e-5)
        assert np.isclose(m_max, 1.0, atol=1e-5)

    # channel-averaged explanations: mask values should be identical across channels
    # (because mask is broadcast on last axis)
    # shape: (B, H, W, 1) after broadcasting
    # so for each (b, i, j) all channels share same value
    for b in range(batch):
        for i in range(h):
            for j in range(w):
                v = mask[b, i, j, :]
                assert np.allclose(v, v[0], atol=1e-6)


def test_perturb_with_mask_timeseries_broadcast():
    """For time-series with rank-2 explanations, mask should be broadcast over features."""
    # inputs: (B, T, D)
    B, T, D = 1, 5, 3
    inputs = tf.ones((B, T, D), dtype=tf.float32)
    # explanations: (B, T) -> will be broadcast to (B, T, 1)
    exps = tf.constant([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=tf.float32)

    pert = BaseAverageXMetric._perturb_with_mask(inputs, exps).numpy()
    # for each timestep, all features should share the same scaling factor
    for t in range(T):
        row = pert[0, t, :]
        assert np.allclose(row, row[0], atol=1e-6)

    # mask still normalized per-sample in [0, 1]
    min_val = pert.min()
    max_val = pert.max()
    assert min_val >= -1e-7
    assert max_val <= 1.0 + 1e-7


def test_average_metrics_common_api():
    """
    New metrics should conform to the same API as existing explanation metrics:
    - accept numpy / tf / dataset inputs
    - work with a standard explainer
    - return python floats
    """
    from xplique.attributions import Saliency

    input_shape, nb_labels, samples = ((16, 16, 3), 10, 20)
    model = generate_model(input_shape, nb_labels)
    explainers = [Saliency(model), GradCAM(model), Occlusion(model)]

    inputs_np, targets_np = generate_data(input_shape, nb_labels, samples)
    inputs_tf = tf.cast(inputs_np, tf.float32)
    targets_tf = tf.cast(targets_np, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np))
    batched_dataset = dataset.batch(4)

    metrics_classes = [AverageDropMetric, AverageIncreaseMetric, AverageGainMetric]

    for inputs, targets in [
        (inputs_np, targets_np),
        (inputs_tf, targets_tf),
        (dataset, None),
        (batched_dataset, None),
    ]:
        for explainer in explainers:
            explanations = explainer(inputs, targets)

            for Metric in metrics_classes:
                metric = Metric(model, inputs, targets, batch_size=4)
                assert isinstance(metric, ExplanationMetric)
                score = metric(explanations)
                assert type(score) in (float, np.float32, np.float64)


def test_average_metrics_add_activation():
    """
    Check that AverageDrop/Increase/Gain work with activation='sigmoid'/'softmax'
    and outputs stay in [0,1].
    """
    input_shape, nb_labels, samples = ((16, 16, 3), 5, 8)
    x, y = generate_data(input_shape, nb_labels, samples)
    model = generate_model(input_shape, nb_labels)

    explainer = Saliency(model)
    explanations = explainer(x, y)

    activations = ["sigmoid", "softmax"]
    metrics_classes = [AverageDropMetric, AverageIncreaseMetric, AverageGainMetric]

    for activation in activations:
        for Metric in metrics_classes:
            metric = Metric(model, x, y, batch_size=4, activation=activation)
            score = metric(explanations)
            assert type(score) in (float, np.float32, np.float64)
            # with probabilities in [0,1], all three metrics must be in [0,1]
            assert 0.0 <= score <= 1.0


def test_average_metrics_data_types_and_shapes():
    """
    Ensure average metrics work for tabular, time-series and different image shapes.
    """
    data_types_input_shapes = {
        "tabular": (20,),
        "time-series": (20, 10),
        "images rgb": (20, 16, 3),
        "images black and white": (28, 28, 1),
    }

    metrics_classes = [AverageDropMetric, AverageIncreaseMetric, AverageGainMetric]

    for _, input_shape in data_types_input_shapes.items():
        input_shape, nb_labels, samples = (input_shape, 5, 15)
        inputs, targets = generate_data(input_shape, nb_labels, samples)

        # choose model type depending on data shape (same pattern as existing tests)
        if len(input_shape) == 3:  # image => conv2D classifier
            model = generate_model(input_shape, nb_labels)
            # make last layer linear to avoid implicit softmax
            model.layers[-1].activation = tf.keras.activations.linear
        else:  # others => dense regression model
            model = generate_regression_model(input_shape, nb_labels)

        explainer = Saliency(model)
        explanations = explainer(inputs, targets)

        for Metric in metrics_classes:
            metric = Metric(model, inputs, targets)
            score = metric(explanations)
            assert type(score) in (float, np.float32, np.float64)


def test_average_metrics_constant_operator_yields_zero():
    """
    Using an operator that is constant w.r.t. inputs should give identically zero scores
    for Drop / Increase / Gain (base == after for all samples).
    """
    def constant_operator(model, inputs, targets):
        # returns a constant scalar per sample, independent of inputs/targets
        batch_size = tf.shape(inputs)[0]
        return tf.ones((batch_size,), dtype=tf.float32)

    input_shape, nb_labels, samples = ((8, 8, 1), 2, 6)
    x, y = generate_data(input_shape, nb_labels, samples)

    # dummy model, never used in operator
    model = lambda z: tf.ones((tf.shape(z)[0], nb_labels), dtype=tf.float32)

    explanations = np.random.randn(*x.shape).astype(np.float32)

    for Metric in (AverageDropMetric, AverageIncreaseMetric, AverageGainMetric):
        metric = Metric(
            model=model,
            inputs=x,
            targets=y,
            batch_size=3,
            operator=constant_operator,
        )
        score = metric(explanations)
        assert almost_equal(score, 0.0, epsilon=1e-6)


def test_average_drop_manual_match_with_custom_operator():
    """
    For a custom operator equal to the per-sample mean of inputs, AverageDropMetric
    should match a manual implementation using the same operator, mask, and formula.
    """

    def mean_operator(model, inputs, targets):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        flat = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
        return tf.reduce_mean(flat, axis=1)

    # positive inputs to keep everything nicely behaved
    B, H, W, C = 4, 4, 4, 1
    x = np.random.uniform(0.1, 1.0, size=(B, H, W, C)).astype(np.float32)
    y = np.eye(2, dtype=np.float32)[np.zeros(B, dtype=np.int32)]  # dummy targets
    explanations = np.random.uniform(0.0, 1.0, size=(B, H, W, C)).astype(np.float32)

    model = lambda z: tf.ones((tf.shape(z)[0], 2), dtype=tf.float32)

    metric = AverageDropMetric(
        model=model,
        inputs=x,
        targets=y,
        batch_size=2,
        operator=mean_operator,
    )

    score = metric(explanations)

    # manual implementation using the same ingredients
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    phi_tf = tf.convert_to_tensor(explanations, dtype=tf.float32)

    base = mean_operator(model, x_tf, y).numpy()
    pert = BaseAverageXMetric._perturb_with_mask(x_tf, phi_tf)
    after = mean_operator(model, pert, y).numpy()

    _EPS = 1e-8
    manual_ad = np.maximum(base - after, 0.0) / (base + _EPS)
    manual_score = float(manual_ad.mean())

    assert almost_equal(score, manual_score, epsilon=1e-5)


def test_average_metrics_batch_size_invariance():
    """
    AverageDrop / Increase / Gain should be invariant (up to numerical noise)
    w.r.t. the batch_size used internally by BaseAverageXMetric.evaluate.
    """

    def mean_operator(model, inputs, targets):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        flat = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
        return tf.reduce_mean(flat, axis=1)

    input_shape, nb_labels, samples = ((6, 6, 1), 2, 7)
    x, y = generate_data(input_shape, nb_labels, samples)
    explanations = np.random.rand(*x.shape).astype(np.float32)
    model = lambda z: tf.ones((tf.shape(z)[0], nb_labels), dtype=tf.float32)

    batch_sizes = [1, 2, 4, None]
    metrics = (AverageDropMetric, AverageIncreaseMetric, AverageGainMetric)

    for Metric in metrics:
        scores = []
        for bs in batch_sizes:
            metric = Metric(
                model=model,
                inputs=x,
                targets=y,
                batch_size=bs,
                operator=mean_operator,
            )
            s = metric(explanations)
            scores.append(s)

        # all scores for different batch_sizes should be very close
        for s in scores[1:]:
            assert almost_equal(scores[0], s, epsilon=1e-5)
