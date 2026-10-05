# KernelBanzhaf

KernelBanzhaf estimates signed effects of already-encoded concept channels by centered,
full-column-rank least squares under uniform binary coalitions. It inherits from
[Banzhaf](banzhaf.md), with the same constructor defaults, support, masks, fixed-target scalar
scores, output layout, and seed semantics. Only the estimator and its rank requirements differ.
It is not KernelSHAP: effects are in score units, not Shapley values, and need not sum to the
full-input minus zero-input score.

## API

```python
KernelBanzhaf(model, batch_size=32, operator=None, nb_samples=1024, seed=0)
```

{{xplique.attributions.KernelBanzhaf}}

## Shared contract

The [shared concept-channel contract](../api_attributions.md#shared-concept-channel-contract)
applies unchanged, as do the Banzhaf [`nb_samples` design](banzhaf.md#nb_samples): for `d`
active channels, all `2**d` masks are enumerated if they fit in the positive even budget;
otherwise `nb_samples // 2` independent Bernoulli(0.5) masks are sampled and their complements
appended. Exact masks follow increasing binary integers, with the lowest active channel as the
least significant bit, and exact enumeration is seed-independent.

## Estimator

For one nonempty input, let `M` be the evaluated binary mask matrix of shape `(Q, d)` and `Y`
the corresponding fixed-target scalar scores. Both exact and antithetic designs have column
means of `0.5`. KernelBanzhaf solves the centered least-squares problem

$$
X = M - \tfrac{1}{2}, \qquad y = Y - \operatorname{mean}(Y),
\qquad \widehat{\beta} = \arg\min_{\beta} \|X\beta-y\|_2^2.
$$

The solve uses an SVD in `float64`, requires full column rank, and casts effects to `float32`.
There is no ridge regularization or minimum-norm fallback for rank-deficient designs.
The centered, unweighted regression is the Banzhaf formulation introduced by
[Liu et al. (2025)](#references): each retained channel contributes `+0.5` and each
removed channel `-0.5` to a design row. Centering the observed scores has no effect on
the fitted coefficients here because the exact and complementary sampled designs have
zero-mean columns.

With exact enumeration, `X.T @ X = (Q / 4) * I`, so the solution equals Banzhaf's uniform
conditional-mean contrasts for **arbitrary games**, not just additive scores, up to numerical
precision. With sampling, complementary masks balance each column but do **not** generally
make columns orthogonal. Least-squares coefficients therefore need not equal Banzhaf's sampled
conditional-mean estimates, even on the same masks. Positive effects increase the score and
negative effects suppress it; zero effects can reflect cancellation, including balanced XOR
or parity interactions.

Liu et al. also study complementary-pair sampling. This implementation applies their
regression to the **active channels of one already-encoded input**, switches to exhaustive
enumeration when it fits the budget, and refuses rank-deficient samples. The paper's
empirical or theoretical accuracy results should not be read as guarantees for every
sampled concept-channel game. Despite the name, [KernelSHAP](kernel_shap.md) estimates
Shapley values using a different regression weighting
([Lundberg and Lee, 2017](#references)).

## Rank requirements

In sampled mode, the `Q = nb_samples` rows form complementary pairs, so the centered design
has rank at most `Q / 2`. If `d > Q / 2`, a precheck raises `ValueError`. Passing this necessary
condition does not guarantee full rank: the realized design is also checked using its singular
values, with tolerance

$$
\tau = \operatorname{eps}_{64}\,\max(Q,d)\,s_{\max}.
$$

Here `eps64` is machine epsilon for `float64` and `smax` is the largest singular value of `X`.
All `d` singular values must exceed this tolerance; otherwise a `ValueError` is raised.
Both rank checks happen **before model or operator calls for the affected input**. Earlier
inputs in the same explanation call may already have been evaluated. There is no resampling.

A larger sampled budget can help but does not guarantee rank. Exact enumeration guarantees
full rank for nonempty support, but requires `2**d` evaluations. To avoid sampled rank failures
when feasible, choose a positive even budget at least `2**d` for every input's active support.

## Example

This coefficient-consuming model has two signed linear effects. A budget of four guarantees
exact enumeration for both inputs, avoiding sampled rank failures.

```python
import numpy as np
import tensorflow as tf

from xplique.attributions import KernelBanzhaf

inputs = tf.keras.Input(shape=(2,))
scores = tf.keras.layers.Dense(
    1, use_bias=False, trainable=False,
    kernel_initializer=tf.keras.initializers.Constant([[2.0], [-3.0]]),
)(inputs)
model = tf.keras.Model(inputs, scores)


def fixed_score(model, masked_coefficients, fixed_targets):
    predictions = model(masked_coefficients, training=False)
    return tf.reduce_sum(predictions * fixed_targets, axis=-1)


coefficients = tf.constant([[1.0, 2.0], [0.0, 1.0]], dtype=tf.float32)
targets = tf.ones((2, 1), dtype=tf.float32)
explainer = KernelBanzhaf(model, operator=fixed_score, nb_samples=4, seed=0)
effects = explainer.explain(coefficients, targets)
np.testing.assert_allclose(effects.numpy(), [[2.0, -6.0], [0.0, -3.0]], atol=1e-6)
```

## References

- Liu, Y., Witter, R. T., Korn, F., et al. (2025). [Kernel Banzhaf: A Fast and Robust
  Estimator for Banzhaf Values](https://arxiv.org/abs/2410.08336). arXiv:2410.08336.
- Lundberg, S. M., and Lee, S.-I. (2017). [A Unified Approach to Interpreting Model
  Predictions](https://arxiv.org/abs/1705.07874). *NeurIPS*.
