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

The [Banzhaf parameter documentation](banzhaf.md#parameters-in-depth) and
[input/output contract](banzhaf.md#inputs-and-outputs) apply unchanged:

- `model` consumes masked coefficients directly, optionally decoding them before prediction.
  Do not re-encode perturbations.
- Execution is eager, not symbolic or inside `tf.function`. Dense NumPy arrays and TensorFlow
  tensors are sanitized to `float32`, must remain finite, have rank at least two, and have a
  nonempty final concept axis. Sparse and ragged tensors are unsupported.
- For each input, a channel is active if any coefficient is exactly nonzero **after float32
  sanitization**. There is no threshold or top-k screening; signed coefficients are allowed.
  Each active channel is retained or zeroed jointly across all positions, using a zero baseline.
- `operator=None` uses the standard fixed-target classification operator. A custom
  `operator(model, inputs, targets)` must return finite scalar scores of shape `(B,)` or `(B, 1)`.
  The input's target is repeated across perturbations, never reselected from perturbed predictions.
- `nb_samples` is a positive even per-input mask budget, default `1024`. For `d` active channels,
  enumerate all `2**d` masks if they fit in the budget. Otherwise sample `nb_samples // 2`
  independent Bernoulli(0.5) masks and append their complements. The budget counts masks, not
  pairs. Exact masks follow increasing binary integers, with the lowest active channel as the
  least significant bit.
- `batch_size=32` limits perturbations evaluated together; inputs are explained one at a time.
  `None` evaluates an input's entire design in one call and is not memory-bounded. Empty support
  returns zeros without any model or operator calls, including no zero-input evaluation.
- `seed=0` is a signed 64-bit integer folded with the input index for stateless sampling.
  Reproducibility depends on input order and grouping into explanation calls. Exact enumeration
  is seed-independent. Changing perturbation batch size preserves the design; matching effects
  also requires deterministic, batch-independent model and operator inference.
- The result is a dense `float32` TensorFlow tensor with the sanitized input shape, such as
  `(N, K)`, `(N, T, K)`, or `(N, H, W, K)`. Inactive channels have zero effect. Each active
  channel's effect is broadcast over all positions, even positions with zero coefficients.
  These are global channel effects, **not spatial maps**: average position axes, rather than
  summing, to recover `(N, K)` effects.
- Call `explainer.explain(coefficients, targets)` or `explainer(coefficients, targets)`.
  A paired `tf.data.Dataset` is passed as `explainer.explain(dataset, None)` and materialized
  eagerly, not streamed. For a channel-last PyTorch decoder, use `TorchWrapper` with
  `is_channel_first=False`, `requires_grad=False`, and the model in evaluation mode.

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

With exact enumeration, `X.T @ X = (Q / 4) * I`, so the solution equals Banzhaf's uniform
conditional-mean contrasts for **arbitrary games**, not just additive scores, up to numerical
precision. With sampling, complementary masks balance each column but do **not** generally
make columns orthogonal. Least-squares coefficients therefore need not equal Banzhaf's sampled
conditional-mean estimates, even on the same masks. Positive effects increase the score and
negative effects suppress it; zero effects can reflect cancellation, including balanced XOR
or parity interactions.

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
