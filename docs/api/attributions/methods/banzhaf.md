# Banzhaf

Banzhaf attributes a fixed scalar score to already-encoded concept channels using signed
conditional-mean contrasts under uniform binary masks. Each player is a whole concept channel,
not a pixel or token. Positive effects increase the score on average; negative effects decrease it.
These are not absolute sensitivities, RISE scores, or Shapley values, and need not sum to the
difference between the full-input and zero-input scores.

## API

```python
Banzhaf(model, batch_size=32, operator=None, nb_samples=1024, seed=0)
```

{{xplique.attributions.Banzhaf}}

## Parameters in-depth

#### `model`

A model consuming masked coefficients directly, or a decoder followed by a downstream model.
For a dictionary matrix `D`, a decoder can reconstruct features as `coefficients @ D` before
prediction. Do not re-encode masked coefficients: doing so changes the intervention. Concept
extraction, dictionary fitting, and CRAFT integration are outside this method's scope.

#### `batch_size`

A positive integer limiting the number of perturbations evaluated together, defaulting to `32`.
Inputs are explained one at a time. With `None`, the whole mask design for one input is evaluated
in a single call; this is not memory-bounded. An input with empty active support returns zeros
without any model or operator calls.

#### `operator`

The standard Xplique operator, defaulting to classification when `None`. A custom operator has
signature `operator(model, inputs, targets)` and must return one finite scalar per perturbation,
with shape `(B,)` or `(B, 1)` for a perturbation batch of size `B`. Vector-valued scores and
non-finite scores are invalid. Targets are fixed for each explained input and repeated across
its perturbations; do not select a new target from each perturbed prediction.

#### `nb_samples`

A positive, even integer giving the per-input mask budget, defaulting to `1024`. This requirement
also applies when exact enumeration is possible. If there are `d` active channels and
`2**d <= nb_samples`, all `2**d` binary masks are enumerated. Otherwise, sample `nb_samples // 2`
independent masks with independent Bernoulli(0.5) entries and append their complements. This
antithetic design uses exactly `nb_samples` masks and balances each channel's on/off counts.
The budget counts evaluated masks, not complementary pairs. Exact enumeration does not depend
on the seed.

#### `seed`

The random seed, defaulting to `0`, is folded with the input index using stateless sampling.
Reproducibility therefore depends on input order and grouping into explanation calls: splitting
or reordering inputs can change their sampled designs. For a fixed seed, order, grouping, and
configuration, changing perturbation `batch_size` does not change the design. Matching effects
also requires a deterministic, batch-independent model and operator; stochastic inference or
predictions depending on other batch members do not satisfy this prerequisite.

## Inputs and outputs

Call `explainer.explain(coefficients, targets)` or `explainer(coefficients, targets)` in eager
mode. Symbolic execution and wrapping the explainer in `tf.function` are not supported by this
contract. Dense NumPy arrays and TensorFlow tensors are sanitized to `float32`. Coefficients
must be finite after sanitization, have rank at least two, and have a nonempty final concept axis.
Typical shapes are `(N, K)`, `(N, T, K)`, and `(N, H, W, K)`; arbitrary intervening position
dimensions are allowed. Sparse and ragged tensors are not supported.

For each sanitized input `U`, the exact active support is

$$
A(U) = \{k : \text{at least one entry of } U[\ldots,k] \ne 0\}.
$$

Support is determined **after float32 sanitization**, with no tolerance, threshold, or top-k
screening. Signed coefficients are allowed. Every active channel receives one binary mask value
broadcast across all its positions, so the intervention is `U * mask` with a zero baseline.
Inactive channels remain zero. A value that underflows to zero during conversion is therefore
inactive; a value that overflows to infinity is invalid, even if it was finite before conversion.

The result is a dense `float32` TensorFlow tensor with exactly the sanitized input shape. Each
channel's scalar effect is broadcast across all positions, including positions whose coefficient
is zero; inactive channels have exactly zero effect. These are **input-shaped global channel
effects, not spatial localization maps**. To obtain `(N, K)` effects, average over the position
axes or select one representative position. Do not sum: this would multiply each effect by the
number of positions. For `(N, K)` inputs, no reduction is needed.

A paired `tf.data.Dataset` must contain `(coefficients, targets)` and be passed as
`explainer.explain(dataset, None)`, with no separate targets.
Dataset sanitization materializes the inputs and targets eagerly; it is not streaming ingestion.
For a channel-last PyTorch concept decoder, use `TorchWrapper` with
`is_channel_first=False` and `requires_grad=False`, and put the PyTorch model in evaluation mode.

## Estimator

Let `v(m)` be the fixed-target scalar score of the masked input and let `d = len(A(U))`.
For each active channel `k`, the Banzhaf effect is

$$
\beta_k = \mathbb{E}[v(M) \mid M_k=1]
          - \mathbb{E}[v(M) \mid M_k=0],
\qquad M \sim \operatorname{Bernoulli}(1/2)^d.
$$

The estimator subtracts the mean of scores for masks with channel `k` off from the mean for
masks with that channel on. Enumeration gives the exact uniform-coalition contrast; the
antithetic design estimates it with balanced conditional sample means. Complementary masks
may differ in multiple channels, so they are not individual one-channel marginal differences.

For evaluated masks `m_i`, the same estimator can be written as

$$
\widehat{\beta}_k =
\frac{\sum_{i:m_{ik}=1} v(m_i)}{\#\{i:m_{ik}=1\}}
- \frac{\sum_{i:m_{ik}=0} v(m_i)}{\#\{i:m_{ik}=0\}}.
$$

Both denominators are nonzero for active channels in either design. Empty support is handled
separately by returning an all-zero explanation, without evaluating even the zero-input score.
Exact masks follow increasing binary integers, with the lowest active channel as the least
significant bit. A zero effect can indicate a null concept or cancellation: balanced XOR and
parity games can have zero marginal effects despite depending on each channel.

## Example

This functional Keras model decodes two concept coefficients with a fixed dictionary, then
applies fixed downstream Dense weights. No training, encoder, or CRAFT changes are needed.

```python
import numpy as np
import tensorflow as tf

from xplique.attributions import Banzhaf

dictionary = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
downstream_weights = np.array([[2.0], [-3.0], [0.0]], dtype=np.float32)

inputs = tf.keras.Input(shape=(2,))
features = tf.keras.layers.Dense(
    3, use_bias=False, trainable=False,
    kernel_initializer=tf.keras.initializers.Constant(dictionary),
)(inputs)
scores = tf.keras.layers.Dense(
    1, use_bias=False, trainable=False,
    kernel_initializer=tf.keras.initializers.Constant(downstream_weights),
)(features)
decoder = tf.keras.Model(inputs, scores)

coefficients = tf.constant([[1.0, 2.0], [0.0, 1.0]], dtype=tf.float32)
targets = tf.ones((2, 1), dtype=tf.float32)


def fixed_score(model, masked_coefficients, fixed_targets):
    predictions = model(masked_coefficients, training=False)
    return tf.reduce_sum(predictions * fixed_targets, axis=-1)


explainer = Banzhaf(decoder, operator=fixed_score, nb_samples=4, seed=0)
effects = explainer.explain(coefficients, targets)
np.testing.assert_allclose(effects.numpy(), [[2.0, -6.0], [0.0, -3.0]])

dataset = tf.data.Dataset.from_tensor_slices((coefficients, targets)).batch(2)
dataset_effects = explainer.explain(dataset, None)
np.testing.assert_allclose(dataset_effects.numpy(), effects.numpy())
```

Both inputs use exact enumeration because their active supports have at most two channels.
The negative effect of the second channel is retained rather than clipped or made absolute.
