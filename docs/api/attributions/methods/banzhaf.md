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

This is the uniform-coalition Banzhaf value for the **local coefficient-removal game**
([Banzhaf, 1965](#references); [Dubey and Shapley, 1979](#references)). The players here
are active concept channels, and the value function is the model's fixed-target score after
zeroing omitted channels. The same game-theoretic value can also be written as an average of
one-channel marginal differences across all coalitions of the other active channels.

The estimator subtracts the mean of scores for masks with channel `k` off from the mean for
masks with that channel on. Enumeration gives the exact uniform-coalition contrast; the
antithetic design estimates it with balanced conditional sample means. Complementary masks
may differ in multiple channels, so they are not individual one-channel marginal differences.
Reusing each evaluated coalition score across all channels follows the conditional-mean
sample-reuse estimator discussed by [Wang and Jia (2023)](#references). Appending complements
is this implementation's balanced paired design, related to antithetic sampling studied by
[Staudacher and Pollmann (2023)](#references); that study does not establish an accuracy
guarantee for this particular concept-channel game.

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

## Comparison with KernelBanzhaf

[KernelBanzhaf](kernel_banzhaf.md) inherits the same signature, defaults, and input/output,
masking, scalar-target, and seed contracts, but fits centered full-rank least squares instead
of conditional sample means. Exact enumeration gives the same effects for arbitrary games,
up to numerical precision. Sampled effects generally differ: complementary masks balance
columns without guaranteeing orthogonality. KernelBanzhaf rejects rank-deficient designs
before model or operator calls for the affected input, with no resampling or regularized
fallback; Banzhaf does not require full rank. See its [rank requirements](kernel_banzhaf.md#rank-requirements)
before choosing a sampled budget.

## Related attribution methods

[KernelSHAP](kernel_shap.md) uses a different coalition weighting to estimate Shapley values
([Lundberg and Lee, 2017](#references)); Banzhaf is not constrained to allocate the
full-versus-empty score difference. [RISE](rise.md) uses randomized input masks for image
saliency ([Petsiuk et al., 2018](#references)), whereas this method compares the mean score
*with* a concept channel to the mean score *without* it. Neither comparison implies that
pixel-based estimators operate on the same intervention or baseline as concept removal.

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

## References

- Banzhaf, J. F. (1965). *Weighted Voting Doesn't Work: A Mathematical Analysis*.
  Rutgers Law Review, 19(2), 317–343.
- Dubey, P., and Shapley, L. S. (1979). [Mathematical Properties of the Banzhaf Power
  Index](https://doi.org/10.1287/moor.4.2.99). *Mathematics of Operations Research*, 4(2), 99–131.
- Wang, J. T., and Jia, R. (2023). [Data Banzhaf: A Robust Data Valuation Framework for
  Machine Learning](https://proceedings.mlr.press/v206/wang23e.html). *AISTATS*.
- Staudacher, J., and Pollmann, T. (2023). [Assessing Antithetic Sampling for Approximating
  Shapley, Banzhaf, and Owen Values](https://doi.org/10.3390/appliedmath3040049).
  *AppliedMath*, 3(4), 957–988.
- Lundberg, S. M., and Lee, S.-I. (2017). [A Unified Approach to Interpreting Model
  Predictions](https://arxiv.org/abs/1705.07874). *NeurIPS*.
- Petsiuk, V., Das, A., and Saenko, K. (2018). [RISE: Randomized Input Sampling for
  Explanation of Black-box Models](https://arxiv.org/abs/1806.07421). *BMVC*.
