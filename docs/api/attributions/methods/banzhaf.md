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

The `model`, `batch_size`, `operator`, and `seed` parameters, accepted inputs, exact active
support, and input-shaped outputs follow the
[shared concept-channel contract](../api_attributions.md#shared-concept-channel-contract).
Banzhaf masks are binary: each active channel is retained or zeroed jointly across all its
positions. Signed effects are returned unchanged, without clipping or absolute values.

#### `nb_samples`

A positive, even integer giving the per-input mask budget, defaulting to `1024`. This requirement
also applies when exact enumeration is possible. If there are `d` active channels and
`2**d <= nb_samples`, all `2**d` binary masks are enumerated. Otherwise, sample `nb_samples // 2`
independent masks with independent Bernoulli(0.5) entries and append their complements. This
antithetic design uses exactly `nb_samples` masks and balances each channel's on/off counts.
The budget counts evaluated masks, not complementary pairs. Exact enumeration does not depend
on the seed.

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

## Pairwise interactions

`explain_interactions(inputs, targets, pairs=None, pair_batch_size=256)` also reports the
**signed** Banzhaf interaction of each requested active pair, in score units:

$$
I^B_{ij}=\mathbb E_{M_{-ij}}[v(1,1,M_{-ij})-v(1,0,M_{-ij})
-v(0,1,M_{-ij})+v(0,0,M_{-ij})].
$$

For signs $s_i=2M_i-1$, this equals $4\mathbb E[v(M)s_is_j]$. Exact enumeration
computes this contrast exactly; sampled mode uses four times the unbiased sample
covariance of $s_is_j$ with the **complement-averaged** scores. Each mask/complement
pair is one independent group, so at least four sampled evaluations are needed for
requested active pairs. Constant offsets cancel; a negative score represents a
negative mixed effect (a two-bit 0/1 XOR has interaction $-2$).

The same model evaluations yield singleton and pair scores. Pair computations cost
`O(nb_samples * requested_active_pairs)` in sampled mode; `pair_batch_size` bounds
temporary pair features. Exact zero for an inactive requested pair does not imply
anything about an unrequested pair. Higher-order effects can cancel in the pair
average (three-bit parity has zero pair scores).

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
