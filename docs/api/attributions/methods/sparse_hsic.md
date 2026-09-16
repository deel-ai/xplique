# SparseHSIC

SparseHSIC measures the marginal statistical dependence between each active concept channel's
binary retention mask and a fixed scalar model score. It operates on already-encoded,
channel-last coefficients and assigns one unsigned dependence score to each whole channel.

The scores are marginal HSIC estimates. They are not signed effects and are not total-order
sensitivity indices: a large score indicates dependence, but does not say whether retaining the
channel raises or lowers the model score and does not isolate all interactions involving that
channel.

## API

```python
SparseHSIC(model, batch_size=32, operator=None, nb_samples=1024, seed=0)
```

{{xplique.attributions.SparseHSIC}}

## Parameters in-depth

#### `model`

A model consuming masked coefficients directly, or a decoder followed by a downstream model.
Inputs must already be encoded: re-encoding a perturbed input would define a different
intervention. SparseHSIC masks complete channels against a global all-zero channel baseline.

#### `batch_size`

A positive integer limiting the number of masked versions evaluated together, defaulting to
`32`. Inputs are explained one at a time. Batching changes only how the model and operator are
called; it does not change the sampled masks or estimator. The masks, scalar outputs, and output
Gram matrix for one input are still retained. With `None`, all perturbations for one input are
evaluated in one call and model evaluation is not memory-bounded.

#### `operator`

The standard Xplique operator, defaulting to classification when `None`. A custom operator has
signature `operator(model, inputs, targets)` and must return one finite scalar per perturbation,
with shape `(B,)` or `(B, 1)` for a perturbation batch of size `B`. The target belonging to the
input being explained is repeated and remains fixed across all perturbations; it must not be
reselected from each perturbed prediction.

#### `nb_samples`

The number $n$ of masks evaluated for each nonempty input, defaulting to `1024`. It must be an
integer at least `2`. For $d$ active channels, SparseHSIC draws an $n \times d$ matrix whose
entries are independent $\operatorname{Bernoulli}(1/2)$ variables and performs exactly $n$ model
evaluations.

The design is always IID sampling. SparseHSIC does not enumerate small games, append antithetic
complements, force balanced columns, or resample degenerate columns. Consequently, an active
channel can happen to be always retained or always removed in a finite design; its centered mask
is then constant and its estimate is zero. An input with no active channels returns an all-zero
explanation without evaluating the model or operator.

#### `seed`

A signed 64-bit integer seed, defaulting to `0`. Stateless random generation folds the original
input index into the seed. Sampling is independent of TensorFlow's global random state and of
perturbation `batch_size`, but reproducibility depends on input order and grouping into explanation
calls. Skipping an empty-support input does not renumber later inputs. Exact random values are only
guaranteed in a matching software and hardware environment; matching explanations also require
deterministic, batch-independent model and operator inference.

## Inputs and outputs

Call `explainer.explain(coefficients, targets)` or `explainer(coefficients, targets)` in eager
mode. Symbolic execution and wrapping the explainer in `tf.function` are not supported. Dense
NumPy arrays and TensorFlow tensors are sanitized to `float32`; paired `tf.data.Dataset` inputs are
materialized eagerly. Coefficients must be finite after sanitization, have rank at least two, and
have a nonempty final concept axis. Typical channel-last shapes are `(N, K)`, `(N, T, K)`, and
`(N, H, W, K)`.

For each sanitized input $U$, the exact active support is

$$
A(U) = \{k : \text{at least one entry of } U[\ldots,k] \ne 0\}.
$$

Support is determined **after float32 sanitization**, without a tolerance, threshold, or top-k
selection. Each binary mask value is broadcast over every position of its active channel. The
perturbation is therefore global channel retention or removal against zero, not spatial masking;
inactive channels remain exactly zero.

The result is a dense `float32` TensorFlow tensor with exactly the sanitized input shape. One score
per active channel is broadcast over all its positions, including positions where that channel's
coefficient is zero. Inactive channels have exactly zero attribution. The output contains global
channel dependence scores, not a spatial localization map. Average over position axes, rather
than summing, to recover an `(N, K)` matrix.

## Kernels and bandwidth

For mask rows $m_i$ and $m_j$, channel $k$ uses the equality kernel

$$
K^{(k)}_{ij} = \mathbf{1}[m_{ik}=m_{jk}].
$$

For scalar scores $y_i$, the output kernel is the RBF kernel

$$
L_{ij} = \exp\left(-\frac{(y_i-y_j)^2}{2\sigma^2}\right).
$$

The bandwidth $\sigma$ is the median of the strictly positive pairwise absolute score distances
$|y_i-y_j|$ for $i<j$. Repeated distances retain their pair multiplicity; the median is not taken
over a set of unique values. Zero distances are excluded. If no strictly positive distance exists,
the bandwidth falls back to `1`. A constant output is detected explicitly and produces exact zero
scores for every channel.

## Biased HSIC estimator

Let $H=I_n-\mathbf{1}\mathbf{1}^{\mathsf T}/n$, and define the centered Gram matrices
$K_c^{(k)}=HK^{(k)}H$ and $L_c=HLH$. SparseHSIC uses the biased empirical estimator

$$
\widehat{\operatorname{HSIC}}_k
= \frac{\operatorname{tr}(K_c^{(k)}L_c)}{n^2}.
$$

The implementation does not construct a `(d, n, n)` stack of input Gram matrices. If $M$ is the
$n \times d$ binary mask matrix and $Z=HM$ contains its centered columns, centering the equality
kernel gives $K_c^{(k)}=2z_kz_k^{\mathsf T}$. Since $HZ=Z$, all channel estimates are obtained as

$$
\frac{2\,\operatorname{diag}(Z^{\mathsf T}LZ)}{n^2}.
$$

This keeps memory at $O(n^2+nd)$ instead of $O(dn^2)$, while forming and applying the output Gram
matrix remains quadratic in `nb_samples`. Kernels and estimator algebra use `float64`; the final
input-shaped explanation is cast to `float32`. Tiny negative values caused by roundoff are clamped
to zero. If the median distance exceeds the finite `float64` range, distances are computed after a
common positive rescaling; this leaves the RBF distance-to-bandwidth ratios unchanged. Scores are
otherwise not normalized and have no upper clipping.

## Interpretation and related methods

SparseHSIC is a marginal dependence measure. In a balanced XOR or parity game, any one mask bit
can be independent of the output even though the output depends jointly on every bit. Its
population marginal HSIC is then zero; a finite IID sample can still report nonzero Monte Carlo
noise. SparseHSIC should therefore not be interpreted as a signed contribution or as a measure
that necessarily captures every interaction.

[HsicAttributionMethod](hsic.md) is image-oriented: it perturbs spatial patches and returns a
spatial attribution map. SparseHSIC instead perturbs the exact active channels of already-encoded
coefficients and broadcasts global channel scores back to the input shape.

[SparseSobol](sparse_sobol.md) reports unsigned Jansen total-order variance sensitivity, including
interactions involving a channel, using a replicated design and more than $n$ evaluations when
channels are active. [Banzhaf](banzhaf.md) reports signed conditional-mean effects under binary
retention masks. These estimands are different even when all three methods use a zero baseline.

## Example

```python
import tensorflow as tf

from xplique.attributions import SparseHSIC


def fixed_score(model, masked_coefficients, fixed_targets):
    predictions = model(masked_coefficients, training=False)
    return tf.reduce_sum(predictions * fixed_targets, axis=-1)


model = tf.keras.Sequential([tf.keras.layers.Dense(1, use_bias=False)])
coefficients = tf.constant([[1.0, -2.0, 0.0]], dtype=tf.float32)
targets = tf.ones((1, 1), dtype=tf.float32)

explainer = SparseHSIC(model, operator=fixed_score, nb_samples=256, seed=7)
scores = explainer(coefficients, targets)  # shape (1, 3); scores[0, 2] is exactly zero
```

## Reference

Gretton, A., Bousquet, O., Smola, A., and Scholkopf, B. (2005).
[Measuring statistical dependence with Hilbert-Schmidt
norms](https://doi.org/10.1007/11564089_7). *Algorithmic Learning Theory*, 63-77.
