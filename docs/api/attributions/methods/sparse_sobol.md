# SparseSobol

SparseSobol measures the total-order sensitivity of a fixed scalar score to already-encoded
concept channels. It restricts the design to channels that are active in each input, applies one
mask value to a whole channel across all positions, and scatters the resulting indices back into
the original channel space.

The indices are unsigned variance sensitivities, including each channel's interactions of every
order. They are **not Banzhaf effects**: their sign does not indicate whether retaining a concept
raises or lowers the score, and they need not behave like conditional-mean contrasts.

## API

```python
SparseSobol(
    model,
    batch_size=32,
    operator=None,
    nb_design=32,
    mask_distribution="uniform",
    seed=0,
)
```

{{xplique.attributions.SparseSobol}}

## Parameters in-depth

#### `model`

A model consuming masked coefficients directly, or a decoder followed by a downstream model.
Inputs must already be encoded; do not re-encode a masked input because that changes the
intervention. Masking always uses a global channel-wise zero baseline.

#### `batch_size`

A positive integer limiting the number of perturbed versions sent to the model together,
defaulting to `32`. It batches model evaluations only: inputs are explained one at a time and the
complete mask design and scalar outputs are retained. With `None`, the full design for one input
is evaluated in one call and is not memory-bounded. Changing this value does not change the design.

#### `operator`

The standard Xplique operator, defaulting to classification when `None`. A custom operator has
signature `operator(model, inputs, targets)` and must return one finite scalar per perturbation,
with shape `(B,)` or `(B, 1)` for a perturbation batch of size `B`. The target of the input being
explained is repeated and remains fixed across all perturbations; it is not selected again from
each perturbed prediction.

#### `nb_design`

The number $n$ of rows in each base Monte Carlo matrix, defaulting to `32`. It must be an integer
at least `2`, but does **not** need to be a power of two. It is not the number of model evaluations.
For an input with $d$ active channels, SparseSobol evaluates exactly

$$
n(d+2)
$$

perturbations: $n$ rows from each of $A$ and $B$, followed by $n$ rows for each of the $d$
replicated matrices $C_i$. Empty-support inputs return zeros without model or operator calls.

#### `mask_distribution`

The intervention distribution, either `"uniform"` (the default) or `"bernoulli"`:

- `"uniform"` draws independent values in $[0,1)$, continuously attenuating each active channel.
- `"bernoulli"` draws independent binary values with equal probability, retaining or removing
  each active channel.

These are different sensitivity games, not interchangeable sampling optimizations. Uniform masks
measure sensitivity to graded attenuation; Bernoulli masks measure sensitivity to binary retention
against the zero baseline.

#### `seed`

A signed 64-bit integer seed, defaulting to `0`. SparseSobol uses stateless random generation for
independent, identically distributed $A$ and $B$ matrices and folds the original input index into
the seed. Sampling is independent of TensorFlow's global random state and of perturbation
`batch_size`, but reproducibility depends on input order and grouping into explanation calls.
Skipping an empty-support input does not renumber later inputs. Exact random values are only
guaranteed within a matching software and hardware environment; the seed is not a promise of
bitwise portability across TensorFlow versions, devices, or platforms. Matching explanations also
requires deterministic, batch-independent model and operator inference.

## Inputs and outputs

Call `explainer.explain(coefficients, targets)` or `explainer(coefficients, targets)` in eager
mode. Symbolic execution and wrapping the explainer in `tf.function` are not supported. Dense
NumPy arrays and TensorFlow tensors are sanitized to `float32`; paired `tf.data.Dataset` inputs are
materialized eagerly. Coefficients must be finite after sanitization, have rank at least two, and
have a nonempty final concept axis. Typical channel-last shapes are `(N, K)`, `(N, T, K)`, and
`(N, H, W, K)`.

For each sanitized input $U$, SparseSobol uses the exact active support

$$
A(U) = \{k : \text{at least one entry of } U[\ldots,k] \ne 0\}.
$$

Support is determined **after float32 sanitization**, without a tolerance, threshold, or top-k
selection. Every mask value is broadcast over all positions of its active channel. Thus the
perturbation is coefficient attenuation or removal relative to the all-zero channel baseline,
not spatial masking. Inactive channels remain exactly zero.

The result is a dense `float32` TensorFlow tensor with exactly the sanitized input shape. Each
active channel's single total-order index is broadcast over all its positions, including positions
where that channel's coefficient is zero; inactive channels have exactly zero attribution. The
output therefore contains global channel sensitivities, **not a spatial localization map**. Average
over position axes, rather than summing, to recover an `(N, K)` matrix.

## Replicated design

For the $d$ active channels, SparseSobol draws independent IID Monte Carlo matrices
$A,B \in \mathbb{R}^{n \times d}$ from the selected mask distribution. It then constructs $C_i$
as a copy of $A$ whose column $i$ is replaced by column $i$ of $B$. Masks and model outputs use
this exact order:

```text
A, B, C_0, C_1, ..., C_(d-1)
```

Each block contains `nb_design` rows, and active channels follow ascending indices in the ambient
channel axis. This is an IID Monte Carlo design, not the quasi-Monte Carlo Sobol sequence used by
some other sensitivity workflows.

## Jansen estimator

Let $f(A_j)$ and $f(C_{i,j})$ be the fixed-target scores for row $j$. SparseSobol reports Jansen's
total-order estimate

$$
\widehat{S}_{T_i} =
\frac{\sum_{j=1}^{n}\left(f(A_j)-f(C_{i,j})\right)^2}
     {2n\,\max\left(s_A^2,10^{-12}\right)},
$$

where $s_A^2$ is the unbiased `float32` sample variance of the $A$ outputs, using denominator
$n-1$. Calculations and returned indices use `float32`. The $B$ outputs are evaluated to preserve
the complete replicated-design contract even though this Jansen total-order formula does not
currently consume them.

The squared differences make the result unsigned. Constant scores produce exact zero indices.
There is no clipping to $[0,1]$: a finite Monte Carlo estimate may exceed `1`. The `1e-12`
variance floor protects constant and near-constant reference outputs and is part of the
estimator's numerical convention.

## SparseSobol versus SobolAttributionMethod

[SobolAttributionMethod](sobol.md) is the image-oriented explainer: it creates and resizes spatial
grid masks and returns a spatial attribution map. SparseSobol instead accepts already-encoded,
channel-last coefficients, limits the design to their exact active channels, applies global
channel masks, and returns input-shaped broadcast channel indices. Choose the method according to
the variables in the attribution game, not merely because both use total-order Sobol indices.

## Example

```python
import tensorflow as tf

from xplique.attributions import SparseSobol


def fixed_score(model, masked_coefficients, fixed_targets):
    predictions = model(masked_coefficients, training=False)
    return tf.reduce_sum(predictions * fixed_targets, axis=-1)


inputs = tf.keras.Input(shape=(3,))
scores = tf.keras.layers.Dense(1, use_bias=False)(inputs)
model = tf.keras.Model(inputs, scores)

coefficients = tf.constant([[1.0, -2.0, 0.0]], dtype=tf.float32)
targets = tf.ones((1, 1), dtype=tf.float32)
explainer = SparseSobol(
    model,
    operator=fixed_score,
    nb_design=64,
    mask_distribution="bernoulli",
    seed=7,
)
indices = explainer(coefficients, targets)  # shape (1, 3); indices[0, 2] is exactly zero
```

## Reference

Jansen, M. J. W. (1999). [Analysis of variance designs for model
output](https://doi.org/10.1016/S0010-4655(98)00154-4). *Computer Physics Communications*,
117(1-2), 35-43.
