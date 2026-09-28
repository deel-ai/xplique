# SparseSobol

SparseSobol measures the total-order sensitivity of a fixed scalar score to already-encoded
concept channels. It restricts the design to channels that are active in each input, applies one
mask value to a whole channel across all positions, and scatters the resulting indices back into
the original channel space.

The indices are unsigned variance sensitivities, including each channel's interactions of every
order. They are **not Banzhaf effects**: their sign does not indicate whether retaining a concept
raises or lowers the score, and they need not behave like conditional-mean contrasts.

This adapts total-order global sensitivity analysis ([Sobol', 2001](#references)) to
already-encoded concept channels; concept importance by such interventions is also central
to [CRAFT (Fel et al., 2023)](#references).

## API

```python
SparseSobol(
    model,
    batch_size=32,
    operator=None,
    nb_design=32,
    mask_distribution="uniform",
    seed=0,
    interaction_kind="pure",
)
```

{{xplique.attributions.SparseSobol}}

## Parameters in-depth

The `model`, `batch_size`, `operator`, and `seed` parameters, accepted inputs, exact active
support, and input-shaped outputs follow the
[shared concept-channel contract](../api_attributions.md#shared-concept-channel-contract).
SparseSobol folds the input seed once more to draw independent $A$ and $B$ matrices. Each
active channel's single total-order index is broadcast over all its positions.

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
total-order estimate ([Jansen, 1999](#references))

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

## Pairwise interactions

`explain_interactions()` preserves the Jansen **total-order** singleton scores in
`main_effects`. `interaction_scores` measures a different quantity, chosen by
`interaction_kind="pure"` (default) or `"total"`. The option is specified in the
constructor and does not affect `explain()`.

For a requested active pair $(i,j)$, add a block $C_{ij}$ formed by replacing
columns $i,j$ of $A$ with those of $B$. With $n=\texttt{nb_design}$, the pure
second-order estimate is

$$
\widehat S_{ij}=\frac{\widehat{\operatorname{Cov}}(f(B),f(C_{ij}))
-\widehat{\operatorname{Cov}}(f(B),f(C_i))
-\widehat{\operatorname{Cov}}(f(B),f(C_j))}
{\max(s_A^2,10^{-12})}.
$$

Covariances and the sample variance $s_A^2$ here use `float64` and denominator
$n-1$. The ordinary singleton Jansen estimator retains its own `float32`
convention. Pure pair scores isolate the pair's ANOVA variance component; Monte
Carlo estimates can be negative or exceed one and are not clipped. Do **not**
subtract the reported singleton total-order indices to calculate a pure pair. This
closed-index construction follows [Saltelli (2002)](#references).

For `interaction_kind="total"`, report all ANOVA variance components containing
both channels, including higher-order interactions. This is the normalized
*superset importance* of [Liu and Owen (2006)](#references):

$$
\widehat T_{ij}=\frac{\frac1n\sum_r[f(A_r)-f(C_{i,r})-f(C_{j,r})
+f(C_{ij,r})]^2}{4\max(s_A^2,10^{-12})}.
$$

Total-pair estimates are nonnegative but can exceed one at finite sample size.
For binary three-channel parity, each pure pair is zero but each total pair is
positive. Both kinds support continuous uniform and binary Bernoulli masks, which
define different intervention games. A binary AND of two channels has population
$S_{12}=T_{12}=1/3$; continuous uniform multiplication has $1/7$.

The additional inference budget is $n$ evaluations per requested **active** pair;
for exactly two active channels, $C_{ij}=B$ and no extra inference is necessary.
`pair_batch_size` bounds the number of hybrid blocks assembled at once, while
`batch_size` bounds each model call. The base design and its outputs remain in memory.

## SparseSobol versus SobolAttributionMethod

[SobolAttributionMethod](sobol.md) is the image-oriented explainer: it creates and resizes spatial
grid masks and returns a spatial attribution map. SparseSobol instead accepts already-encoded,
channel-last coefficients, limits the design to their exact active channels, applies global
channel masks, and returns input-shaped broadcast channel indices. Choose the method according to
the variables in the attribution game, not merely because both use total-order Sobol indices.

The image-oriented Sobol attribution method is described by
[Fel et al. (2021)](#references); neither that image sampling design nor CRAFT's concept
pipeline is identical to the IID, active-support design used here.

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

## References

- Sobol', I. M. (2001). [Global Sensitivity Indices for Nonlinear Mathematical Models and
  Their Monte Carlo Estimates](https://doi.org/10.1016/S0378-4754(00)00270-6).
  *Mathematics and Computers in Simulation*, 55(1–3), 271–280.
- Jansen, M. J. W. (1999). [Analysis of variance designs for model
  output](https://doi.org/10.1016/S0010-4655(98)00154-4). *Computer Physics Communications*,
  117(1-2), 35-43.
- Fel, T., Cadène, R., Chalvidal, M., et al. (2021). [Look at the Variance! Efficient
  Black-box Explanations with Sobol-based Sensitivity Analysis](https://arxiv.org/abs/2111.04138).
  *NeurIPS*.
- Fel, T., Picard, A., Bethune, L., et al. (2023). [CRAFT: Concept Recursive Activation
  FacTorization for Explainability](https://arxiv.org/abs/2211.10154). *CVPR*.
- Saltelli, A. (2002). [Making Best Use of Model Evaluations to Compute Sensitivity
  Indices](https://doi.org/10.1016/S0010-4655(02)00280-1). *Computer Physics
  Communications*, 145(2), 280–297.
- Liu, R., and Owen, A. B. (2006). [Estimating Mean Dimensionality of Analysis of Variance
  Decompositions](https://doi.org/10.1198/016214505000001410). *Journal of the American
  Statistical Association*, 101(474), 712–721.
