# SparseHSIC

For signed pairwise mixed effects, see [Banzhaf](banzhaf.md#pairwise-interactions)
or [KernelBanzhaf](kernel_banzhaf.md#pairwise-interactions). For pure or
total-pair score-variance sensitivities, see
[SparseSobol](sparse_sobol.md#pairwise-interactions). HSIC pairs measure
output-kernel dependence instead; even additive scalar scores need not have
zero pair components under its RBF output kernel.

SparseHSIC measures the marginal statistical dependence between each active concept channel's
binary retention mask and a fixed scalar model score. It operates on already-encoded,
channel-last coefficients and assigns one unsigned dependence score to each whole channel.
Its separate `explain_interactions()` method computes singleton and pairwise dependence from a
shared mask design and the same masked model evaluations.

The scores are marginal HSIC estimates. They are not signed effects and are not total-order
sensitivity indices: a large score indicates dependence, but does not say whether retaining the
channel raises or lowers the model score and does not isolate all interactions involving that
channel.

HSIC originates with [Gretton et al. (2005)](#references); its use for black-box
attribution was developed by [Novello, Fel, and Vigouroux (2022)](#references).

## API

```python
SparseHSIC(model, batch_size=32, operator=None, nb_samples=1024, seed=0)
```

{{xplique.attributions.SparseHSIC}}

```python
explainer.explain_interactions(inputs, targets=None, *, pairs=None, pair_batch_size=256)
```

The returned `ConceptInteractionResult` objects are described in the
[shared concept-channel contract](../api_attributions.md#shared-concept-channel-contract).

## Parameters in-depth

The `model`, `batch_size`, `operator`, and `seed` parameters, accepted inputs, exact active
support, input-shaped outputs, and the `explain_interactions()` arguments and result format
follow the [shared concept-channel contract](../api_attributions.md#shared-concept-channel-contract).
SparseHSIC masks complete channels against a global all-zero channel baseline; the masks,
scalar outputs, and output Gram matrix for one input are retained regardless of `batch_size`.

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

The IID active-channel masks, binary equality kernel, median-positive-distance bandwidth,
and memory-reduced computation specify this implementation; citing HSIC or image-oriented
HSIC attribution does not imply that their sampling and bandwidth choices are the same.

## Pairwise decomposition

The centered binary kernel of [Novello, Fel, and Vigouroux (2022)](#references) is
$k_0(a,b)=\mathbf{1}[a=b]-1/2$. The paper's joint kernel for channels $i,j$ is
$(1+k_{0,i})(1+k_{0,j})$. Subtracting the singleton components leaves
$K^{\mathrm{int}}_{ij}=K_{0,i}\odot K_{0,j}$, **not** a signed synergy score. With
$s_i=2M_i-1$, $w_{ij}=s_i\odot s_j$, and $q_{ij}=Hw_{ij}$, the efficient estimate is

$$
\widehat I_{ij}=\frac{q_{ij}^{\mathsf T}Lq_{ij}}{4n^2}.
$$

The sign products are centered *after* multiplication; multiplying empirically centered
singleton columns is incorrect for an unbalanced finite design. Both singleton and pairwise
effects use one uncentered output RBF Gram $L$ per input. The normalization $n^{-2}$ matches
this implementation's existing singleton scores; the paper uses $(n-1)^{-2}$. All pair
scores use `float64` algebra and return `float32`, with negative roundoff clamped to zero.
Constant output scores produce exact zero singleton and pair effects.

`pair_batch_size` limits the pair-feature chunk size independently of the model inference
`batch_size`. For $d$ active concepts, $P$ selected active pairs, and
chunk size $B$, computation takes $n$ masked evaluations, $O(n^2d+n^2P)$ kernel arithmetic,
and $O(n^2+nd+nB)$ working memory, plus $O(P)$ indexed result storage. All-pairs output and
runtime remain quadratic in $d$; chunking only limits intermediate pair memory.

## Interpretation and related methods

SparseHSIC is a marginal dependence measure. In a balanced XOR or parity game, any one mask bit
can be independent of the output even though the output depends jointly on every bit. Its
population marginal HSIC is then zero; a finite IID sample can still report nonzero Monte Carlo
noise. SparseHSIC should therefore not be interpreted as a signed contribution or as a measure
that necessarily captures every interaction.

Pairwise dependence can reveal XOR even when its singleton scores vanish, but three-way parity
can have zero singleton **and** pair components. IID finite designs introduce estimation noise
for components that vanish in the population. Do not screen pairs solely by singleton score:
this would discard XOR-like cases. Since the output RBF kernel measures dependence in feature
space, an additive scalar model need not have zero pair components. Pair effects are not signed
synergy, second-order Sobol indices, or game-theoretic interaction values.

[HsicAttributionMethod](hsic.md) is image-oriented: it perturbs spatial patches and returns a
spatial attribution map. SparseHSIC instead perturbs the exact active channels of already-encoded
coefficients and broadcasts global channel scores back to the input shape. The image-oriented
method is described by [Novello et al. (2022)](#references).

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

For XOR, use a fixed scalar score and inspect the indexed pair result:

```python
import tensorflow as tf
from xplique.attributions import SparseHSIC

def xor_score(model, masked, targets):
    del model, targets
    bits = tf.cast(masked > 0, tf.int32)
    return tf.cast(tf.math.floormod(bits[:, 0] + bits[:, 1], 2), tf.float32)

explainer = SparseHSIC(lambda inputs: inputs, operator=xor_score, nb_samples=1024, seed=7)
interaction = explainer.explain_interactions([[1.0, 1.0, 0.0]], [[1.0]])[0]
pair = interaction.pair_indices[tf.argmax(interaction.interaction_scores)].numpy()
assert list(pair) == [0, 1]
```

The third, inactive channel is excluded. In a finite IID run, the singleton effects can be
small rather than identically zero.

## References

- Gretton, A., Bousquet, O., Smola, A., and Schölkopf, B. (2005).
  [Measuring statistical dependence with Hilbert-Schmidt
  norms](https://doi.org/10.1007/11564089_7). *Algorithmic Learning Theory*, 63-77.
- Novello, P., Fel, T., and Vigouroux, D. (2022). [Making Sense of Dependence: Efficient
  Black-box Explanations Using Dependence Measure](https://arxiv.org/abs/2206.06219).
  *NeurIPS*.
