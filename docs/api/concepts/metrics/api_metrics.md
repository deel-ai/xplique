# Concept-space metrics

These standalone CPU NumPy/SciPy diagnostics evaluate activation reconstructions,
coefficients, and concept banks. Import from `xplique.metrics.concepts`; existing
attribution metric imports and behavior are unchanged. They measure complementary
properties, not a composite quality score or proof of semantic interpretability.

## Common API

Reusable objects store configuration only. Pass changing scientific arrays, including
concept banks, to each call. `metric(...)` delegates to `metric.evaluate(...)`.
Each evaluation is independent, including after an exception. No fitting, decoding,
activation extraction, file I/O, global random-state changes, or persistent caches occur.

Unlike attribution metrics, whose detailed results are generally unreduced per-sample
or perturbation-step scores, concept `detailed_evaluate` returns structured scientific
intermediates. Pooled R2, RMSE and relative error are **not** means of observation scores.

| Entry point | Headline result | Detailed result | Granularity / scalar recovery |
| --- | --- | --- | --- |
| `ReconstructionR2` | Pooled R2 float | Sums, counts, R2 | Global/feature; `r2` |
| `ReconstructionError` | Selected pooled error float | All errors and observation vectors | Global/observation; selected field (`relative_l2` maps to `relative_l2_error`) |
| `ConceptUsage` | Multivariate report dict | No detailed method | Concept/global |
| `ConceptSparsity` | Inactive fraction float | Firing rates and support sizes | Concept/observation; `sparsity` |
| `ConceptAlignment` | Coverage-weighted similarity float | `HungarianMatchingResult` | Pair/bank; `coverage_weighted_similarity` |
| `CoactiveSupportCoherence` | Mean support Babel float | Distribution report and observation arrays | Observation; `mean_support_babel` |
| Babel score / curve functions | Score at an order | Curve report through maximum order | Bank; endpoint `babel_score` |

Detailed array evaluations are opt-in and may store O(N) observation vectors. Scalar
and streamed evaluations do not collect these vectors. `ConceptUsage` already returns
all its meaningful intermediates, so it has no `detailed_evaluate`.

Configuration arguments are keyword-only, except required Babel `order`, which can
be positional or keyword. Scalar outputs are Python floats; vector
outputs are NumPy arrays, not serialized lists. NaN and infinity are intentional
scientific results in documented degenerate cases; serialization is the caller's concern.

### Inputs, observations and memory

Use real, finite integer/floating NumPy arrays. Complex, object, boolean, nonfinite,
and empty inputs are rejected. Computation uses float64; inputs are never mutated.
Activations/reconstructions have identical original shapes `(..., F)` and coefficients
have shape `(..., K)`, with at least two axes. Banks have shape `(K, F)`.
All leading axes flatten into equally weighted observations. A spatial token and an
image vector therefore have different meanings; no image grouping or sample weighting
is inferred. To evaluate images as observations, reshape each image into one feature vector.

Set `batch_size` for large activation sets. The default `None` processes each supplied
array at once. Chunk size changes temporary memory, not the mathematical reduction
(up to float64 roundoff). Contiguous arrays use views; strided arrays gather only the
current observation chunk, without a full flattened copy. Float64 conversion and
finiteness checks happen per chunk. Exact detailed medians/quantiles still require O(N).

Reconstruction and usage/sparsity additionally offer one-pass `evaluate_batches`:

```python
import numpy as np
from xplique.metrics.concepts import ReconstructionR2, ConceptUsage

x = np.arange(24, dtype=float).reshape(6, 4)
y = x + 0.1
r2 = ReconstructionR2(batch_size=2)
score = r2.evaluate_batches((
    (x[:1], y[:1]), (x[1:4], y[1:4]), (x[4:], y[4:]),
))
assert np.isclose(score, r2(x, y))
bank = np.eye(4)
usage = ConceptUsage(batch_size=2)
report = usage.evaluate_batches(iter([x[:1], x[1:]]), concept_bank=bank)
```

Each call creates fresh local sufficient statistics. Caller batches are further sliced
to `batch_size`; feature/concept counts must stay fixed within one evaluation but may
change on the next call. Reconstruction uses stable mergeable central moments with
O(F) state; usage uses O(K). Do not modify arrays while evaluation is in progress.

Convert framework tensors explicitly when necessary: PyTorch
`tensor.detach().cpu().numpy()`, TensorFlow `tensor.numpy()`. This is not a GPU,
autograd, graph-mode, distributed, or sparse-container API. Xplique's existing package
import path still requires TensorFlow; this feature does not make Xplique TensorFlow-free.

## Reconstruction and usage example

```python
from xplique.metrics.concepts import (
    ReconstructionR2, ReconstructionError, ConceptUsage, ConceptSparsity,
)

bank = np.array([[1., 0.], [0.6, 0.8]])
coefficients = np.array([[1., 0.], [0., 2.], [-1., 1.]])
x = coefficients @ bank
reconstructed = x + 0.1

r2 = ReconstructionR2()
assert r2(x, reconstructed) == r2.detailed_evaluate(x, reconstructed)['r2']
error = ReconstructionError(error='relative_l2', batch_size=2)
details = error.detailed_evaluate(x, reconstructed)
assert error(x, reconstructed) == details['relative_l2_error']
assert np.isclose(details['mse'], details['observation_mse'].mean())
assert np.isclose(details['rmse'], np.sqrt(details['observation_mse'].mean()))

usage = ConceptUsage(active_threshold=0.0)
report = usage(coefficients, bank)
sparsity = ConceptSparsity()
assert sparsity(coefficients, bank) == sparsity.detailed_evaluate(
    coefficients, bank
)['sparsity']
```

For targets X and reconstructions X_hat, SSE is `sum((X-X_hat)**2)`.
Feature-centered pooled R2 is `1-SSE/SST`, with SST summing centered squared
deviations from each feature's observation mean. `r2_centering='global'` instead
centers all scalar elements about one mean. Negative scores are permitted. At exactly
zero SST, the default is NaN even for exact reconstruction; `force_finite=True`
returns 1 if SSE is zero, else 0. One observation has zero feature-centered SST.

With every feature nonconstant, feature-centered pooled R2 equals scikit-learn's
`r2_score(X, X_hat, multioutput='variance_weighted')`. Global centering equals
`r2_score(X.ravel(), X_hat.ravel())` when SST is nonzero. Constant features are an
intentional exception: pooled SSE retains their residuals, whereas scikit-learn's
finite variance-weighted aggregation discards them. All-constant targets also differ
from scikit-learn's uniform per-feature fallback. No sklearn call implements the metric.

MSE is SSE divided by `n_elements`; RMSE is its square root; MAE is the mean absolute
residual. Relative L2 is the ratio of pooled Frobenius norms. It is zero for two zero
arrays and infinity for a zero target with a nonzero reconstruction. Detailed outputs
include all four observation-error vectors and observation-MSE mean, population std,
minimum, maximum and exact NumPy median. Neither global RMSE nor global relative L2
is generally the mean of its observation vector. Never average batch R2 or norm ratios.

Usage reports signed coefficient means and population standard deviations, effective
usage `mean(abs(U), axis=0)*norm(D)`, and component energy
`sum(U**2, axis=0)*norm(D)**2`. Energy shares are NaN when total energy is zero.
Nonorthogonal components have cross-terms: energies are not additive explained variance.
Norm-weighted quantities describe linear decoder components, excluding affine offsets.

Activity is strictly `abs(U)*norm(D) > active_threshold` by default; coefficient basis
uses `abs(U) > active_threshold`. The threshold must be explicitly chosen in the
caller's units; zero has no implicit epsilon. Contribution activity, effective usage
and component energy are invariant to reciprocal nonzero bank/coefficient rescaling.
Coefficient activity is not. Usage allows zero concept vectors; angular diagnostics do not.

{{xplique.metrics.concepts.reconstruction.ReconstructionR2}}
{{xplique.metrics.concepts.reconstruction.ReconstructionError}}
{{xplique.metrics.concepts.usage.ConceptUsage}}
{{xplique.metrics.concepts.usage.ConceptSparsity}}

## Matching signed concept banks

```python
from xplique.metrics.concepts import ConceptAlignment, compare_concept_banks

reference = np.eye(3)
compared = reference[[2, 0, 1]] * np.array([[-1.], [1.], [-1.]])
alignment = ConceptAlignment(similarity='absolute_cosine')
details = alignment.detailed_evaluate(reference, compared)
assert alignment(reference, compared) == details.coverage_weighted_similarity
assert np.isclose(details.mean_similarity, 1.)
assert details.n_reference_concepts == details.n_compared_concepts == details.n_matched_pairs == 3
pairs = compare_concept_banks({'run_a': reference, 'run_b': compared},
                             similarity='absolute_cosine')
```

Use signed cosine for directional agreement, absolute cosine only when simultaneous
atom/coefficient sign flips are a valid symmetry, and dot product for scale-sensitive
agreement. Banks must share feature coordinates; equal dimensions alone do not establish
comparability across models or preprocessing. The assignment maximizes total selected
similarity and always returns min(bank sizes) pairs, without establishing significance.
Coverage is matched pairs divided by max(bank sizes); the headline is mean matched
similarity times coverage. It may be negative, and dot scores are unbounded. Zero cosine
has polarity +1 by arbitrary convention. Tied optimal assignments may vary by SciPy version.

The helper returns all pairs or a reference star. Call these stability measurements only
under a defined repeated-fit protocol; pair comparisons are not independent replicates.

{{xplique.metrics.concepts.matching.ConceptAlignment}}
{{xplique.metrics.concepts.matching.HungarianMatchingResult}}
{{xplique.metrics.concepts.matching.compare_concept_banks}}

## Geometry, Babel and observed supports

```python
from xplique.metrics.concepts import (
    concept_bank_geometry_metrics, concept_bank_babel_score,
    concept_bank_babel_curve, CoactiveSupportCoherence,
)

bank = np.array([[1., 0.], [0., 1.], [1., 1.]])
geometry = concept_bank_geometry_metrics(bank)
curve = concept_bank_babel_curve(bank, max_order=2)
assert concept_bank_babel_score(bank, 2) == curve['babel_score']
u = np.array([[1., 0., 1.], [0., 0., 0.], [0., 1., 0.]])
coactive = CoactiveSupportCoherence(batch_size=2)
details = coactive.detailed_evaluate(u, bank)
assert np.isclose(coactive(u, bank), details['mean_support_babel'])
assert np.isclose(details['observation_scores'].mean(), details['mean_support_babel'])
```

Angular summaries normalize nonzero rows and count unordered pairs once. Mutual coherence
is the maximum absolute off-diagonal cosine; mean absolute cosine summarizes all pairs.
Both are zero for one concept. Positive-duplicate/antipodal counts use strict signed
threshold comparisons: these are heuristic near-pair counts, not semantic equivalences.

Spectral geometry uses unit rows by default; `spectrum='raw'` preserves relative atom
scales. Original norms are always reported. For squared-singular-value probabilities p,
energy rank is the smallest cumulative rank reaching `energy_fraction` after excluding
relative singular values at or below `rank_rtol` and renormalizing retained energy.
At `energy_fraction=1.0` it returns the numerical rank exactly; fractions exactly on an
interior cumulative boundary remain subject to float64 roundoff. Stable rank is
sum(sigma squared)/sigma_max squared, and energy entropy rank is exp(-sum(p log p)).
Zero entropy terms contribute zero. Condition number uses the thin SVD, not the K-by-K
Gram matrix: it can be finite for an overcomplete bank. It is infinity when
sigma_min <= rank_rtol*sigma_max; default rank_rtol is max(K,F)*float64 machine epsilon.
The Welch bound is sqrt((K-F)/(F*(K-1))) for K>F, otherwise zero: an ambient-dimensional
theoretical reference, not a measured quality score or rank-adapted bound.[^welch]

For zero-diagonal absolute unit-row Gram G, Babel coherence of integer order p in
[0,K-1] is the maximum sum of the p largest values in a row. It is zero at order zero,
mutual coherence at order one, nondecreasing, and bounded by p, not 1. A truncated
order-zero curve still reports actual mutual coherence. Babel functions perform no SVD.

For observed support S, coactive coherence is max over k in S of sum over l in S,
l != k, of G[k,l]. Empty and singleton supports contribute zero to the unconditional
mean. The conditional multi-support mean is NaN when no support has size at least two.
Detailed reports use exact NumPy linear quantiles and include support sizes/counts;
they are not bank-wide worst-case Babel guarantees. The support definition is exactly
the usage definition, including contribution-based scaling and strict thresholding.

Costs include O(K squared F) Gram work/O(K squared) storage, thin SVD work of order
O(K F min(K,F)), and O(N K squared) matrix work for coactive scoring.
Temporary coactive scoring storage is O(chunk_size K), bounded by `batch_size` when set.
Exact observation medians and coactive quantiles require O(N) extra storage. Scalar
coactive evaluation keeps only running sums/counts, besides the Gram and current chunk.

Babel-based sufficient sparse-recovery bounds concern appropriate exact, noiseless
sparse-representation settings, e.g. the OMP/Basis Pursuit condition
mu_1(p-1)+mu_1(p)<1.[^tropp] They are not semantic-quality certificates; failing the
sufficient bound does not prove recovery failure.

{{xplique.metrics.concepts.geometry.concept_bank_geometry_metrics}}
{{xplique.metrics.concepts.geometry.concept_bank_babel_score}}
{{xplique.metrics.concepts.geometry.concept_bank_babel_curve}}
{{xplique.metrics.concepts.geometry.CoactiveSupportCoherence}}

## Migration from adapter diagnostics

The adapter API is unchanged. To approximate its historical conventions, use
`ReconstructionR2(r2_centering='global', force_finite=True)`, coefficient activity
with `active_threshold=1e-8`, and `spectrum='raw'`. Historical epsilon/zero handling
is intentionally not reproduced bitwise. Spatial observation errors now reduce over
the last feature axis, not all dimensions after the first; reshape explicitly for
image-level grouping.

| Adapter output/lifecycle | Concept API |
| --- | --- |
| update/compute/reset | independent evaluate / evaluate_batches |
| scalar-element n_samples | n_elements, separate n_observations |
| normalized_error | relative_l2_error |
| track_per_sample | opt-in detailed_evaluate, observation terminology |
| avg_activation / std_activation | mean_coefficient / std_coefficient |
| energy / energy_share | component_energy / component_energy_share |
| normalized_score | coverage_weighted_similarity |
| reference_concepts / compared_concepts / matched_pairs | n_reference_concepts / n_compared_concepts / n_matched_pairs |
| effective_rank / entropy_rank | energy_rank / energy_entropy_rank |
| max_abs_cosine | mutual_coherence (no redundant alias) |
| n_channels / n_atoms | n_features / n_concepts |

[^tropp]: J. A. Tropp (2004), *Greed is good: algorithmic results for sparse approximation*, IEEE Transactions on Information Theory 50(10), 2231–2242. <https://doi.org/10.1109/TIT.2004.834793>
[^welch]: L. R. Welch (1974), *Lower bounds on the maximum cross correlation of signals*, IEEE Transactions on Information Theory 20(3), 397–399. <https://doi.org/10.1109/TIT.1974.1055219>
