# API: Concept-Space Metrics

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/tree/master/xplique/metrics/concepts) |
📰 [Hoyer (2004)](https://www.jmlr.org/papers/v5/hoyer04a.html) |
📰 [Tropp (2004)](https://doi.org/10.1109/TIT.2004.834793) |
📰 [Welch (1974)](https://doi.org/10.1109/TIT.1974.1055219)

## Context

Automatic concept extraction methods such as [CRAFT](../craft.md) and
[Holistic CRAFT](../holistic_craft.md) decompose intermediate neural activations $X$ into
concept coefficients $U$ and a concept bank $D$ such that $X \approx U D$.
Good reconstruction alone does not tell you whether the learned concepts are distinct,
sparsely used, or consistently recovered across runs.

The `xplique.metrics.concepts` module provides standalone CPU NumPy/SciPy diagnostics
answering five complementary questions:

1. **Reconstruction:** How faithfully do the concepts reconstruct the original activations?
2. **Usage & sparsity:** Which concepts actually fire, how much energy do they carry, and how sparse are their activations?
3. **Bank alignment and code consistency:** Do two runs learn similar directions and use them on the same observations?
4. **Bank geometry:** Is the concept bank redundant, near-duplicate, or rank-deficient?
5. **Coactive coherence:** Do correlated concepts actually activate *together* on real observations?

To check whether a factorizer preserves model outputs, see
[Prediction Fidelity](prediction_fidelity.md). Those scorers compare decoded task
predictions rather than operating on the activations or concept bank.

!!! info
    These metrics evaluate complementary geometric and statistical properties of concept
    representations. They do **not** define a single composite quality score or prove human
    semantic interpretability, and they leave existing
    [attribution metrics](../../attributions/metrics/api_metrics.md) unchanged.

## Available Metrics

| Family | Entry points | Headline output | Detailed output | Documentation |
| --- | --- | --- | --- | --- |
| **Reconstruction** | `ReconstructionR2`, `ReconstructionError` | Pooled $R^2$ or selected error (`float`) | Global/feature sums or per-observation error vectors | [Reconstruction](reconstruction.md) |
| **Usage & Sparsity** | `ConceptUsage`, `ConceptSparsity`, `HoyerSparsity` | Usage report, inactive fraction, or mean Hoyer sparsity | Firing rates, support sizes, or per-observation Hoyer scores | [Usage & Sparsity](usage.md) |
| **Bank Alignment & Code Consistency** | `ConceptAlignment`, `compare_concept_banks`, `MatchedCodeConsistency` | Coverage-weighted similarity or mean matched code correlation | Assignment, per-pair similarities, polarities, and correlations | [Alignment](alignment.md) |
| **Geometry & Coherence** | `concept_bank_geometry_metrics`, `concept_bank_babel_score`, `concept_bank_babel_curve`, `CoactiveSupportCoherence` | Geometry dict, Babel score/curve, or mean support coherence (`float`) | Observation support scores, quantiles & support counts | [Geometry & Coherence](geometry_coherence.md) |
| **Prediction Fidelity** | `ClassificationPredictionFidelity`, `DetectionPredictionFidelity`, `PosePredictionFidelity`, `SemanticSegmentationPredictionFidelity` | Multivariate agreement report (`dict`) with explicit denominators | Complete report returned directly by `evaluate` | [Prediction Fidelity](prediction_fidelity.md) |

## Quick Start with a Concept Factorizer

Because all metrics operate directly on NumPy arrays, they plug straight into any
`ConceptFactorizer`, CRAFT / Holistic CRAFT run, or external dictionary-learning method:

```python
import numpy as np
from xplique.concepts.tf.factorizer import TfSklearnNMFFactorizer
from xplique.metrics.concepts import (
    CoactiveSupportCoherence,
    ConceptSparsity,
    ConceptUsage,
    ReconstructionError,
    ReconstructionR2,
    concept_bank_geometry_metrics,
)

# Fit a concept factorizer on non-negative activations of shape (N, F).
rng = np.random.default_rng(0)
activations = rng.uniform(0.0, 1.0, size=(64, 4)) @ rng.uniform(0.0, 1.0, size=(4, 16))
activations = activations.astype(np.float32)

factorizer = TfSklearnNMFFactorizer(n_components=4, max_iter=500, random_state=0)
concept_bank, coefficients = factorizer.fit(activations)
reconstructed = factorizer.decode(coefficients)

# Evaluate reconstruction, usage, geometry, and observed co-activation.
r2 = ReconstructionR2()(activations, reconstructed)
rel_l2 = ReconstructionError(error="relative_l2")(activations, reconstructed)
usage = ConceptUsage()(coefficients, concept_bank)
sparsity = ConceptSparsity()(coefficients, concept_bank)
geometry = concept_bank_geometry_metrics(concept_bank)
coactive = CoactiveSupportCoherence()(coefficients, concept_bank)

print(f"R2: {r2:.3f}, Relative L2: {rel_l2:.3f}, Sparsity: {sparsity:.3f}")
print(f"Active concepts: {usage['n_active_concepts']}/{usage['n_concepts']}")
print(f"Mutual coherence: {geometry['mutual_coherence']:.3f}, Coactive coherence: {coactive:.3f}")
```

!!! tip "Using with TensorFlow or PyTorch tensors"
    Convert framework tensors to NumPy explicitly before evaluation:
    `tensor.numpy()` in TensorFlow, or `tensor.detach().cpu().numpy()` in PyTorch.
    For `HolisticCraft`, you can evaluate `coeffs_u = craft.transform(images)` and
    `concept_bank = craft.factorization.concept_bank_w` directly.

## Common API

All class-based metrics store **configuration only** (keyword-only arguments) and accept
data on each call:

- `metric.evaluate(...)` — computes the headline result (and `metric(...)` is an alias).
- `metric.detailed_evaluate(...)` — returns structured scientific intermediates from which
  the headline scalar can be recovered. `ConceptUsage` and the prediction fidelity scorers
  already return their complete multivariate reports in `evaluate(...)`, so they have no
  separate `detailed_evaluate`.
- `metric.evaluate_batches(...)` — available on reconstruction, usage/sparsity,
  matched-code consistency, coactive coherence, and prediction fidelity metrics for
  single-pass streaming.

Each call is completely independent (including after an exception): inputs are never
mutated, no global random state is touched, and no data is cached between calls.

!!! note "Difference from attribution metrics"
    Unlike attribution metrics, whose `detailed_evaluate` generally returns unreduced
    per-sample or perturbation-step scores, concept `detailed_evaluate` returns structured
    intermediates at observation, concept, or bank granularity. Pooled $R^2$, RMSE, and
    relative $L_2$ error are **not** averages of per-observation or per-batch scores.

### Inputs, Observation Flattening, and Types

- Pass real, finite integer or floating NumPy arrays (complex, boolean, object, empty,
  and `NaN`/`inf` inputs raise `ValueError`). Computations run in `float64`.
- Activations and reconstructions must have identical shape `(..., F)` with at least two
  axes; coefficients have shape `(..., K)` with at least two axes; concept banks have
  shape `(K, F)`.
- **Leading axes are flattened into equally weighted observations.** For example, a
  spatial activation map `(N, H, W, F)` is treated as $N \cdot H \cdot W$ token
  observations, not $N$ images. To evaluate whole images as observations, reshape
  explicitly to `(N, H * W * F)` first.
- Scalar outputs are Python `float`s and array outputs are NumPy arrays. Documented
  degenerate cases intentionally return `NaN` or `inf` (for instance, $R^2$ on constant
  targets or relative $L_2$ with a zero target).

### Memory Control and Streaming (`batch_size` and `evaluate_batches`)

For large activation sets, set `batch_size` to chunk observations without changing the
mathematical result (up to `float64` roundoff). Contiguous arrays are sliced as views;
strided arrays gather only the current chunk rather than copying the full array.

`ReconstructionR2`, `ReconstructionError`, `ConceptUsage`, `ConceptSparsity`,
`HoyerSparsity`, `MatchedCodeConsistency`, `CoactiveSupportCoherence`, and all four
prediction fidelity scorers (`ClassificationPredictionFidelity`,
`DetectionPredictionFidelity`, `PosePredictionFidelity`,
`SemanticSegmentationPredictionFidelity`) provide single-pass `evaluate_batches(...)`
using evaluation-local running statistics:

```python
import numpy as np
from xplique.metrics.concepts import ConceptUsage, ReconstructionR2

x = np.arange(24, dtype=float).reshape(6, 4)
y = x + 0.1

# Reconstruction expects an iterable of (activations, reconstructed) pairs.
r2 = ReconstructionR2(batch_size=2)
streamed_r2 = r2.evaluate_batches([
    (x[:2], y[:2]),
    (x[2:4], y[2:4]),
    (x[4:], y[4:]),
])

# Usage/sparsity consume an iterable of coefficient batches with a fixed bank.
bank = np.eye(4)
usage = ConceptUsage(batch_size=2)
report = usage.evaluate_batches(iter([x[:2], x[2:]]), concept_bank=bank)

print(f"Streamed R2: {streamed_r2:.6f}, Sparsity: {report['sparsity']:.3f}")
```

!!! warning "Streaming input shape and memory caveats"
    - For reconstruction `evaluate_batches`, pass an iterable of pairs such as
      `[(activations, reconstructed), ...]`, **not** a bare `(activations, reconstructed)`
      tuple.
    - Opt-in `detailed_evaluate` methods that compute per-observation vectors, exact
      medians, or quantiles still require $O(N)$ storage.
    - Do not mutate arrays while an evaluation is in progress.
