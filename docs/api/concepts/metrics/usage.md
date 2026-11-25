# Usage & Sparsity Metrics

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/metrics/concepts/usage.py) |
📰 [Hoyer (2004)](https://www.jmlr.org/papers/v5/hoyer04a.html)

Usage and sparsity metrics diagnose how concept coefficients $U \in \mathbb{R}^{N \times K}$
and concept vectors $D \in \mathbb{R}^{K \times F}$ behave across a dataset—identifying
dead concepts, heavily dominant concepts, and how sparsely concepts fire per observation.

- **`ConceptUsage`**: Full multivariate report of per-concept moments, norm-weighted usage,
  component energies, firing rates, and overall sparsity.
- **`ConceptSparsity`**: Scalar inactive fraction with optional observation support sizes
  via `detailed_evaluate`.
- **`HoyerSparsity`**: Mean concentration of component magnitudes per observation,
  including when codes are dense and a support count cannot distinguish them.

## Definitions

Let $d_k \in \mathbb{R}^F$ be row $k$ of `concept_bank` with Euclidean norm $\|d_k\|_2$.
For each observation $n$ and concept $k$, the strict activity indicator $A_{n,k} \in \{0, 1\}$
uses threshold $\tau \ge 0$ (`active_threshold`):

$$
A_{n,k} = \begin{cases}
\mathbb{1}\!\left(|U_{n,k}| \cdot \|d_k\|_2 > \tau\right) & \text{if } \texttt{activity\_basis="contribution"} \text{ (default)} \\
\mathbb{1}\!\left(|U_{n,k}| > \tau\right) & \text{if } \texttt{activity\_basis="coefficient"}
\end{cases}
$$

From $U$, $D$, and $A$, the key reported quantities are:

- **Effective usage (per concept):** $\frac{1}{N}\sum_{n=1}^{N} |U_{n,k}| \|d_k\|_2$ (activation units).
- **Component energy (per concept):** $E_k = \sum_{n=1}^{N} \left(U_{n,k} \|d_k\|_2\right)^2$,
  with share $E_k / \sum_j E_j$ (`NaN` when total energy is zero).
- **Active fraction (per concept):** $\frac{1}{N}\sum_{n=1}^{N} A_{n,k} \in [0, 1]$.
- **Sparsity (global):** $1 - \frac{1}{NK}\sum_{n,k} A_{n,k} \in [0, 1]$.
- **Hoyer sparsity (per observation):** for linear component magnitudes
  $a_{n,k} = |U_{n,k}| \|d_k\|_2$ over all $K$ concept slots (no threshold),
  [Hoyer (2004)](https://www.jmlr.org/papers/v5/hoyer04a.html) defines:

$$
H(a_n) = \frac{\sqrt{K} - \|a_n\|_1 / \|a_n\|_2}{\sqrt{K} - 1} \in [0, 1] \quad (K > 1,\; \|a_n\|_2 > 0)
$$

Zero-contribution observations ($\|a_n\|_2 = 0$) and all observations when $K = 1$ are
undefined (`NaN`) and excluded from the mean (returning `NaN` if no observation is scorable).

## Score Interpretation

- **Contribution basis is scale-invariant:** Factorizations are often invariant to
  rescaling $U_{\cdot,k} \leftarrow \alpha U_{\cdot,k}$ and $d_k \leftarrow \alpha^{-1} d_k$
  for $\alpha \neq 0$. `activity_basis="contribution"` (along with `effective_usage`,
  `component_energy`, and `HoyerSparsity`) is invariant to this rescaling, whereas
  `activity_basis="coefficient"` is not.
- **Threshold units:** `active_threshold` uses strict inequality (`>`) with no implicit
  epsilon—choose it in activation units for `"contribution"` or raw coefficient units for
  `"coefficient"`.
- **Support sparsity vs. Hoyer concentration:** Compare `n_active_concepts` against
  `n_concepts` to spot unused concepts, and inspect `HoyerSparsity` alongside mean active
  count (`observation_support_sizes.mean()`) and reconstruction quality. On dense codes,
  `ConceptSparsity` can report identical support counts for an equally weighted row
  ($H = 0$) and a nearly one-hot row ($H \approx 1$). Higher sparsity is not automatically
  better if reconstruction fidelity degrades.
- **Zero concept vectors:** Allowed in usage and sparsity metrics (giving zero
  contribution), unlike angular geometry and alignment metrics which reject zero rows.

!!! warning "Component energy is not explained variance"
    Because concept vectors $d_k$ are generally non-orthogonal, cross-terms appear in
    $\|U D\|_F^2$. `component_energy_share` therefore measures relative linear-component
    energy, **not** additive explained variance. Norm-weighted quantities also exclude
    any affine decoder bias.

## Examples

### Concept usage and support sparsity

```python
import numpy as np
from xplique.metrics.concepts import ConceptSparsity, ConceptUsage

bank = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
coefficients = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [-1.0, 1.0, 0.0],
])

usage = ConceptUsage(active_threshold=0.0)
report = usage(coefficients, bank)

sparsity = ConceptSparsity(active_threshold=0.0)
sparsity_score = sparsity(coefficients, bank)
sparsity_details = sparsity.detailed_evaluate(coefficients, bank)

print(f"Active concepts: {report['n_active_concepts']}/{report['n_concepts']}")
print(f"Active fraction per concept: {np.round(report['active_fraction'], 3)}")
print(f"Energy share per concept: {np.round(report['component_energy_share'], 3)}")
print(f"Global sparsity: {sparsity_score:.3f}")
print(f"Observation support sizes: {sparsity_details['observation_support_sizes']}")
```

### Hoyer concentration alongside support size and reconstruction

```python
import numpy as np
from xplique.metrics.concepts import ConceptSparsity, HoyerSparsity, ReconstructionR2

bank = np.eye(2)
codes = np.array([[1.0, 1.0], [1.0, 0.001], [0.0, 0.0]])
original = np.array([[1.1, 1.0], [1.0, 0.0], [0.0, 0.0]])
support = ConceptSparsity().detailed_evaluate(codes, bank)["observation_support_sizes"]
print("Mean active count:", support.mean())
print("Per-observation Hoyer:", HoyerSparsity().detailed_evaluate(codes, bank)[
    "observation_hoyer_sparsity"
])
print("Reconstruction R²:", ReconstructionR2()(original, codes @ bank))
```

## API Reference

{{xplique.metrics.concepts.usage.ConceptUsage}}

{{xplique.metrics.concepts.usage.ConceptSparsity}}

{{xplique.metrics.concepts.usage.HoyerSparsity}}
