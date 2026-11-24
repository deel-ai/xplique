# Usage & Sparsity Metrics

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/metrics/concepts/usage.py)

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

## Score Interpretation

- **Contribution basis is scale-invariant:** Factorizations are often invariant to
  rescaling $U_{\cdot,k} \leftarrow \alpha U_{\cdot,k}$ and $d_k \leftarrow \alpha^{-1} d_k$
  for $\alpha > 0$. `activity_basis="contribution"` (along with `effective_usage` and
  `component_energy`) is invariant to this rescaling, whereas `activity_basis="coefficient"`
  is not.
- **Threshold units:** `active_threshold` uses strict inequality (`>`) with no implicit
  epsilon—choose it in activation units for `"contribution"` or raw coefficient units for
  `"coefficient"`.
- **Dead concepts & sparsity:** Compare `n_active_concepts` against `n_concepts` to spot
  unused concepts. Higher `sparsity` means fewer concepts fire per observation, which is
  often desirable as long as reconstruction fidelity remains acceptable.
- **Zero concept vectors:** Allowed in usage and sparsity metrics (giving zero
  contribution), unlike angular geometry and alignment metrics which reject zero rows.

!!! warning "Component energy is not explained variance"
    Because concept vectors $d_k$ are generally non-orthogonal, cross-terms appear in
    $\|U D\|_F^2$. `component_energy_share` therefore measures relative linear-component
    energy, **not** additive explained variance. Norm-weighted quantities also exclude
    any affine decoder bias.

## Example

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

## API Reference

### Hoyer sparsity

For each observation, [Hoyer sparsity](https://www.jmlr.org/papers/v5/hoyer04a.html)
applies $(\sqrt{K} - \|a\|_1/\|a\|_2)/(\sqrt{K}-1)$ to
$a_k = |u_k| \|d_k\|_2$. It needs no activity threshold and uses all $K$ concept slots.
An equally weighted nonzero row scores zero; a one-component row scores one.
Zero-contribution rows and rows with $K=1$ are undefined and excluded from the mean.
Report mean active count and reconstruction quality alongside concentration.

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

{{xplique.metrics.concepts.usage.ConceptUsage}}

{{xplique.metrics.concepts.usage.ConceptSparsity}}

{{xplique.metrics.concepts.usage.HoyerSparsity}}
