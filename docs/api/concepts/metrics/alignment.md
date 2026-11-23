# Concept Bank Alignment

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/metrics/concepts/matching.py)

Concept bank alignment measures how closely two concept dictionaries
$D^{(1)} \in \mathbb{R}^{K_1 \times F}$ and $D^{(2)} \in \mathbb{R}^{K_2 \times F}$
match in a shared feature space—for example, when comparing repeated extraction runs with
different random seeds, different factorizers, or different numbers of concepts.

- **`ConceptAlignment`**: Optimal one-to-one Hungarian matching between two banks.
- **`HungarianMatchingResult`**: Structured dataclass returned by `detailed_evaluate`.
- **`compare_concept_banks`**: Helper comparing all pairs of banks or all banks against
  a chosen reference bank.

## Definition

Given a pairwise similarity matrix $S \in \mathbb{R}^{K_1 \times K_2}$, `ConceptAlignment`
solves the maximum-sum linear assignment problem via `scipy.optimize.linear_sum_assignment`,
matching $M = \min(K_1, K_2)$ pairs $(i_m, j_m)$:

$$
\text{mean\_similarity} = \frac{1}{M}\sum_{m=1}^{M} S_{i_m, j_m}, \qquad
\text{coverage} = \frac{\min(K_1, K_2)}{\max(K_1, K_2)}, \qquad
\text{score} = \text{mean\_similarity} \times \text{coverage}
$$

Three pairwise similarity modes are supported:

| `similarity` | Definition | When to use |
| --- | --- | --- |
| `"cosine"` *(default)* | $\frac{\langle d_i, d'_j \rangle}{\|d_i\|_2 \|d'_j\|_2} \in [-1, 1]$ | Directional agreement when sign matters (e.g., non-negative NMF concepts). |
| `"absolute_cosine"` | $\left|\frac{\langle d_i, d'_j \rangle}{\|d_i\|_2 \|d'_j\|_2}\right| \in [0, 1]$ | Only when simultaneous concept/coefficient negation is a valid symmetry (e.g., PCA/SVD or signed dictionary learning). |
| `"dot"` | $\langle d_i, d'_j \rangle \in (-\infty, \infty)$ | Scale-sensitive comparison when concept vector norms carry meaning. |

## Score Interpretation

- **Higher is better:** For cosine modes, `1.0` indicates identical banks up to concept
  permutation (and sign flips for `"absolute_cosine"`). Signed cosine scores may be
  negative; dot scores are unbounded.
- **Unequal bank sizes:** `coverage` penalizes unmatched concepts when $K_1 \neq K_2$.
  Inspect `mean_similarity` and `coverage` separately in `detailed_evaluate` to tell
  whether a lower score comes from poor directional match or mismatched bank sizes.
- **Per-concept inspection:** Use `assignment_reference_indices`,
  `assignment_compared_indices`, `matched_similarities`, and `matched_polarities` on
  `HungarianMatchingResult` to identify which specific concepts are consistently
  recovered and which ones drift.

!!! warning "Coordinate systems and stability caveats"
    - **Shared coordinates required:** Both banks must live in the exact same feature
      coordinate system. Having the same feature dimension $F$ does not make banks from
      different models, layers, or preprocessing comparable.
    - **Not a statistical significance test:** Hungarian matching always returns
      $\min(K_1, K_2)$ pairs. Pairwise comparisons from `compare_concept_banks` across
      repeated fits are not independent statistical replicates; treat them as descriptive
      stability diagnostics under an explicit repeated-fit protocol.
    - **Ties, polarity, and equality:** Tied optimal assignments may vary across SciPy
      versions, zero cosine receives polarity `+1` by convention, and
      `HungarianMatchingResult` compares by object identity (compare its array fields with
      NumPy testing helpers).

## Example

```python
import numpy as np
from xplique.metrics.concepts import ConceptAlignment, compare_concept_banks

reference = np.eye(3)
permuted_signed = reference[[2, 0, 1]] * np.array([[-1.0], [1.0], [-1.0]])

alignment = ConceptAlignment(similarity="absolute_cosine")
score = alignment(reference, permuted_signed)
details = alignment.detailed_evaluate(reference, permuted_signed)

print(f"Coverage-weighted similarity: {score:.3f}")
print(f"Mean similarity: {details.mean_similarity:.3f}, Coverage: {details.coverage:.3f}")
print(f"Matched polarities: {details.matched_polarities}")

# Compare multiple runs in one call (all pairs, or pass reference_key='run_a').
pairs = compare_concept_banks(
    {"run_a": reference, "run_b": permuted_signed},
    similarity="absolute_cosine",
)
print(f"Compared pairs: {list(pairs.keys())}")
```

## API Reference

{{xplique.metrics.concepts.matching.ConceptAlignment}}

{{xplique.metrics.concepts.matching.HungarianMatchingResult}}

{{xplique.metrics.concepts.matching.compare_concept_banks}}
