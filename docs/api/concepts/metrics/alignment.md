# Concept Bank Alignment & Code Consistency

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/metrics/concepts/matching.py)

Concept bank alignment measures how closely two concept dictionaries
$D^{(1)} \in \mathbb{R}^{K_1 \times F}$ and $D^{(2)} \in \mathbb{R}^{K_2 \times F}$
match in a shared feature space—for example, when comparing repeated extraction runs with
different random seeds, different factorizers, or different numbers of concepts.
Matched-code consistency complements bank alignment by checking whether matched concepts
co-vary on the same held-out observations.

- **`ConceptAlignment`**: Optimal one-to-one Hungarian matching between two banks.
- **`MatchedCodeConsistency`**: Signed Pearson correlation of matched concept codes across
  shared held-out observations, reusing a decoder-bank assignment.
- **`HungarianMatchingResult`**: Structured dataclass returned by `detailed_evaluate`.
- **`compare_concept_banks`**: Helper comparing all pairs of banks or all banks against
  a chosen reference bank.

## Definitions

### Bank Alignment

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

### Matched-Code Consistency

Bank alignment compares decoder directions, not which inputs activate them. Given two
coefficient arrays $U^{(1)} \in \mathbb{R}^{N \times K_1}$ and
$U^{(2)} \in \mathbb{R}^{N \times K_2}$ encoded from **the same held-out observations in
the same order** and an existing `HungarianMatchingResult` with matched pairs $(i_m, j_m)$,
`MatchedCodeConsistency` computes the signed Pearson correlation across observations for
each matched pair:

$$
r_m = s_m \cdot \text{corr}\!\left(U^{(1)}_{\cdot, i_m},\, U^{(2)}_{\cdot, j_m}\right) \in [-1, 1]
$$

where the polarity factor $s_m \in \{-1, +1\}$ equals `matched_polarities[m]` only when
`matching.similarity == "absolute_cosine"`, and $s_m = +1$ otherwise. Pairs with a
constant column or fewer than two observations receive `NaN` and are excluded from the
headline mean (returning `NaN` if no pair is scorable).

## Score Interpretation

- **Higher is better:** For cosine modes, `1.0` indicates identical banks up to concept
  permutation (and sign flips for `"absolute_cosine"`). Signed cosine scores may be
  negative; dot scores are unbounded.
- **Unequal bank sizes:** `ConceptAlignment`'s `mean_similarity * coverage` is algebraically
  $\frac{1}{\max(K_1, K_2)} \sum_{m=1}^{M} S_{i_m, j_m}$, treating unmatched concepts as
  contributing `0.0`. This pulls both positive and negative scores toward `0.0` (so
  unequal sizes act as a downward penalty whenever matched similarities are nonnegative).
  Inspect `mean_similarity` and `coverage` separately in `detailed_evaluate` to tell
  whether a score reflects directional match or mismatched bank sizes.
- **Matched-code consistency vs. coverage:** `MatchedCodeConsistency` averages equally over
  scorable pairs and does **not** multiply by bank coverage. Always inspect
  `mean_correlation` alongside `n_scorable_pairs`, `n_matched_pairs`, and `coverage`.
  Pearson correlation measures linear co-variation across observations, not equality in
  offsets or firing supports. For streams, pass aligned `(reference_codes, compared_codes)`
  pairs with a fixed `matching`; never average correlations computed per batch.
- **Per-concept inspection:** Use `assignment_reference_indices`,
  `assignment_compared_indices`, `matched_similarities`, and `matched_polarities` on
  `HungarianMatchingResult`—together with `matched_correlations` from
  `MatchedCodeConsistency.detailed_evaluate`—to identify which specific concepts are
  consistently recovered and used, and which ones drift.

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

## Examples

### Bank alignment across runs

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

### Matched-code consistency on held-out observations

The example below uses fixed nonnegative dictionaries and NNLS encoding to illustrate
evaluating held-out code agreement independently of model training.

```python
import numpy as np
from scipy.optimize import nnls
from xplique.metrics.concepts import ConceptAlignment, MatchedCodeConsistency

reference = np.eye(2)
compared = np.array([[0.1, 1.0], [1.0, 0.1]])
held_out = np.array([[1.0, 0.5], [0.5, 2.0], [2.0, 0.0], [0.0, 1.0]])
reference_codes = np.stack([nnls(reference.T, row)[0] for row in held_out])
compared_codes = np.stack([nnls(compared.T, row)[0] for row in held_out])

matching = ConceptAlignment().detailed_evaluate(reference, compared)
report = MatchedCodeConsistency().detailed_evaluate(
    reference_codes, compared_codes, matching=matching
)
print(matching.coverage_weighted_similarity, report["mean_correlation"])
print(report["n_scorable_pairs"], report["n_matched_pairs"], report["coverage"])
```

## API Reference

{{xplique.metrics.concepts.matching.ConceptAlignment}}

{{xplique.metrics.concepts.matching.MatchedCodeConsistency}}

{{xplique.metrics.concepts.matching.HungarianMatchingResult}}

{{xplique.metrics.concepts.matching.compare_concept_banks}}
