# Geometry & Coherence Metrics

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/metrics/concepts/geometry.py) |
📰 [Tropp (2004)](https://doi.org/10.1109/TIT.2004.834793) |
📰 [Welch (1974)](https://doi.org/10.1109/TIT.1974.1055219)

Geometry and coherence diagnostics inspect the internal structure of a concept bank
$D \in \mathbb{R}^{K \times F}$—and how its correlated concepts co-activate on real data.

- **`concept_bank_geometry_metrics`**: Angular redundancy, original row norms, spectral
  ranks, conditioning, and the Welch bound.
- **`concept_bank_babel_score`** & **`concept_bank_babel_curve`**: Worst-case cumulative
  coherence $\mu_1(p)$ across groups of $p$ concepts (without SVD).
- **`CoactiveSupportCoherence`**: Observed Babel-style coherence restricted to the
  concepts that actually fire together on each observation.

## Definitions

Let $\tilde{d}_k = d_k / \|d_k\|_2$ be the unit-normalized concept rows (zero rows are
rejected) and let $G \in \mathbb{R}^{K \times K}$ be the zero-diagonal absolute Gram
matrix with $G_{k,\ell} = |\langle \tilde{d}_k, \tilde{d}_\ell \rangle|$ for $k \neq \ell$
and $G_{k,k} = 0$.

### Angular Redundancy & Welch Bound

- **Mutual coherence:** $\mu = \max_{k \neq \ell} G_{k,\ell}$ (`0.0` for $K = 1$).
- **Mean absolute cosine:** average of $G_{k,\ell}$ over all unordered pairs $k < \ell$.
- **Near-duplicate / antipodal counts:** unordered pairs with signed cosine strictly above
  `duplicate_threshold` or below `-duplicate_threshold` (heuristic pair counts, not
  semantic equivalences).
- **Welch bound:**[^welch] ambient-dimensional theoretical lower bound on mutual
  coherence for overcomplete banks ($K > F$), and `0.0` when $K \le F$:

$$
\mu_{\text{Welch}} = \sqrt{\frac{K - F}{F(K - 1)}} \quad (K > F)
$$

### Spectral Geometry

Computed from the thin SVD singular values $\sigma_1 \ge \dots \ge \sigma_r$ of the
unit-normalized bank (`spectrum="unit"`, default) or the scale-preserving raw bank
(`spectrum="raw"`). Original row norms (`norm_min`, `norm_mean`, `norm_max`) are always
reported. With squared-singular-value probabilities $p_i = \sigma_i^2 / \sum_j \sigma_j^2$:

- **Stable rank:** $\sum_i \sigma_i^2 / \sigma_1^2$.
- **Energy entropy rank:** $\exp\!\left(-\sum_i p_i \log p_i\right)$ (with $0 \log 0 = 0$).
- **Energy rank:** smallest cumulative rank reaching `energy_fraction` after excluding
  relative singular values $\sigma_i / \sigma_1 \le \texttt{rank\_rtol}$ and renormalizing
  retained energy. At `energy_fraction=1.0`, this returns the numerical rank exactly.
- **Condition number:** $\sigma_1 / \sigma_r$ on the thin SVD (can be finite for
  overcomplete banks; returns `inf` when $\sigma_r / \sigma_1 \le \texttt{rank\_rtol}$,
  with default $\texttt{rank\_rtol} = \max(K, F)\cdot\varepsilon_{\text{float64}}$).

### Babel Cumulative Coherence & Coactive Support Coherence

For integer order $p \in [0, K - 1]$, the bank-wide **Babel coherence**[^tropp] is the
worst-case cumulative overlap of any concept with $p$ others:

$$
\mu_1(p) = \max_{k} \max_{\substack{\Lambda \subseteq \{1,\dots,K\}\setminus\{k\} \\ |\Lambda| = p}} \sum_{\ell \in \Lambda} G_{k,\ell}
$$

It satisfies $\mu_1(0) = 0$, $\mu_1(1) = \mu$, is nondecreasing, and is bounded by $p$
(not by $1$).

Whereas $\mu_1(p)$ measures the **bank-wide worst case**, **`CoactiveSupportCoherence`**
evaluates overlap only over the concepts that actually fire together on each observation
$n$ (using the exact same strict activity rule as [Usage & Sparsity](usage.md)):

$$
c(S_n) = \begin{cases}
0 & \text{if } |S_n| < 2 \\
\max_{k \in S_n} \sum_{\ell \in S_n \setminus \{k\}} G_{k,\ell} & \text{if } |S_n| \ge 2
\end{cases}
$$

## Score Interpretation

- **Redundancy & rank:** High `mutual_coherence`, rising `positive_duplicate_count`, or an
  `energy_rank` / `stable_rank` far below $\min(K, F)$ suggests redundant or collapsed
  concepts—often a sign that $K$ is larger than the intrinsic concept dimensionality.
- **Bank Babel vs. observed co-activation:** A bank can have moderately high worst-case
  Babel $\mu_1(p)$ yet low `CoactiveSupportCoherence` if correlated concepts rarely fire
  on the same observation. Compare `mean_support_babel` (unconditional, where empty and
  singleton supports score `0.0`) with `mean_multi_support_babel` (conditioned on
  $|S_n| \ge 2$; `NaN` if none exist) in `detailed_evaluate`.
- **Welch bound & recovery bounds:** The Welch bound is an ambient theoretical reference,
  not a quality score. Likewise, Tropp's sufficient sparse-recovery condition
  $\mu_1(p - 1) + \mu_1(p) < 1$ applies to exact noiseless sparse recovery—failing it does
  not prove failure, and passing it does not establish semantic interpretability.[^tropp]

## Example

```python
import numpy as np
from xplique.metrics.concepts import (
    CoactiveSupportCoherence,
    concept_bank_babel_curve,
    concept_bank_babel_score,
    concept_bank_geometry_metrics,
)

bank = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
geometry = concept_bank_geometry_metrics(bank)
curve = concept_bank_babel_curve(bank, max_order=2)
babel_2 = concept_bank_babel_score(bank, 2)

coefficients = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
coactive = CoactiveSupportCoherence(batch_size=2)
details = coactive.detailed_evaluate(coefficients, bank)

print(f"Mutual coherence: {geometry['mutual_coherence']:.3f}, Stable rank: {geometry['stable_rank']:.2f}")
print(f"Babel curve (orders {curve['orders'].tolist()}): {np.round(curve['babel_curve'], 3)} (p=2: {babel_2:.3f})")
print(f"Mean coactive coherence: {details['mean_support_babel']:.3f} (multi-support: {details['mean_multi_support_babel']:.3f})")
```

!!! info "Computational complexity"
    | Operation | Time complexity | Extra memory |
    | --- | --- | --- |
    | Absolute Gram matrix | $O(K^2 F)$ | $O(K^2)$ |
    | Thin SVD (`concept_bank_geometry_metrics`) | $O(K F \min(K, F))$ | $O(K F)$ |
    | `concept_bank_babel_score` / `curve` | $O(K^2 F + K^2 \log K)$ *(no SVD)* | $O(K^2)$ |
    | `CoactiveSupportCoherence` | $O(K^2 F + N K^2)$ | $O(K^2 + \text{batch\_size} \cdot K)$ (+ $O(N)$ for `detailed_evaluate`) |

## API Reference

{{xplique.metrics.concepts.geometry.concept_bank_geometry_metrics}}

{{xplique.metrics.concepts.geometry.concept_bank_babel_score}}

{{xplique.metrics.concepts.geometry.concept_bank_babel_curve}}

{{xplique.metrics.concepts.geometry.CoactiveSupportCoherence}}

[^tropp]: J. A. Tropp (2004), *Greed is good: algorithmic results for sparse approximation*, IEEE Transactions on Information Theory 50(10), 2231–2242. <https://doi.org/10.1109/TIT.2004.834793>
[^welch]: L. R. Welch (1974), *Lower bounds on the maximum cross correlation of signals*, IEEE Transactions on Information Theory 20(3), 397–399. <https://doi.org/10.1109/TIT.1974.1055219>
