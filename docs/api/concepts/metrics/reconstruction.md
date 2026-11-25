# Reconstruction Metrics

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/master/xplique/metrics/concepts/reconstruction.py)

Reconstruction metrics measure how accurately a concept representation reproduces the
original layer activations $X \in \mathbb{R}^{N \times F}$ from reconstructed activations
$\hat{X} \in \mathbb{R}^{N \times F}$ (for example, $\hat{X} = U D$).

- **`ReconstructionR2`**: Pooled coefficient of determination ($R^2$).
- **`ReconstructionError`**: Pooled MSE, RMSE, MAE, or relative $L_2$ (Frobenius) error.

## Definitions

Given flattened target activations $X \in \mathbb{R}^{N \times F}$ and reconstructions
$\hat{X} \in \mathbb{R}^{N \times F}$ with sum of squared errors
$\text{SSE} = \sum_{n,f} (X_{n,f} - \hat{X}_{n,f})^2$:

$$
R^2 = 1 - \frac{\text{SSE}}{\text{SST}}
$$

where the total sum of squares $\text{SST}$ depends on `r2_centering`:

- **`r2_centering="feature"`** (default): centers each feature around its observation mean
  $\bar{X}_{\cdot, f}$, giving $\text{SST} = \sum_{n,f} (X_{n,f} - \bar{X}_{\cdot, f})^2$.
- **`r2_centering="global"`**: centers all $N \cdot F$ scalar entries around one grand
  mean $\bar{X}$, giving $\text{SST} = \sum_{n,f} (X_{n,f} - \bar{X})^2$.

For `ReconstructionError`, the four pooled error modes are:

$$
\text{MSE} = \frac{\text{SSE}}{N F}, \qquad
\text{RMSE} = \sqrt{\text{MSE}}, \qquad
\text{MAE} = \frac{1}{N F}\sum_{n,f} |X_{n,f} - \hat{X}_{n,f}|, \qquad
\text{Relative } L_2 = \frac{\|X - \hat{X}\|_F}{\|X\|_F}
$$

## Score Interpretation

- **`ReconstructionR2` (higher is better):** `1.0` means exact reconstruction, `0.0`
  matches predicting the mean activation, and negative values mean worse-than-mean
  reconstruction. At $\text{SST} = 0$ (constant targets, or a single observation with
  feature centering), the default returns `NaN`; set `force_finite=True` to return `1.0`
  for exact reconstruction and `0.0` otherwise.
- **`ReconstructionError` (lower is better):** `0.0` means exact reconstruction.
  `relative_l2` is scale-free (`0.0` for two zero arrays, `inf` when $X = 0$ and
  $\hat{X} \neq 0$).
- **Pooled vs. observation-level:** Global RMSE and global relative $L_2$ are computed
  from pooled squared sums and are **not** the mean of per-observation RMSE or relative
  errors. Never average batch $R^2$ or norm ratios across minibatches; use
  `evaluate_batches` instead.

!!! note "Relationship to scikit-learn `r2_score`"
    When every feature is non-constant, feature-centered `ReconstructionR2` matches
    `sklearn.metrics.r2_score(X, X_hat, multioutput="variance_weighted")`, and global
    centering matches `r2_score(X.ravel(), X_hat.ravel())` whenever $\text{SST} > 0$.
    When constant features are present, Xplique intentionally keeps their residuals in
    the pooled $\text{SSE}$ instead of discarding them.

## Example

```python
import numpy as np
from xplique.metrics.concepts import ReconstructionError, ReconstructionR2

bank = np.array([[1.0, 0.0], [0.6, 0.8]])
coefficients = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
activations = coefficients @ bank
reconstructed = activations + 0.1

# Pooled R2 and its global/feature-level sufficient statistics.
r2_metric = ReconstructionR2()
r2_score = r2_metric(activations, reconstructed)
r2_details = r2_metric.detailed_evaluate(activations, reconstructed)

# Pooled relative L2 error plus per-observation error distributions.
err_metric = ReconstructionError(error="relative_l2", batch_size=2)
rel_l2 = err_metric(activations, reconstructed)
err_details = err_metric.detailed_evaluate(activations, reconstructed)

print(f"R2: {r2_score:.4f} (SSE={r2_details['sse']:.4f}, SST={r2_details['sst']:.4f})")
print(f"Relative L2: {rel_l2:.4f}, RMSE: {err_details['rmse']:.4f}")
print(f"Median observation MSE: {err_details['observation_mse_median']:.4f}")
```

!!! warning "Numerical range"
    Squared sums are accumulated in `float64` without rescaling. Extremely large finite
    inputs (beyond roughly $10^{150}$ in magnitude, depending on element count) can
    overflow and produce `inf` or `NaN`. Rescale inputs first if needed.

## API Reference

{{xplique.metrics.concepts.reconstruction.ReconstructionR2}}

{{xplique.metrics.concepts.reconstruction.ReconstructionError}}
