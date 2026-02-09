# Sparseness

- [**Metrics**: Complexity](https://colab.research.google.com/drive/13boAsXGVKS0LaNzslOdjSkYIrpBJdh7K) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13boAsXGVKS0LaNzslOdjSkYIrpBJdh7K) </sub>

Sparseness measures the concentration of attribution maps using the Gini index. Higher sparseness values indicate that importance is concentrated in fewer features, which is often desirable for interpretability.

!!! quote
    We use the Gini Index to measure the sparseness of explanations. Sparser explanations are often more interpretable as they highlight fewer, more important features.

    -- <cite>[Concise Explanations of Neural Networks using Adversarial Training (2020)](https://proceedings.mlr.press/v119/chalasani20a.html)</cite>[^1]

Formally, the Sparseness is computed as the Gini coefficient of the L1-normalized absolute attributions:

$$ \text{Gini}(x) = \frac{2 \sum_{i=1}^{n} i \cdot x_{(i)}}{n \sum_{i=1}^{n} x_{(i)}} - \frac{n + 1}{n} $$

Where $x_{(i)}$ is the $i$-th smallest component and $n$ is the number of features.

!!!info
    The better the explanation, the **higher** the Sparseness score. Higher values indicate sparser explanations where importance is concentrated in fewer features.

## Score Interpretation

- **Higher scores are better**: A high Sparseness score indicates that the explanation focuses on a small subset of features.
- Values range from 0 (perfectly uniform distribution, low sparseness) to ~1 (maximally concentrated, high sparseness).
- For image explanations with 4D tensors `(B, H, W, C)`, channels are averaged before computing sparseness.

## Example

```python
from xplique.metrics import Sparseness
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)
explanations = explainer(inputs, labels)

metric = Sparseness()
score = metric.evaluate(explanations)
```

!!!note
    Unlike fidelity metrics, Sparseness does not require the model or targets—it only evaluates the explanation itself.

{{xplique.metrics.complexity.Sparseness}}

[^1]: [Concise Explanations of Neural Networks using Adversarial Training (2020)](https://proceedings.mlr.press/v119/chalasani20a.html)

