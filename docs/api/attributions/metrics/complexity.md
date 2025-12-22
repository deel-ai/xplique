# Complexity

Complexity measures the entropy of attribution maps to evaluate how concentrated or diffuse the explanations are. Lower complexity (lower entropy) indicates more concentrated, sparse explanations, which are often considered more interpretable.

!!! quote
    We measure the complexity of an explanation as the entropy of the fractional contribution of feature attributions.

    -- <cite>[Evaluating and Aggregating Feature-based Model Explanations (2020)](https://arxiv.org/abs/2005.00631)</cite>[^1]

Formally, the complexity is defined as the Shannon entropy of the normalized absolute attributions:

$$ \text{Complexity} = -\sum_{i} p_i \log(p_i) $$

Where $p_i$ is the normalized absolute attribution for feature $i$:

$$ p_i = \frac{|a_i|}{\sum_j |a_j|} $$

!!!info
    The better the explanation, the **lower** the Complexity score. Lower entropy indicates more concentrated, interpretable explanations.

## Score Interpretation

- **Lower scores are better**: A low Complexity score indicates that the explanation is concentrated in a few key features, making it easier to interpret.
- Higher entropy values (approaching $\log(n)$ where $n$ is the number of features) indicate more uniform/diffuse explanations.
- For image explanations with 4D tensors `(B, H, W, C)`, channels are averaged before computing complexity.

## Example

```python
from xplique.metrics import Complexity
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)
explanations = explainer(inputs, labels)

metric = Complexity()
score = metric.evaluate(explanations)
```

!!!note
    Unlike fidelity metrics, Complexity does not require the model or targets—it only evaluates the explanation itself.

{{xplique.metrics.complexity.Complexity}}

[^1]: [Evaluating and Aggregating Feature-based Model Explanations (2020)](https://arxiv.org/abs/2005.00631)

