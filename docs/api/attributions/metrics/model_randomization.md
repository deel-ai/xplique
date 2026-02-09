# Model Randomization Metric

- [**Metrics**: Randomization](https://colab.research.google.com/drive/13lNkZqKajRJ63XllkQddgrPOYF1Xv9-Y) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13lNkZqKajRJ63XllkQddgrPOYF1Xv9-Y) </sub>

Model Randomization metric tests whether explanations degrade when model parameters are randomized. This implements a sanity check to verify that explainers are sensitive to model parameters.

!!! quote
    We propose sanity checks for saliency methods. [...] We find that some widely deployed saliency methods are independent of both the data the model was trained on, and the model parameters.

    -- <cite>[Sanity Checks for Saliency Maps (2018)](https://arxiv.org/abs/1810.03292)</cite>[^1]

For each sample $(x, y)$:

1. Compute explanation under the original model
2. Randomize model parameters according to a strategy
3. Compute explanation under the randomized model
4. Measure Spearman rank correlation between both explanations

!!!info
    A **low Spearman correlation** indicates that explanations are sensitive to the model parameters (desirable for a faithful explainer).

## Score Interpretation

- **Lower scores are better**: A low correlation means the explanations change significantly when model parameters are randomized, indicating the explainer properly depends on the learned weights.
- Values range from -1 to 1, where 1 means perfectly correlated explanations.
- High correlation values suggest the explainer may not be faithfully using model information.

## Randomization Strategies

The metric supports different randomization strategies via `ModelRandomizationStrategy`:

### ProgressiveLayerRandomization

Randomizes model weights layer-by-layer, starting from the output layers (by default) and progressing toward the input layers.

```python
from xplique.metrics import ProgressiveLayerRandomization

# Randomize top 25% of layers (default)
strategy = ProgressiveLayerRandomization(stop_layer=0.25)

# Randomize up to a specific layer
strategy = ProgressiveLayerRandomization(stop_layer='conv2')

# Randomize from input toward output
strategy = ProgressiveLayerRandomization(stop_layer=3, reverse=False)
```

## Example

```python
from xplique.metrics import ModelRandomizationMetric, ProgressiveLayerRandomization
from xplique.attributions import Saliency

# load images, labels and model
# ...
explainer = Saliency(model)

# Use default strategy (randomize top 25% of layers)
metric = ModelRandomizationMetric(model, inputs, labels)
score = metric.evaluate(explainer)

# Or specify a custom strategy
strategy = ProgressiveLayerRandomization(stop_layer=0.5)
metric = ModelRandomizationMetric(model, inputs, labels, randomization_strategy=strategy)
score = metric.evaluate(explainer)
```

!!!warning
    This metric clones the model internally for randomization. The original model weights are preserved.

{{xplique.metrics.randomization.ModelRandomizationMetric}}

[^1]: [Sanity Checks for Saliency Maps (2018)](https://arxiv.org/abs/1810.03292)

