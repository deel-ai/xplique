# COLE: Contributions Oriented Local Explanations

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub> [View colab tutorial]()**WIP** |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub> [View source](https://github.com/deel-ai/xplique/blob/master/xplique/example_based/similar_examples.py) |
📰 [Paper](https://researchrepository.ucd.ie/handle/10197/11064)

COLE for Contributions Oriented Local Explanations was introduced by Kenny & Keane in 2019.

!!! quote
    Our method COLE is based on the premise that the contributions of features in a model’s classification represent the most sensible basis to inform case-based explanations.

    -- <cite>[COLE paper](https://researchrepository.ucd.ie/handle/10197/11064)</cite>[^1]

The core idea of the COLE approach is to use [attribution maps](api/attributions/api_attributions/) to define a relevant search space for the K-Nearest Neighbors (KNN) search.

More specifically, the COLE approach is based on the following steps:
- (1) Given an input sample $x$, compute the attribution map $A(x)$
- (2) Consider the projection space defined by: $p: x \rightarrow A(x) \odot x$ ($\odot$ denotes the element-wise product)
- (3) Perform a KNN search in the projection space to find the most similar training samples

!!! info
    In the original paper, the authors focused on Multi-Layer Perceptrons (MLP) and three attribution methods (LPR, Integrated Gradient, and DeepLift). We decided to implement a COLE method that generalizes to a more broader range of Neural Networks and attribution methods that are gradient-based (see [API Attributions documentation](api/attributions/api_attributions/) for definition).

## Example

```python
from xplique.example_based import Cole
from xplique.attributions import Saliency

model = ... # load the model
cases_dataset = ... # load the training dataset
target_dataset = ... # load the target dataset (predicted one-hot encoding of model's predictions)
k = 5

# instantiate the Cole object
cole = Cole(
    cases_dataset=cases_dataset,
    model=model,
    k=k,
    attribution_method=Saliency,
)

# load the test samples
test_samples = ... # load the test samples
test_targets = ... # load the test targets

# search the most similar samples with the COLE method
similar_samples = cole.explain(test_samples, test_targets)
```

## Notebooks

TODO: Add the notebook

{{xplique.example_based.similar_examples.Cole}}

[^1]: [Twin-Systems to Explain Artificial Neural Networks using Case-Based Reasoning:
Comparative Tests of Feature-Weighting Methods in ANN-CBR Twins for XAI (2019)](https://researchrepository.ucd.ie/handle/10197/11064)