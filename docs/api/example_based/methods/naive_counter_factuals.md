# Naive Counterfactuals

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub> [View colab tutorial]()**WIP** |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub> [View source](https://github.com/deel-ai/xplique/blob/master/xplique/example_based/counterfactuals.py) |
📰 [Paper](https://www.semanticscholar.org/paper/Nearest-unlike-neighbor-(NUN)%3A-an-aid-to-decision-Dasarathy/48c1a310f655b827e5e7d712c859b25a4e3c0902)

!!!note
    The paper referenced here is not exactly the one we implemented as we a "naive" of it. However, it is probably the closest in essence of what we implemented.

We define here a "naive" counterfactual method that is based on the Nearest Unlike Neighbor (NUN) concept introduced by Dasarathy in 1991[^1]. In essence, the NUN of a sample $(x, y)$ is the closest sample in the training dataset which has a different label than $y$.

Thus, in this naive approach to counterfactuals, we yield the $k$ nearest training instances that have a different label than the target of the input sample in a greedy fashion. 

As it is mentioned in the [API documentation](api/example_based/methods/api_example_based/), by setting a `Projection` object, one can use the projection space to compute the distances between the samples (e.g. search for the CF in the latent space of a model).

## Example

```python
from xplique.example_based import NaiveCounterfactuals

# load the training dataset
cases_dataset = ... # load the training dataset
targets_dataset = ... # load the targets of the training dataset

k = 5

# instantiate the NaiveCounterfactuals object
ncf = NaiveCounterfactuals(cases_dataset=cases_dataset,
                           targets_dataset=targets_dataset,
                           k=k,
                          )

# load the test samples and targets
test_samples = ... # load the test samples to search for
test_targets = ... # load the targets of the test samples

# search the CFs for the test samples
counterfactuals = ncf.explain(test_samples, test_targets)
```

## Notebooks

TODO: Add notebooks

{{xplique.example_based.counterfactuals.NaiveCounterfactuals}}

[^1] [Nearest unlike neighbor (NUN): an aid to decision making](https://www.semanticscholar.org/paper/Nearest-unlike-neighbor-(NUN)%3A-an-aid-to-decision-Dasarathy/48c1a310f655b827e5e7d712c859b25a4e3c0902)