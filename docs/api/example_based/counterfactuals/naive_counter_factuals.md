# Naive CounterFactuals

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub> [View colab tutorial](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub> [View source](https://github.com/deel-ai/xplique/blob/master/xplique/example_based/counterfactuals.py) |
📰 [Paper](https://www.semanticscholar.org/paper/Nearest-unlike-neighbor-(NUN)%3A-an-aid-to-decision-Dasarathy/48c1a310f655b827e5e7d712c859b25a4e3c0902)

!!!note
    The paper referenced here is not exactly the one we implemented as we use a "naive" version of it. However, it is probably the closest in essence of what we implemented.

We define here a "naive" counterfactual method that is based on the Nearest Unlike Neighbor (NUN) concept introduced by Dasarathy in 1991[^1]. In essence, the NUN of a sample $(x, y)$ is the closest sample in the training dataset which has a different label than $y$.

Thus, in this naive approach to counterfactuals, we yield the $k$ nearest training instances that have a different label than the target of the input sample in a greedy fashion. 

As it is mentioned in the [API documentation](../../api_example_based/), by setting a `Projection` object, one will map the inputs to a space where the distance function is meaningful.

## Example

```python
from xplique.example_based import NaiveCounterFactuals
from xplique.example_based.projections import LatentSpaceProjection

# load the training dataset and the model
cases_dataset = ... # load the training dataset
targets_dataset = ... # load the one-hot encoding of predicted labels of the training dataset
model = ...

# load the test samples
test_samples = ... # load the test samples to search for
test_targets = ... # compute a one hot encoding of the model's prediction on the samples

# parameters
k = 5  # number of example for each input
case_returns = "all"  # elements returned by the explain function
distance = "euclidean"
latent_layer = "last_conv"  # where to split your model for the projection

# construct a projection with your model
projection = LatentSpaceProjection(model, latent_layer=latent_layer)

# instantiate the NaiveCounterFactuals object
ncf = NaiveCounterFactuals(
    cases_dataset=cases_dataset,
    targets_dataset=targets_dataset,
    k=k,
    projection=projection,
    case_returns=case_returns,
    distance=distance,
)

# search the CFs for the test samples
output_dict = ncf.explain(
    inputs=test_samples,
    targets=test_targets,
)
```

## Notebooks

- [**Example-based Methods**: Getting started](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF)

{{xplique.example_based.counterfactuals.NaiveCounterFactuals}}

[^1] [Nearest unlike neighbor (NUN): an aid to decision making](https://www.semanticscholar.org/paper/Nearest-unlike-neighbor-(NUN)%3A-an-aid-to-decision-Dasarathy/48c1a310f655b827e5e7d712c859b25a4e3c0902)