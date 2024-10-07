# Label Aware Counterfactuals

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub> [View colab tutorial](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub> [View source](https://github.com/deel-ai/xplique/blob/master/xplique/example_based/counterfactuals.py) |
📰 [Paper](https://www.semanticscholar.org/paper/Nearest-unlike-neighbor-(NUN)%3A-an-aid-to-decision-Dasarathy/48c1a310f655b827e5e7d712c859b25a4e3c0902)

!!!note
    The paper referenced here is not exactly the one we implemented. However, it is probably the closest in essence of what we implemented.

In contrast to the [Naive Counterfactuals](../../counterfactuals/naive_counter_factuals/) approach, the Label Aware CounterFactuals leverage an *a priori* knowledge of the Counterfactuals' (CFs) targets to guide the search for the CFs (*e.g.* one is looking for a CF of the digit 8 in MNIST dataset within the digit 0 instances).

!!!warning
    Consequently, for this class, when a user call the `explain` method, the user is expected to provide both the `targets` corresponding to the input samples and `cf_expected_classes` a one-hot encoding of the label expected for the CFs. But in most cases, the `targets` can be set to `None` as they are computed internally by projections.

!!!info
    One can use the `Projection` object to compute the distances between the samples (e.g. search for the CF in the latent space of a model).

## Example

```python
from xplique.example_based import LabelAwareCounterFactuals
from xplique.example_based.projections import LatentSpaceProjection

# load the training dataset and the model
cases_dataset = ... # load the training dataset
targets_dataset = ... # load the one-hot encoding of predicted labels of the training dataset
model = ...

# load the test samples
test_samples = ... # load the test samples to search for
test_cf_expacted_classes = ... # WARNING: provide the one-hot encoding of the expected label of the CFs

# parameters
k = 5  # number of example for each input
case_returns = "all"  # elements returned by the explain function
distance = "euclidean"
latent_layer = "last_conv"  # where to split your model for the projection

# construct a projection with your model
projection = LatentSpaceProjection(model, latent_layer=latent_layer)

# instantiate the LabelAwareCounterfactuals object
lacf = LabelAwareCounterFactuals(
    cases_dataset=cases_dataset,
    targets_dataset=targets_dataset,
    k=k,
    projection=projection,
    case_returns=case_returns,
    distance=distance,
)

# search the CFs for the test samples
output_dict = lacf.explain(
    inputs=test_samples,
    targets=None,  # not necessary for this projection
    cf_expected_classes=test_cf_expacted_classes,
)
```

## Notebooks

- [**Example-based Methods**: Getting started](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF)

{{xplique.example_based.counterfactuals.LabelAwareCounterFactuals}}