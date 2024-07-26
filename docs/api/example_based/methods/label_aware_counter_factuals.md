# Label Aware Counterfactuals

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub> [View colab tutorial]()**WIP** |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub> [View source](https://github.com/deel-ai/xplique/blob/master/xplique/example_based/counterfactuals.py) |
📰 [Paper](https://www.semanticscholar.org/paper/Nearest-unlike-neighbor-(NUN)%3A-an-aid-to-decision-Dasarathy/48c1a310f655b827e5e7d712c859b25a4e3c0902)

!!!note
    The paper referenced here is not exactly the one we implemented. However, it is probably the closest in essence of what we implemented.

In contrast to the [Naive Counterfactuals](api/example_based/methods/naive_counter_factuals/) approach, the Label Aware Counterfactuals leverage an *a priori* knowledge of the Counterfactuals' (CFs) targets to guide the search for the CFs (*e.g.* one is looking for a CF of the digit 8 in MNIST dataset within the digit 0 instances).

!!!warning
    Consequently, for this class, when a user call the `explain` method, the user is not expected to provide the targets corresponding to the input samples but rather a one-hot encoding of the targets of the CFs to search for.

!!!info
    One can use the `Projection` object to compute the distances between the samples (e.g. search for the CF in the latent space of a model).

## Example

```python
from xplique.example_based import LabelAwareCounterfactuals

# load the training dataset
cases_dataset = ... # load the training dataset
targets_dataset = ... # load the targets of the training dataset

k = 5

# instantiate the LabelAwareCounterfactuals object
lacf = LabelAwareCounterfactuals(cases_dataset=cases_dataset,
                                targets_dataset=targets_dataset,
                                k=k,
                               )

# load the test samples
test_samples = ... # load the test samples to search for
test_cf_targets = ... # WARNING: provide the one-hot encoding of the targets of the CFs to search for

# search the CFs for the test samples
counterfactuals = lacf.explain(test_samples, test_cf_targets)
```

## Notebooks

TODO: Add notebooks

{{xplique.example_based.counterfactuals.LabelAwareCounterfactuals}}