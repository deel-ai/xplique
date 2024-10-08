# Similar-Examples

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub> [View colab tutorial](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub> [View source](https://github.com/deel-ai/xplique/blob/master/xplique/example_based/similar_examples.py)

We designate here as *Similar Examples* all methods that given an input sample, search for the most similar **training** samples given a distance function `distance`. Furthermore, one can define the search space using a `projection` function (see [Projections](../../projections/)). This function should map an input sample to the search space where the distance function is defined and meaningful (**e.g.** the latent space of a Convolutional Neural Network).
Then, a K-Nearest Neighbors (KNN) search is performed to find the most similar samples in the search space.

## Example

```python
from xplique.example_based import SimilarExamples

cases_dataset = ... # load the training dataset
targets = ... # load the one-hot encoding of predicted labels of the training dataset

# parameters
k = 5
distance = "euclidean"
case_returns = ["examples", "nuns"]

# define the projection function
def custom_projection(inputs: tf.Tensor, np.ndarray, targets: tf.Tensor, np.ndarray = None):
    '''
    Example of projection,
    inputs are the elements to project.
    targets are optional parameters to orientate the projection.
    '''
    projected_inputs = # do some magic on inputs, it should use the model.
    return projected_inputs

# instantiate the SimilarExamples object
sim_ex = SimilarExamples(
    cases_dataset=cases_dataset,
    targets_dataset=targets,
    k=k,
    projection=custom_projection,
    distance=distance,
)

# load the test samples and targets
test_samples = ... # load the test samples to search for
test_targets = ... # load the one-hot encoding of the test samples' predictions

# search the most similar samples with the SimilarExamples method
similar_samples = sim_ex.explain(test_samples, test_targets)
```

# Notebooks

- [**Example-based Methods**: Getting started](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF)

{{xplique.example_based.similar_examples.SimilarExamples}}