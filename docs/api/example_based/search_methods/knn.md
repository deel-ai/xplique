# K Nearest Neighbors

KNN method to search examples. Based on `sklearn.neighbors.NearestNeighbors` [see the documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html).
The kneighbors method is implemented in a batched way to handle large datasets and try to be memory efficient.

In addition, we also added a `FilterKNN` class that allows to filter the neighbors based on a given criterion avoiding potentially a compute of the distances for all the samples. It is useful when the candidate neighbors are sparse and the distance computation is expensive.

## Examples

```python
from xplique.example_based.search_methods import ORDER
from xplique.example_based.search_methods import KNN

# set some parameters
k = 5
cases_dataset = ... # load the training dataset
test_samples = ... # load the test samples to search for

distance = "euclidean"
order = ORDER.ASCENDING

# create the KNN object
knn = KNN(cases_dataset = cases_dataset
          k = k,
          distance = distance,
          order = order)

k_nearest_neighbors = knn.kneighbors(test_samples)
```

```python
from xplique.example_based.search_methods import ORDER
from xplique.example_based.search_methods import FilterKNN

# set some parameters
k = 5
cases_dataset = ... # load the training dataset
targets = ... # load the targets of the training dataset

test_samples = ... # load the test samples to search for
test_targets = ... # load the targets of the test samples

distance = "euclidean"
order = ORDER.ASCENDING

# define a filter function
def filter_fn(cases, inputs, targets, cases_targets):
    # filter the cases that have the same target as the input
    mask = tf.not_equal(targets, cases_targets)
    return mask

# create the KNN object
filter_knn = FilterKNN(cases_dataset=cases_dataset,
                targets_dataset=targets,
                k=k,
                distance=distance,
                order=order,
                filter_fn=filter_fn)

k_nearest_neighbors = filter_knn.kneighbors(test_samples, test_targets)
```

## Notebooks

TODO: add all notebooks that use this search method

{{xplique.example_based.search_methods.knn.KNN}}
{{xplique.example_based.search_methods.knn.FilterKNN}}