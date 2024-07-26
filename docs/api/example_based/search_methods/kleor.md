# KLEOR Search Methods

Those search methods are used for the [KLEOR](api/example_based/methods/kleor/) methods.

It encompasses the two following classes:
- `KLEORSimMissSearch`: looks for Semi-Factuals examples by searching for the Nearest Unlike Neighbor (NUN) of the query. The NUN is the closest example to the query that has a different prediction than the query. Then, the method searches for the K-Nearest Neighbors (KNN) of the NUN that have the same prediction as the query.
- `KLEORGlobalSim`: in addition to the previous method, the SF should be closer to the query than the NUN to be a candidate.

## Examples

```python
from xplique.example_based.search_methods import KLEORSimMissSearch
from xplique.example_based.search_methods import KLEORGlobalSim

cases_dataset = ... # load the training dataset
targets = ... # load the targets of the training dataset

test_samples = ... # load the test samples to search for
test_targets = ... # load the targets of the test samples

# set some parameters
k = 5
distance = "euclidean"

# create the KLEORSimMissSearch object
kleor_sim_miss_search = KLEORSimMissSearch(cases_dataset=cases_dataset,
                                           targets_dataset=targets,
                                           k=k,
                                           distance=distance)

# create the KLEORGlobalSim object
kleor_global_sim = KLEORGlobalSim(cases_dataset=cases_dataset,
                                   targets_dataset=targets,
                                   k=k,
                                   distance=distance)

# search for the K-Nearest Neighbors of the test samples
sim_miss_neighbors = kleor_sim_miss_search.find_examples(test_samples, test_targets)
global_sim_neighbors = kleor_global_sim.find_examples(test_samples, test_targets)
```

## Notebooks

TODO: add the notebook for KLEOR

{{xplique.example_based.search_methods.kleor.KLEORSimMissSearch}}
{{xplique.example_based.search_methods.kleor.KLEORGlobalSim}}
