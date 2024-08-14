# KLEOR

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub> [View colab tutorial](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub> [View source](https://github.com/deel-ai/xplique/blob/master/xplique/example_based/semifactuals.py) |
📰 [Paper](https://www.researchgate.net/publication/220106308_KLEOR_A_Knowledge_Lite_Approach_to_Explanation_Oriented_Retrieval)

KLEOR for Knowledge-Light Explanation-Oriented Retrieval was introduced by Cummins & Bridge in 2006. It is a method that use counterfactuals, Nearest Unlike Neighbor (NUN), to guide the selection of a semi-factual (SF) example.

Given a distance function $dist$, the NUN of a sample $(x, y)$ is the closest sample in the training dataset which has a different label than $y$.

The KLEOR method actually have three variants including:

- The Sim-Miss approach
- The Global-Sim approach

In the Sim-Miss approach, the SF of the sample $(x,y)$ is the closest training sample from the corresponding NUN which has the same label as $y$.

Denoting the training dataset as $\mathcal{D}$:

$$Sim-Miss(x, y, NUN(x,y), \mathcal{D}) = arg \\ min_{(x',y') \in \mathcal{D} \\ | \\ y'=y} dist(x', NUN(x,y))$$

In the Global-Sim approach, they add an additional constraint that the SF should lie between the sample $(x,y)$ and the NUN that is: $dist(x, SF) < dist(x, NUN(x,y))$.

We extended to the $k$ nearest neighbors of the NUN for both approaches.

!!!info
    In our implementation, we rather consider the labels predicted by the model $\hat{y}$ (*i.e.* the targets) rather than $y$!

!!!tips
    As KLEOR methods use counterfactuals, they can also return them. Therefore, it is possible to obtain both semi-factuals and counterfactuals with an unique method. To do so "nuns" and "nuns_labels" should be added to the `cases_returns` list.

## Examples

```python
from xplique.example_based import KLEORSimMiss

# loading
cases_dataset = ... # load the training dataset
targets = ... # load the one-hot encoding of predicted labels of the training dataset

# parameters
k = 5
distance = "euclidean"
case_returns = ["examples", "nuns"]

# instantiate the KLEOR object
kleor_sim_miss = KLEORSimMiss(cases_dataset=cases_dataset,
                              targets_dataset=targets,
                              k=k,
                              distance=distance,
                             )

# load the test samples and targets
test_samples = ... # load the test samples to search for
test_targets = ... # load the one-hot encoding of the test samples' predictions

# search the SFs for the test samples
sim_miss_sf = kleor_sim_miss.explain(test_samples, test_targets)

# get the semi-factuals
semifactuals = global_sim_sf["examples"]

# get the counterfactuals
counterfactuals = global_sim_sf["nuns"]
```

```python
from xplique.example_based import KLEORGlobalSim

# loading
cases_dataset = ... # load the training dataset
targets = ... # load the one-hot encoding of predicted labels of the training dataset

# parameters
k = 5
distance = "euclidean"
case_returns = ["examples", "nuns"]

# instantiate the KLEOR object
kleor_global_sim = KLEORGlobalSim(cases_dataset=cases_dataset,
                                  targets_dataset=targets,
                                  k=k,
                                  distance=distance,
                                  case_returns=case_returns,
                                 )

# load the test samples and targets
test_samples = ... # load the test samples to search for
test_targets = ... # load the one-hot encoding of the test samples' predictions

# search the SFs for the test samples
global_sim_sf = kleor_global_sim.explain(test_samples, test_targets)

# get the semi-factuals
semifactuals = global_sim_sf["examples"]

# get the counterfactuals
counterfactuals = global_sim_sf["nuns"]
```

## Notebooks

- [**Example-based Methods**: Getting started](https://colab.research.google.com/drive/1gA7mhWhWzdKholZWkTvAg4FzFnzS8NHF)

{{xplique.example_based.semifactuals.KLEORSimMiss}}
{{xplique.example_based.semifactuals.KLEORGlobalSim}}