# MMDCritic

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/antonin/example-based-merge/xplique/example_based/search_methods/proto_greedy_search.py) |
📰 [Paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf)

`MMDCritic` finds prototypes and criticisms by maximizing two separate objectives based on the Maximum Mean Discrepancy (MMD).

!!! quote
    MMD-critic uses the MMD statistic as a measure of similarity between points and potential prototypes, and
    efficiently selects prototypes that maximize the statistic. In addition to prototypes, MMD-critic selects criticism samples i.e. samples that are not well-explained by the prototypes using a regularized witness function score.

    -- <cite>[Efficient Data Representation by Selecting Prototypes with Importance Weights (2019).](https://arxiv.org/abs/1707.01212)</cite>

First, to find prototypes $\mathcal{P}$, a greedy algorithm is used to maximize $F(\mathcal{P})$ s.t. $|\mathcal{P}| \le m_p$ where $F(\mathcal{P})$ is defined as:
\begin{equation}
    F(\mathcal{P})=\frac{2}{|\mathcal{P}|\cdot n}\sum_{i,j=1}^{|\mathcal{P}|,n}\kappa(p_i,x_j)-\frac{1}{|\mathcal{P}|^2}\sum_{i,j=1}^{|\mathcal{P}|}\kappa(p_i,p_j),
\end{equation}
where $m_p$ the number of prototypes to be found. They used diagonal dominance conditions on the kernel to ensure monotonocity and submodularity of $F(\mathcal{P})$. 

Second, to find criticisms $\mathcal{C}$, the same greedy algorithm is used to select points that maximize another objective function $J(\mathcal{C})$. 

!!!warning
    For `MMDCritic`, the kernel must satisfy a condition that ensures the submodularity of the set function. The Gaussian kernel meets this requirement and it is recommended. If you wish to choose a different kernel, it must satisfy the condition described by [Kim et al., 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf).


## Example

```python
from xplique.example_based import MMDCritic
from xplique.example_based.projections import LatentSpaceProjection

# load the training dataset and the model
cases_dataset = ... # load the training dataset
model = ...

# load the test samples
test_samples = ... # load the test samples to search for

# parameters
case_returns = "all"  # elements returned by the explain function
latent_layer = "last_conv"  # where to split your model for the projection
nb_global_prototypes = 5
nb_local_prototypes = 1
kernel_fn = None  # the default rbf kernel will be used, the distance will be based on this

# construct a projection with your model
projection = LatentSpaceProjection(model, latent_layer=latent_layer)

mmd = MMDCritic(
    cases_dataset=cases_dataset,
    nb_global_prototypes=nb_global_prototypes,
    nb_local_prototypes=nb_local_prototypes,
    projection=projection,
    case_returns=case_returns,
)

# compute global explanation
global_prototypes = mmd.get_global_prototypes()

# compute local explanation
local_prototypes = mmd.explain(test_samples)
```

## Notebooks

- [**Example-based: Prototypes**](https://colab.research.google.com/drive/1-bUvXxzWrBqLLfS_4TvErcEfyzymTVGz)


{{xplique.example_based.prototypes.MMDCritic}}

[^1]: [Visual Explanations from Deep Networks via Gradient-based Localization (2016).](https://arxiv.org/abs/1610.02391)

