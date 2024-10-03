# ProtoGreedy

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/antonin/example-based-merge/xplique/example_based/search_methods/proto_greedy_search.py) |
📰 [Paper](https://arxiv.org/abs/1707.01212)

`ProtoGreedy` associated non-negative weights to prototypes which are indicative of their importance. This approach allows for identifying both prototypes and criticisms (the least weighted examples among prototypes) by maximizing the same weighted objective function.

!!! quote
    Our work notably generalizes the recent work
    by [Kim et al. (2016)](../mmd_critic/)) where in addition to selecting prototypes, we
    also associate non-negative weights which are indicative of their
    importance. This extension provides a single coherent framework
    under which both prototypes and criticisms (i.e. outliers) can be
    found. Furthermore, our framework works for any symmetric
    positive definite kernel thus addressing one of the key open
    questions laid out in Kim et al. (2016).

    -- <cite>[Efficient Data Representation by Selecting Prototypes with Importance Weights (2019).](https://arxiv.org/abs/1707.01212)</cite>

More precisely, the weighted objective $F(\mathcal{P},w)$ is defined as:
\begin{equation}   
F(\mathcal{P},w)=\frac{2}{n}\sum_{i,j=1}^{|\mathcal{P}|,n}w_i\kappa(p_i,x_j)-\sum_{i,j=1}^{|\mathcal{P}|}w_iw_j\kappa(p_i,p_j),
\end{equation}
where $w$ are non-negative weights for each prototype. The problem then consist on finding a subset $\mathcal{P}$ with a corresponding $w$ that maximizes $J(\mathcal{P}) \equiv \max_{w:supp(w)\in \mathcal{P},w\ge 0} J(\mathcal{P},w)$ s.t. $|\mathcal{P}| \leq m=m_p+m_c$. 

!!!info
    For ProtoGreedy, any kernel can be used, as these methods rely on weak submodularity instead of full submodularity.


## Example

```python
from xplique.example_based import ProtoGreedy
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

protogreedy = ProtoGreedy(
    cases_dataset=cases_dataset,
    nb_global_prototypes=nb_global_prototypes,
    nb_local_prototypes=nb_local_prototypes,
    projection=projection,
    case_returns=case_returns,
)

# compute global explanation
global_prototypes = protogreedy.get_global_prototypes()

# compute local explanation
local_prototypes = protogreedy.explain(test_samples)
```

## Notebooks

- [**Prototypes**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**ProtoGreedy**: Going Further](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X)


{{xplique.example_based.prototypes.ProtoGreedy}}

[^1]: [Visual Explanations from Deep Networks via Gradient-based Localization (2016).](https://arxiv.org/abs/1610.02391)
