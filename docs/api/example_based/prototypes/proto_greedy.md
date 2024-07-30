# ProtoGreedySearch

<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" width="20">
</sub>[View colab tutorial](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X) |
<sub>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="20">
</sub>[View source](https://github.com/deel-ai/xplique/blob/antonin/example-based-merge/xplique/example_based/search_methods/proto_greedy_search.py) |
📰 [Paper](https://arxiv.org/abs/1707.01212)

`ProtoGreedySearch` associated non-negative weights to prototypes which are indicative of their importance. This approach allows for identifying both prototypes and criticisms (the least weighted examples among prototypes) by maximmizing the same weighted objective function.

!!! quote
    Our work notably generalizes the recent work
    by Kim et al. (2016) where in addition to selecting prototypes, we
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

[Gurumoorthy et al., 2019](https://arxiv.org/abs/1707.01212) demonstrate that this problem is weakly submodular, which immediately leads to a standard greedy algorithm which they call `ProtoGreedy`.

`ProtoGreedy` is algorithmically similar to greedy algorithm used by [Kim et al., 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf) where both the methods greedily select the next element that maximizes the increment of the scoring function.

## Example

```python
from xplique.example_based import ProtoGreedy

# load data and labels
# ...

explainer = ProtoGreedy(cases_dataset, labels_dataset, targets_dataset, k, 
                   projection, case_returns, batch_size, distance, 
                   nb_prototypes, kernel_type, 
                   kernel_fn, gamma)
# compute global explanation
global_prototypes = explainer.get_global_prototypes()
# compute local explanation
local_prototypes = explainer(inputs)
```

## Notebooks

- [**Prototypes**: Getting started](https://colab.research.google.com/drive
/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2)
- [**ProtoGreedy**: Going Further](https://colab.research.google.com/drive/1nsB7xdQbU0zeYQ1-aB_D-M67-RAnvt4X)


{{xplique.example_based.search_methods.ProtoGreedySearch}}

[^1]: [Visual Explanations from Deep Networks via Gradient-based Localization (2016).](https://arxiv.org/abs/1610.02391)